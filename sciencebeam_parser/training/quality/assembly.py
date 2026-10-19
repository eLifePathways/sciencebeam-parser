"""Joining the record generation wrote with the counts only assembly can take.

Generation sees the JATS and the TEI; the delft conversion sees the TEI and the
labels it produces, and is the only place the last count exists. It reads what
generation recorded rather than extending it, so that generated output stays
reproducible from generation alone and can be assembled more than once.
"""
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from sciencebeam_parser.training.quality.gate import QualityVerdict
from sciencebeam_parser.training.quality.review import ReviewStanding, read_review_standing
from sciencebeam_parser.utils.io import auto_uploading_output_file, glob


LOGGER = logging.getLogger(__name__)


@dataclass
class GeneratedDocumentRecord:
    """A row generation wrote, which corpus it came from, and where review left it.

    The review is read beside the record rather than being part of it: generation
    measures and a reader judges, at different times and into different files.
    """
    document_id: str
    corpus: Optional[str]
    json_dict: Mapping[str, Any]
    review: str = ReviewStanding.UNREVIEWED

    @property
    def labelled_content_hash(self) -> Optional[str]:
        """What a verdict about this document has to be pinned to."""
        return self.json_dict.get('labelled_content_hash')

    @property
    def jats_reference_count(self) -> Optional[int]:
        return self.json_dict.get('jats', {}).get('reference_count')

    @property
    def entity_element_count(self) -> Optional[int]:
        return self.json_dict.get('entity_element_count')

    @property
    def written(self) -> Optional[bool]:
        return self.json_dict.get('written')

    @property
    def jats_status(self) -> Optional[str]:
        return self.json_dict.get('jats', {}).get('status')

    @property
    def has_generated_output(self) -> bool:
        """Whether generation wrote a file for this document at all.

        A row with no `written` at all is a document that failed or timed out.
        """
        return self.json_dict.get('written') is True


def get_corpus_name_for_record_file_path(record_file_path: str) -> Optional[str]:
    """The corpus a record belongs to, which is the directory above its model's.

    A record sits in its model's `quality` directory, so the corpus is three
    levels up rather than two.
    """
    records_directory = os.path.dirname(record_file_path)
    model_directory = os.path.dirname(records_directory)
    corpus_directory = os.path.dirname(model_directory)
    return os.path.basename(corpus_directory) or None


def read_generated_document_records(
    record_path_pattern: str
) -> Dict[str, GeneratedDocumentRecord]:
    """Read the records a pattern matches, one JSON object per file."""
    record_file_list = glob(record_path_pattern)
    if not record_file_list:
        raise RuntimeError(
            'no quality record found for file pattern %r' % record_path_pattern
        )
    LOGGER.info('reading %d quality records', len(record_file_list))
    LOGGER.debug('quality record files: %r', record_file_list)
    record_by_document_id: Dict[str, GeneratedDocumentRecord] = {}
    for record_file_path in record_file_list:
        corpus = get_corpus_name_for_record_file_path(record_file_path)
        with open(record_file_path, 'r', encoding='utf-8') as record_file:
            json_dict = json.load(record_file)
        record_by_document_id[json_dict['document_id']] = GeneratedDocumentRecord(
            document_id=json_dict['document_id'],
            corpus=corpus,
            json_dict=json_dict,
            review=read_review_standing(
                record_file_path, json_dict.get('labelled_content_hash')
            ),
        )
    return record_by_document_id


@dataclass
class AssembledDocumentRecord:
    """What assembly measured for one document, beside what generation recorded."""
    document_id: str
    model_name: str
    corpus: Optional[str] = None
    sequence_count: int = 0
    matched_row_ratio: Optional[float] = None
    matched_row_count: int = 0
    unmatched_row_count: int = 0
    uncovered_row_count: int = 0
    unrecognised_path_counts: Optional[Dict[str, int]] = None
    entity_start_count: Optional[int] = None
    label_start_counts: Optional[Dict[str, int]] = None
    generated: Optional[GeneratedDocumentRecord] = None
    verdict: Optional['QualityVerdict'] = None
    review: str = ReviewStanding.UNREVIEWED

    @property
    def entity_element_count(self) -> Optional[int]:
        return self.generated.entity_element_count if self.generated else None

    @property
    def lost_at_parse(self) -> Optional[int]:
        """Entities the parse did not return for an element the TEI holds."""
        element_count = self.entity_element_count
        if element_count is None or self.entity_start_count is None:
            return None
        return element_count - self.entity_start_count

    def to_json_dict(self) -> Dict[str, Any]:
        json_dict: Dict[str, Any] = {
            'document_id': self.document_id,
            'model': self.model_name,
            'corpus': self.corpus,
            'sequence_count': self.sequence_count,
            # Recorded for every document, so that a rejection a regeneration
            # lapsed is visible rather than only absent.
            'review': self.review,
        }
        if self.matched_row_ratio is not None:
            # What the pairing did with the raw file, for a model read with one.
            # `uncovered` is the tail a TEI stops short of, counted apart from
            # the rows it had lines for and could not place.
            json_dict['pairing'] = {
                'matched_row_ratio': round(self.matched_row_ratio, 4),
                'matched_row_count': self.matched_row_count,
                'unmatched_row_count': self.unmatched_row_count,
                'uncovered_row_count': self.uncovered_row_count,
            }
        if self.unrecognised_path_counts:
            json_dict['unrecognised_element_paths'] = dict(
                sorted(self.unrecognised_path_counts.items())
            )
        if self.entity_start_count is not None:
            json_dict['entity_start_count'] = self.entity_start_count
        if self.label_start_counts:
            json_dict['label_start_counts'] = self.label_start_counts
        if self.generated is not None:
            json_dict['generated'] = dict(self.generated.json_dict)
        if self.verdict is not None:
            json_dict['excluded'] = self.verdict.is_excluded
            if self.verdict.is_excluded:
                json_dict['exclusion_reasons'] = list(self.verdict.exclusion_reasons)
                json_dict['exclusion_detail'] = dict(self.verdict.detail)
        return json_dict


@dataclass
class CorpusAssemblySummary:
    corpus: Optional[str]
    document_count: int = 0
    sequence_count: int = 0
    entity_element_count: Optional[int] = None
    entity_start_count: Optional[int] = None
    documents_losing_entities: List[str] = field(default_factory=list)
    documents_without_generated_record: List[str] = field(default_factory=list)
    matched_row_count: int = 0
    unmatched_row_count: int = 0
    uncovered_row_count: int = 0
    documents_with_unmatched_rows: List[str] = field(default_factory=list)
    unrecognised_path_count: int = 0
    documents_with_unrecognised_paths: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        parts = [
            f'{self.document_count} documents',
            f'{self.sequence_count} sequences',
        ]
        # A model counted by label has no entity count, and reporting it as zero
        # would read as every entity lost.
        if self.entity_start_count is not None:
            parts.append(
                f'{self.entity_start_count} entities against '
                f'{self.entity_element_count or 0} elements'
            )
        if self.documents_losing_entities:
            parts.append(
                f'{len(self.documents_losing_entities)} documents lost entities at '
                f'the parse: {sorted(self.documents_losing_entities)}'
            )
        if self.documents_without_generated_record:
            parts.append(
                f'{len(self.documents_without_generated_record)} documents with no '
                f'record from generation'
            )
        covered = self.matched_row_count + self.unmatched_row_count
        if covered:
            parts.append(
                f'{self.matched_row_count}/{covered} rows paired with a TEI line '
                f'({self.matched_row_count / covered:.4f}), '
                f'{len(self.documents_with_unmatched_rows)} documents short of all '
                f'of theirs'
            )
        if self.uncovered_row_count:
            parts.append(
                f'{self.uncovered_row_count} rows after a TEI stopped short'
            )
        # Zero for a TEI we wrote ourselves, so anything else is a finding about
        # the corpus being read rather than routine.
        if self.unrecognised_path_count:
            parts.append(
                f'{self.unrecognised_path_count} lines labelled by an enclosing '
                f'element, over {len(self.documents_with_unrecognised_paths)} documents'
            )
        return '; '.join(parts)


def get_assembly_summary_by_corpus(
    assembled_records: Sequence[AssembledDocumentRecord]
) -> Dict[Optional[str], CorpusAssemblySummary]:
    summary_by_corpus: Dict[Optional[str], CorpusAssemblySummary] = {}
    for record in assembled_records:
        summary = summary_by_corpus.setdefault(
            record.corpus, CorpusAssemblySummary(corpus=record.corpus)
        )
        summary.document_count += 1
        summary.sequence_count += record.sequence_count
        if record.entity_start_count is not None:
            summary.entity_start_count = (
                (summary.entity_start_count or 0) + record.entity_start_count
            )
            summary.entity_element_count = (
                (summary.entity_element_count or 0) + (record.entity_element_count or 0)
            )
        summary.matched_row_count += record.matched_row_count
        summary.unmatched_row_count += record.unmatched_row_count
        summary.uncovered_row_count += record.uncovered_row_count
        if record.unmatched_row_count:
            summary.documents_with_unmatched_rows.append(record.document_id)
        if record.unrecognised_path_counts:
            summary.unrecognised_path_count += sum(
                record.unrecognised_path_counts.values()
            )
            summary.documents_with_unrecognised_paths.append(record.document_id)
        if record.lost_at_parse:
            summary.documents_losing_entities.append(record.document_id)
        if record.generated is None:
            summary.documents_without_generated_record.append(record.document_id)
    return summary_by_corpus


def get_document_ids_without_generated_output(
    record_by_document_id: Mapping[str, GeneratedDocumentRecord]
) -> List[str]:
    """Documents generation recorded as having produced no file.

    Taken from the record's own `written`, not from what the training data is
    missing: assembly is often pointed at part of a corpus, and inferring this
    from the files present would report every document outside that part.
    """
    return sorted(
        document_id
        for document_id, record in record_by_document_id.items()
        if not record.has_generated_output
    )


def write_assembly_records(
    output_file_path: str,
    assembled_records: Sequence[AssembledDocumentRecord]
) -> None:
    LOGGER.info('writing assembly quality record to: %r', output_file_path)
    with auto_uploading_output_file(
        output_file_path, mode='w', encoding='utf-8'
    ) as output_file:
        for record in assembled_records:
            output_file.write(json.dumps(record.to_json_dict(), sort_keys=True) + '\n')
