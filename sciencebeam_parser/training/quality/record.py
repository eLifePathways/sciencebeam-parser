"""The per-document quality record a generated corpus carries.

One file per model, one row per source document, written as the run proceeds and
whether the document succeeded or not: a document that produced no output at all
is invisible to anything that iterates generated files, and is the case this
record exists to make visible.

The record is per model because generation is run per model -- a corpus holds one
model's data at one document set and another model's at a different one, so a
record covering a whole corpus would describe the last run rather than the data
beside it.
"""
import json
import logging
import os
from dataclasses import dataclass, field
from typing import IO, Any, Dict, Optional, Sequence

from sciencebeam_parser.utils.io import auto_uploading_output_file


LOGGER = logging.getLogger(__name__)


QUALITY_RECORD_FILENAME = 'quality.jsonl'


class DocumentStatus:
    OK = 'ok'
    ERROR = 'error'
    TIMEOUT = 'timeout'


class JatsStatus:
    OK = 'ok'
    MISSING = 'missing'
    UNPARSABLE = 'unparsable'
    UNREADABLE = 'unreadable'


def get_quality_record_file_path(
    output_path: str,
    model_name: str,
    use_directory_structure: bool
) -> str:
    """Where a model's record goes, following the layout of its training data."""
    if use_directory_structure:
        return os.path.join(output_path, model_name, QUALITY_RECORD_FILENAME)
    return os.path.join(output_path, model_name + '.' + QUALITY_RECORD_FILENAME)


@dataclass
class JatsQualityRecord:
    status: str
    reference_count: Optional[int] = None
    aligned_reference_count: Optional[int] = None

    def to_json_dict(self) -> Dict[str, Any]:
        json_dict: Dict[str, Any] = {'status': self.status}
        if self.reference_count is not None:
            json_dict['reference_count'] = self.reference_count
        if self.aligned_reference_count is not None:
            json_dict['aligned_reference_count'] = self.aligned_reference_count
        return json_dict


@dataclass
class ModelQualityRecord:
    """What one model wrote for one document.

    `entity_element_count` is None for a model whose labels mark regions rather
    than repeated entities; `written` is False when the generator produced no
    entities and so wrote no file.
    """
    model_name: str
    written: bool
    entity_element_count: Optional[int] = None
    label_counts: Optional[Dict[str, Dict[str, int]]] = None

    def to_json_dict(self) -> Dict[str, Any]:
        json_dict: Dict[str, Any] = {'written': self.written}
        if self.entity_element_count is not None:
            json_dict['entity_element_count'] = self.entity_element_count
        if self.label_counts:
            json_dict['label_counts'] = self.label_counts
        return json_dict


@dataclass
class DocumentQualityRecord:
    """Everything measured for one document, across the models a run generated for."""
    document_id: str
    source_filename: str
    status: str = DocumentStatus.OK
    jats: JatsQualityRecord = field(
        default_factory=lambda: JatsQualityRecord(status=JatsStatus.MISSING)
    )
    models: Sequence[ModelQualityRecord] = ()

    def to_json_dict_by_model(
        self,
        model_names: Sequence[str]
    ) -> Dict[str, Dict[str, Any]]:
        """One row per model, so that each model's record covers the whole document set.

        A document that failed carries no model record, and still gets a row for
        every model the run was asked for: it is absent from all of their data.
        """
        model_record_by_name = {
            model_record.model_name: model_record
            for model_record in self.models
        }
        return {
            model_name: {
                'document_id': self.document_id,
                'source_filename': self.source_filename,
                'status': self.status,
                'model': model_name,
                'jats': self.jats.to_json_dict(),
                **(
                    model_record_by_name[model_name].to_json_dict()
                    if model_name in model_record_by_name
                    else {}
                ),
            }
            for model_name in model_names
        }


def get_failed_document_quality_record(
    source_filename: str,
    document_id: str,
    status: str,
) -> DocumentQualityRecord:
    return DocumentQualityRecord(
        document_id=document_id,
        source_filename=source_filename,
        status=status,
    )


class QualityRecordWriter:
    """Append records as JSON lines, one file per model, flushing each line.

    The writer runs in the parent process, which knows the document set the run
    was asked for -- a worker that timed out or died cannot report itself.
    """
    def __init__(
        self,
        output_path: str,
        model_names: Sequence[str],
        use_directory_structure: bool = True
    ):
        self.output_path = output_path
        self.model_names = list(model_names)
        self.use_directory_structure = use_directory_structure
        self._file_context_by_model: Dict[str, Any] = {}
        self._file_by_model: Dict[str, IO] = {}
        self.written_count = 0

    def __enter__(self) -> 'QualityRecordWriter':
        for model_name in self.model_names:
            file_path = get_quality_record_file_path(
                self.output_path, model_name, self.use_directory_structure
            )
            LOGGER.info('writing quality record to: %r', file_path)
            file_context = auto_uploading_output_file(
                file_path, mode='w', encoding='utf-8'
            )
            self._file_context_by_model[model_name] = file_context
            self._file_by_model[model_name] = file_context.__enter__()
        return self

    def __exit__(self, *args) -> None:
        for file_context in self._file_context_by_model.values():
            file_context.__exit__(*args)
        self._file_context_by_model.clear()
        self._file_by_model.clear()

    def write(self, record: DocumentQualityRecord) -> None:
        for model_name, json_dict in record.to_json_dict_by_model(
            self.model_names
        ).items():
            output_file = self._file_by_model[model_name]
            output_file.write(json.dumps(json_dict, sort_keys=True) + '\n')
            output_file.flush()
        self.written_count += 1
