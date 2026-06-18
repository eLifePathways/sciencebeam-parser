from dataclasses import dataclass, field
from typing import Collection, Mapping, Set

from sciencebeam_parser.training.jats.annotated_document import JatsAnnotatedLayoutDocument


@dataclass
class CoverageResult:
    """Summary of how well a set of required fields was matched."""
    required_fields_present: Set[str] = field(default_factory=set)
    required_fields_missing: Set[str] = field(default_factory=set)
    required_matching_fields_matched: Set[str] = field(default_factory=set)
    required_matching_fields_missing: Set[str] = field(default_factory=set)

    @property
    def is_passing(self) -> bool:
        return (
            not self.required_fields_missing
            and not self.required_matching_fields_missing
        )

    def __str__(self) -> str:
        parts = []
        if self.required_fields_missing:
            parts.append(f'required fields absent: {sorted(self.required_fields_missing)}')
        if self.required_matching_fields_missing:
            parts.append(
                f'matching fields not aligned: '
                f'{sorted(self.required_matching_fields_missing)}'
            )
        return '; '.join(parts) if parts else 'OK'


def check_coverage(
    annotated: JatsAnnotatedLayoutDocument,
    field_values_by_field: Mapping[str, bool],
    required_fields: Collection[str],
    require_matching_fields: Collection[str],
) -> CoverageResult:
    """
    Args:
        annotated: the annotated layout document
        field_values_by_field: mapping of field_name → whether that field appeared in JATS
        required_fields: fields that must be present AND aligned
        require_matching_fields: fields that must be aligned IF they appear in JATS
    """
    aligned_fields: Set[str] = {
        entry[0]
        for entry in annotated.token_label_by_id.values()
    }
    present_fields: Set[str] = {
        f for f, present in field_values_by_field.items() if present
    }

    result = CoverageResult()
    for f in required_fields:
        if f not in present_fields or f not in aligned_fields:
            result.required_fields_missing.add(f)
        else:
            result.required_fields_present.add(f)

    for f in require_matching_fields:
        if f not in present_fields:
            continue  # not in JATS → no constraint
        if f not in aligned_fields:
            result.required_matching_fields_missing.add(f)
        else:
            result.required_matching_fields_matched.add(f)

    return result
