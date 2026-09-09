from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from sciencebeam_parser.document.layout_document import LayoutDocument, LayoutToken


# id(token) -> (field_name, sub_field_name_or_None, instance_id)
TokenLabelById = Dict[int, Tuple[str, Optional[str], int]]


@dataclass
class JatsAnnotatedLayoutDocument:
    layout_document: LayoutDocument
    token_label_by_id: TokenLabelById = field(default_factory=dict)

    def get_token_field(self, token: LayoutToken) -> Optional[str]:
        entry = self.token_label_by_id.get(id(token))
        return entry[0] if entry is not None else None

    def get_token_sub_field(self, token: LayoutToken) -> Optional[str]:
        entry = self.token_label_by_id.get(id(token))
        return entry[1] if entry is not None else None

    def get_token_instance(self, token: LayoutToken) -> int:
        entry = self.token_label_by_id.get(id(token))
        return entry[2] if entry is not None else 0

    def set_token_label(
        self,
        token: LayoutToken,
        field_name: str,
        sub_field_name: Optional[str] = None,
        instance_id: int = 0,
    ) -> None:
        self.token_label_by_id[id(token)] = (field_name, sub_field_name, instance_id)

    def get_aligned_instance_count(self, field_name: str) -> int:
        """Number of instances of a repeated field that got at least one token.

        Instance ids are assigned per parent match and start at 1, so this is the
        count of values the aligner placed -- fewer than the JATS holds means
        alignment lost them, rather than a later stage.
        """
        return len({
            entry[2]
            for entry in self.token_label_by_id.values()
            if entry[0] == field_name and entry[2]
        })

    def coverage_ratio(self) -> float:
        total = sum(1 for _ in self.layout_document.iter_all_tokens())
        if total == 0:
            return 1.0
        return len(self.token_label_by_id) / total
