from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from sciencebeam_parser.document.layout_document import LayoutDocument, LayoutToken


# id(token) -> (field_name, sub_field_name_or_None)
TokenLabelById = Dict[int, Tuple[str, Optional[str]]]


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

    def set_token_label(
        self,
        token: LayoutToken,
        field_name: str,
        sub_field_name: Optional[str] = None,
    ) -> None:
        self.token_label_by_id[id(token)] = (field_name, sub_field_name)

    def coverage_ratio(self) -> float:
        total = sum(1 for _ in self.layout_document.iter_all_tokens())
        if total == 0:
            return 1.0
        return len(self.token_label_by_id) / total
