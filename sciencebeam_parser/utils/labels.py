from typing import FrozenSet, Tuple


OTHER_LABELS: FrozenSet[str] = frozenset({'<other>', 'O'})


def strip_tag_prefix(tag: str) -> str:
    if tag and (tag.startswith('B-') or tag.startswith('I-')):
        return tag[2:]
    return tag


def get_split_prefix_label(prefixed_tag: str) -> Tuple[str, str]:
    if '-' in prefixed_tag:
        prefix, tag = prefixed_tag.split('-', maxsplit=1)
    else:
        prefix = ''
        tag = prefixed_tag
    return prefix, tag
