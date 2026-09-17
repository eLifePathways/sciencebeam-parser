from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional


PROMPT_ROOT = Path(__file__).parent / 'prompts'


class UnknownPromptError(ValueError):
    pass


@lru_cache(maxsize=None)
def get_prompt_template(task: str, prompt_version: str) -> str:
    path = PROMPT_ROOT / task / f'{prompt_version}.md'
    if not path.is_file():
        available = sorted(p.stem for p in (PROMPT_ROOT / task).glob('*.md')) \
            if (PROMPT_ROOT / task).is_dir() else []
        raise UnknownPromptError(
            f'no prompt {prompt_version!r} for {task!r}; available: {available}'
        )
    return path.read_text(encoding='utf-8').strip()


def get_prompt(
    task: str,
    prompt_version: str,
    rendered_input: str,
    substitutions: Optional[Mapping[str, str]] = None
) -> str:
    """Token replacement rather than `str.format`, because a template that shows
    the model what json to produce contains braces of its own.
    """
    template = get_prompt_template(task, prompt_version)
    for name, value in (substitutions or {}).items():
        template = template.replace('{{' + name + '}}', value)
    return template + '\n\n' + rendered_input
