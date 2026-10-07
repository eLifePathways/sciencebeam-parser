"""Where the training-source configuration lives.

It sits in the generated data repo rather than here, beside the corpus it
describes: the declaration of what each pair should be is read with the data, and
that repo's evaluation config reads the same dataset block instead of restating
it. The commands are here, because generating needs this repo's environment.

The default is the symlink the Makefile expects at `data/generated-training-data`;
anything else is passed with `--config`.
"""
from __future__ import annotations

from pathlib import Path

from benchmarks.training_selection import SELECTION_DIRECTORY_NAME

DEFAULT_CONFIG = "data/generated-training-data/training-source.yml"


def get_default_selection_path(config_path: str) -> Path:
    """Where a corpus's recorded selection sits, relative to the config.

    Beside it, because the selection is what the declaration names: both describe
    the corpus and both are committed with it.
    """
    return Path(config_path).parent / SELECTION_DIRECTORY_NAME
