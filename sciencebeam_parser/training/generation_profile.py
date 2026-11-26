from typing import Any, Dict, Optional

from sciencebeam_parser.config.config import AppConfig
from sciencebeam_parser.resources.default_config import DEFAULT_CONFIG_FILE


# What a run generates under when it names nothing. Stated here rather than
# taken from the shipped `profile:`, which says what the service serves: a
# corpus pre-annotated by the models trained on it is a different experiment
# from one pre-annotated by GROBID, and which of the two a run wants is not
# answered by what a deployment happens to serve.
DEFAULT_GENERATION_PROFILE = 'grobid_crf'


def get_generation_config(profile_name: Optional[str]) -> AppConfig:
    """The configuration a run generates under.

    Choosing a profile means setting the one the parser will resolve, not merging
    its overlay in here: `ScienceBeamParser` builds its models through a
    `ProfileRegistry` that resolves the *default* profile against the
    configuration it was handed, so a `models:` written here is resolved over and
    has no effect.
    """
    config = AppConfig.load_yaml(DEFAULT_CONFIG_FILE)
    name = profile_name or DEFAULT_GENERATION_PROFILE
    config.resolve_profile(name)  # reject an unknown name here, not in a worker
    return AppConfig({**config.props, 'profile': name})


def get_generation_model_config(config: AppConfig, model_name: str) -> Dict[str, Any]:
    """The model configuration the run's profile resolves to, by `--model-name` or file name."""
    return config.resolve_profile().props.get('models', {}).get(model_name.replace('-', '_'), {})
