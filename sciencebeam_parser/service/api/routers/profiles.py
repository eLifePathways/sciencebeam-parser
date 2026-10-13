import logging
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from sciencebeam_parser.app.profiles import ProfileDescription, ProfileRegistry


LOGGER = logging.getLogger(__name__)


PROFILES_PATH = '/profiles'

PROFILES_DESCRIPTION = (
    'What each profile this deployment serves is for. The `name` of any of them '
    'may be sent as `?profile=` to a route that serves a document.'
)


class ProfileResponse(BaseModel):
    name: str
    label: Optional[str] = Field(
        default=None,
        description='A short name for a reader, or null where the profile declares none.'
    )
    description: Optional[str] = Field(
        default=None,
        description='What the profile serves, or null where the profile declares none.'
    )
    alias_names: List[str] = Field(
        default_factory=list,
        description='Other names this profile answers to, which are not listed as choices.'
    )
    is_default: bool = Field(
        default=False,
        description='Whether a request naming no profile is served by this one.'
    )


class ProfileListResponse(BaseModel):
    profiles: List[ProfileResponse]


def get_profile_list_response(
    profile_descriptions: List[ProfileDescription]
) -> ProfileListResponse:
    return ProfileListResponse(profiles=[
        ProfileResponse(
            name=profile_description.name,
            label=profile_description.label,
            description=profile_description.description,
            alias_names=list(profile_description.alias_names),
            is_default=profile_description.is_default
        )
        for profile_description in profile_descriptions
    ])


def create_profiles_router(profile_registry: ProfileRegistry) -> APIRouter:
    """Lists the profiles, with the deployment's own list as the documented example.

    The example is what makes the generated documentation answer "which one do I
    want" on the page: Swagger UI renders a parameter's `enum` as a list of names
    and has no surface for a word about each, so the text has to arrive as a
    response rather than in the parameter's schema. Built here, where the
    registry is in hand, so the page shows what this deployment serves without
    the reader executing anything.
    """
    router = APIRouter(tags=['profiles'])

    example_response = get_profile_list_response(
        profile_registry.get_profile_descriptions()
    ).model_dump()

    @router.get(
        PROFILES_PATH,
        description=PROFILES_DESCRIPTION,
        response_model=ProfileListResponse,
        responses={200: {'content': {'application/json': {'example': example_response}}}}
    )
    def get_profiles_api() -> ProfileListResponse:
        return get_profile_list_response(profile_registry.get_profile_descriptions())

    return router
