from typing import List, NamedTuple, Optional

from lxml import etree

from sciencebeam_parser.document.tei.common import TEI_E


APPLICATION_IDENT = 'sciencebeam-parser'

PROFILE_LABEL_TYPE = 'profile'
PROFILE_DIGEST_LABEL_TYPE = 'profile-digest'


class DocumentAttribution(NamedTuple):
    """What produced a document, as the document itself will say.

    The digest is required and the name is not: a configuration without a
    `profile:` key has no name to give, and a file that recorded nothing at all
    in that case would be exactly the unattributable artefact this exists to
    prevent.
    """
    version: str
    profile_digest: str
    profile_name: Optional[str] = None


def get_tei_application_element(
    attribution: DocumentAttribution
) -> etree.ElementBase:
    """The facts as `label` children rather than as attributes on `application`.

    TEI allows neither an `ident` of our own choosing nor extra attributes here,
    and requires at least one label-like child, which the digest always provides.
    """
    labels: List[etree.ElementBase] = []
    if attribution.profile_name:
        labels.append(TEI_E(
            'label', attribution.profile_name, type=PROFILE_LABEL_TYPE
        ))
    labels.append(TEI_E(
        'label', attribution.profile_digest, type=PROFILE_DIGEST_LABEL_TYPE
    ))
    return TEI_E(
        'application',
        {'ident': APPLICATION_IDENT, 'version': attribution.version},
        *labels
    )
