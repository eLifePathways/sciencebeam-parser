import logging
from typing import Any, Mapping, Optional, Union

from lxml import etree


LOGGER = logging.getLogger(__name__)


T_XSLT_Input = Union[etree.ElementBase, etree.ElementTree]


class XsltTransformerWrapper:
    def __init__(
        self,
        xslt_template: Union[str, bytes],
        xslt_template_parameters: Optional[Mapping[str, Any]] = None
    ):
        self.xslt_template = xslt_template
        if xslt_template_parameters is None:
            xslt_template_parameters = {}
        self.xslt_template_parameters = xslt_template_parameters
        self.__transformer: Optional[etree.XSLT] = None
        # validate the XSLT stylesheet
        etree.fromstring(self.xslt_template)

    @staticmethod
    def from_template_string(
        xslt_template: Union[str, bytes], **kwargs
    ) -> 'XsltTransformerWrapper':
        return XsltTransformerWrapper(xslt_template, **kwargs)

    @staticmethod
    def from_template_file(xslt_template_file: str, **kwargs) -> 'XsltTransformerWrapper':
        return XsltTransformerWrapper.from_template_string(
            etree.tostring(etree.parse(xslt_template_file)),
            **kwargs
        )

    def _get_transformer(self) -> etree.XSLT:
        if self.__transformer is None:
            # The transform function cannot be pickled and needs to be loaded lazily
            transform = etree.XSLT(
                etree.fromstring(self.xslt_template)
            )
            self.__transformer = transform
        return self.__transformer

    def __call__(
        self,
        xslt_input: T_XSLT_Input,
        xslt_template_parameters: Optional[Mapping[str, Any]] = None
    ):
        xslt_template_parameters = {
            **self.xslt_template_parameters,
            **(xslt_template_parameters or {})
        }
        LOGGER.debug(
            'xslt_input: %r (xslt_template_parameters=%r)',
            xslt_input, xslt_template_parameters
        )
        _xslt_transformer = self._get_transformer()
        return _xslt_transformer(
            xslt_input,
            **{
                key: etree.XSLT.strparam(value)
                for key, value in xslt_template_parameters.items()
            }
        )
