from sciencebeam_parser.utils.data_wrapper import (
    MediaDataWrapper,
    get_data_wrapper_with_improved_media_type_or_filename
)
from sciencebeam_parser.utils.media_types import MediaTypes


PDF_DATA_1 = b'PDF 1'
PDF_FILENAME_1 = 'test.pdf'


class TestGetDataWrapperWithImprovedMediaTypeOrFilename:
    def test_should_return_passed_in_media_type_and_filename_if_valid(self):
        source_data_wrapper = MediaDataWrapper(
            data=PDF_DATA_1,
            media_type=MediaTypes.PDF,
            filename=PDF_FILENAME_1
        )
        result = get_data_wrapper_with_improved_media_type_or_filename(source_data_wrapper)
        assert result == source_data_wrapper

    def test_should_return_guess_media_type_if_octet_stream(self):
        source_data_wrapper = MediaDataWrapper(
            data=PDF_DATA_1,
            media_type=MediaTypes.OCTET_STREAM,
            filename=PDF_FILENAME_1
        )
        result = get_data_wrapper_with_improved_media_type_or_filename(source_data_wrapper)
        assert result == source_data_wrapper._replace(
            media_type=MediaTypes.PDF
        )

    def test_should_return_guess_media_type_if_empty(self):
        source_data_wrapper = MediaDataWrapper(
            data=PDF_DATA_1,
            media_type='',
            filename=PDF_FILENAME_1
        )
        result = get_data_wrapper_with_improved_media_type_or_filename(source_data_wrapper)
        assert result == source_data_wrapper._replace(
            media_type=MediaTypes.PDF
        )

    def test_should_return_guess_filename_if_none(self):
        source_data_wrapper = MediaDataWrapper(
            data=PDF_DATA_1,
            media_type=MediaTypes.PDF,
            filename=None
        )
        result = get_data_wrapper_with_improved_media_type_or_filename(source_data_wrapper)
        assert result.filename is not None
        assert result.filename.endswith('.pdf')
