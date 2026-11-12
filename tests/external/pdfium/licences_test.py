from importlib.metadata import distribution


class TestPdfiumLicences:
    def test_should_carry_the_licences_a_binary_distribution_has_to_ship(self):
        """PDFium's licence and its dependencies' ship with the wheel; an image
        that drops `dist-info` drops them, so their presence is asserted."""
        files = distribution('pypdfium2').files or []
        licence_names = [
            str(path) for path in files
            if 'licenses' in str(path).lower() or 'LICENSE' in str(path)
        ]
        assert any('BSD-3-Clause' in name for name in licence_names)
        assert any('pdfium' in name.lower() for name in licence_names)
