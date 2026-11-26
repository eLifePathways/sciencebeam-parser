import textwrap

from scripts.dev.update_readme import (
    get_markdown_with_resolved_links,
    update_readme_text
)


PROJECT_URL = 'https://github.com/user/project'
RELEASE_VERSION_1 = '1.0.1'

LINK_PREFIX_1 = f'{PROJECT_URL}/blob/{RELEASE_VERSION_1}'


class TestGetMarkdownWithResolvedLinks:
    def test_should_return_unchanged_markdown_without_links(self):
        assert get_markdown_with_resolved_links(
            '# Header 1',
            source_base_path='docs',
            link_prefix=LINK_PREFIX_1
        ) == '# Header 1'

    def test_should_return_change_relative_link_without_source_base_path(self):
        assert get_markdown_with_resolved_links(
            '# Header 1\n[link](other.md)',
            source_base_path='',
            link_prefix=LINK_PREFIX_1
        ) == f'# Header 1\n[link]({LINK_PREFIX_1}/other.md)'

    def test_should_return_change_relative_link_with_source_base_path(self):
        assert get_markdown_with_resolved_links(
            '# Header 1\n[link](other.md)',
            source_base_path='docs',
            link_prefix=LINK_PREFIX_1
        ) == f'# Header 1\n[link]({LINK_PREFIX_1}/docs/other.md)'

    def test_should_return_change_relative_link_path_outside_source_base_path(self):
        assert get_markdown_with_resolved_links(
            '# Header 1\n[link](../src/file.xyz)',
            source_base_path='docs',
            link_prefix=LINK_PREFIX_1
        ) == f'# Header 1\n[link]({LINK_PREFIX_1}/src/file.xyz)'

    def test_should_not_replace_absolute_urls(self):
        assert get_markdown_with_resolved_links(
            '# Header 1\n[link](https://host/file.xyz)',
            source_base_path='docs',
            link_prefix=LINK_PREFIX_1
        ) == '# Header 1\n[link](https://host/file.xyz)'


class TestUpdateReadmeText:
    def test_should_keep_original_text(self):
        assert update_readme_text(textwrap.dedent(
            '''
            Text
            '''
        ), source_base_path='docs', link_prefix=LINK_PREFIX_1).strip() == textwrap.dedent(
            '''
            Text
            '''
        ).strip()

    def test_should_strip_special_links(self):
        assert update_readme_text(textwrap.dedent(
            '''
            Text
            [![Label](image_path)](link)
            '''
        ), source_base_path='docs', link_prefix=LINK_PREFIX_1).strip() == textwrap.dedent(
            '''
            Text
            '''
        ).strip()

    def test_should_resolve_relative_links(self):
        assert update_readme_text(textwrap.dedent(
            '''
            [label](../path/to/file.yml)
            '''
        ), source_base_path='docs', link_prefix=LINK_PREFIX_1).strip() == textwrap.dedent(
            f'''
            [label]({LINK_PREFIX_1}/path/to/file.yml)
            '''
        ).strip()
