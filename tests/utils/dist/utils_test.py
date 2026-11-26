from sciencebeam_parser.utils.dist.utils import (
    get_github_content_link_prefix,
    get_markdown_with_resolved_links
)


PROJECT_URL = 'https://github.com/user/project'

RELEASE_VERSION_1 = '1.0.1'
DEV_VERSION_1 = '2021.11.12.dev1234567'

LINK_PREFIX_1 = f'{PROJECT_URL}/blob/{RELEASE_VERSION_1}'


class TestGetGithubContentLinkPrefix:
    def test_should_add_blob_prefix_and_version_for_release(self):
        assert get_github_content_link_prefix(
            PROJECT_URL,
            RELEASE_VERSION_1
        ) == f'{PROJECT_URL}/blob/{RELEASE_VERSION_1}'

    def test_should_use_develop_for_non_release_versions(self):
        assert get_github_content_link_prefix(
            PROJECT_URL,
            DEV_VERSION_1
        ) == f'{PROJECT_URL}/blob/develop'


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
