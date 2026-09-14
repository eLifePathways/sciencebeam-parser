import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import yaml

from lxml import etree

from benchmarks.llm_usage import combine_usage
from benchmarks.predict_llm import (
    _post_with_retry,
    check_restricted_corpora,
    rollout_usage,
    checkpoint_from_config,
    merge_section_documents,
)


def _article(front='', body='', back='', attrs=''):
    return (
        f'<article xmlns:xlink="http://www.w3.org/1999/xlink"{attrs}>'
        f'<front>{front}</front><body>{body}</body><back>{back}</back>'
        f'</article>'
    )


class TestMergeSectionDocuments:
    def test_should_take_each_section_from_its_own_response(self):
        merged = etree.fromstring(merge_section_documents({
            'front': _article(front='<article-title>T</article-title>'),
            'body': _article(body='<sec><title>S</title></sec>'),
            'back': _article(back='<ref-list><ref/></ref-list>'),
        }))
        assert merged.find('front/article-title').text == 'T'
        assert merged.find('body/sec/title').text == 'S'
        assert merged.find('back/ref-list') is not None

    def test_should_not_keep_the_empty_sections_of_the_skeleton(self):
        # Appending rather than replacing would leave two of each, and score
        # the empty one.
        merged = etree.fromstring(merge_section_documents({
            'front': _article(front='<article-title>T</article-title>'),
            'body': _article(body='<sec><title>S</title></sec>'),
            'back': _article(back='<ref-list><ref/></ref-list>'),
        }))
        assert len(merged.findall('front')) == 1
        assert len(merged.findall('body')) == 1
        assert len(merged.findall('back')) == 1

    def test_should_keep_the_article_attributes_of_the_front_response(self):
        merged = etree.fromstring(merge_section_documents({
            'front': _article(attrs=' article-type="research-article"'),
            'body': _article(),
        }))
        assert merged.get('article-type') == 'research-article'

    def test_should_order_sections_as_jats_requires(self):
        merged = etree.fromstring(merge_section_documents({
            'back': _article(back='<ref-list/>'),
            'body': _article(body='<sec/>'),
        }))
        assert [child.tag for child in merged] == ['front', 'body', 'back']

    def test_should_merge_what_is_present_when_a_section_is_absent(self):
        merged = etree.fromstring(merge_section_documents({
            'body': _article(body='<sec><title>S</title></sec>'),
        }))
        assert merged.find('body/sec/title').text == 'S'

    def test_should_reject_an_empty_set_of_responses(self):
        with pytest.raises(ValueError):
            merge_section_documents({})

    def test_should_produce_an_article_rooted_document(self):
        # The scorer picks its mapping from the root element.
        merged = etree.fromstring(merge_section_documents({'front': _article()}))
        assert merged.tag == 'article'


class TestCheckRestrictedCorpora:
    def test_should_allow_the_public_corpora(self):
        check_restricted_corpora(['biorxiv', 'pkp'])

    def test_should_allow_no_opt_in_corpora(self):
        check_restricted_corpora(None)

    def test_should_refuse_the_private_manuscripts(self):
        with pytest.raises(SystemExit, match='not redistributable'):
            check_restricted_corpora(['biorxiv', 'plos-manuscripts'])


class TestCheckpointFromConfig:
    """Generation and the benchmark read one field, so they cannot disagree."""

    def test_should_read_the_version_of_this_tool(self):
        config = {"baselines": [
            {"tool": "grobid", "version": "0.9.1-crf", "profile": "default"},
            {"tool": "jats-agentic-annotation", "version": "ckpt-1", "profile": "default"},
        ]}
        assert checkpoint_from_config(config) == "ckpt-1"

    def test_should_be_none_when_the_tool_is_absent(self):
        config = {"baselines": [{"tool": "grobid", "version": "0.9.1-crf"}]}
        assert checkpoint_from_config(config) is None

    def test_should_be_none_without_baselines(self):
        assert checkpoint_from_config({}) is None

    def test_should_match_the_shipped_config(self):
        # The generation default has to resolve, or a dispatch with no checkpoint
        # exits rather than running for hours.
        with open("benchmarks/eval.yml", encoding="utf-8") as f:
            assert checkpoint_from_config(yaml.safe_load(f))


class TestPostWithRetry:
    """The first real run lost 40 of 60 documents to an unfollowed redirect and
    11 more to transient errors nobody asked again about."""

    @staticmethod
    def _response(status, headers=None):
        request = httpx.Request("POST", "https://example.test/annotate")
        return httpx.Response(status, headers=headers or {}, request=request)

    def test_should_return_the_first_success(self):
        client = MagicMock()
        client.post = AsyncMock(return_value=self._response(200))
        result = asyncio.run(_post_with_retry(client, "https://example.test/annotate", 5, 4))
        assert result.status_code == 200
        assert client.post.await_count == 1

    def test_should_retry_a_transient_status_then_succeed(self):
        client = MagicMock()
        client.post = AsyncMock(side_effect=[self._response(500), self._response(200)])
        with patch("benchmarks.predict_llm.asyncio.sleep", new=AsyncMock()):
            result = asyncio.run(_post_with_retry(client, "https://example.test/annotate", 5, 4))
        assert result.status_code == 200
        assert client.post.await_count == 2

    def test_should_retry_a_timeout(self):
        client = MagicMock()
        client.post = AsyncMock(side_effect=[self._response(408), self._response(200)])
        with patch("benchmarks.predict_llm.asyncio.sleep", new=AsyncMock()):
            asyncio.run(_post_with_retry(client, "https://example.test/annotate", 5, 4))
        assert client.post.await_count == 2

    def test_should_not_retry_a_client_error(self):
        # A 422 will say the same thing however many times it is asked.
        client = MagicMock()
        client.post = AsyncMock(return_value=self._response(422))
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(_post_with_retry(client, "https://example.test/annotate", 5, 4))
        assert client.post.await_count == 1

    def test_should_give_up_after_max_attempts(self):
        client = MagicMock()
        client.post = AsyncMock(return_value=self._response(503))
        with patch("benchmarks.predict_llm.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(httpx.HTTPStatusError):
                asyncio.run(_post_with_retry(client, "https://example.test/annotate", 5, 3))
        assert client.post.await_count == 3

    def test_should_honour_retry_after(self):
        client = MagicMock()
        client.post = AsyncMock(
            side_effect=[self._response(429, {"Retry-After": "7"}), self._response(200)]
        )
        sleep = AsyncMock()
        with patch("benchmarks.predict_llm.asyncio.sleep", new=sleep):
            asyncio.run(_post_with_retry(client, "https://example.test/annotate", 5, 4))
        assert sleep.await_args[0][0] == 7


class TestRolloutUsage:
    """Recorded in the shape benchmarks/llm_usage.py already aggregates, so the
    annotation model's spend lands in the summary like the engine's does."""

    @staticmethod
    def _sections():
        return {
            "front": {"n_turns": 4, "n_calls": 3, "n_structural_calls": 3,
                      "n_merged_candidate_spans": 0, "elapsed_ms": 40000,
                      "terminated_by": "finish"},
            "body": {"n_turns": 30, "n_calls": 28, "n_structural_calls": 25,
                     "n_merged_candidate_spans": 12, "elapsed_ms": 120000,
                     "terminated_by": "finish"},
            "back": {"n_turns": 9, "n_calls": 8, "n_structural_calls": 8,
                     "n_merged_candidate_spans": 0, "elapsed_ms": 60000,
                     "terminated_by": "finish"},
        }

    def test_should_count_a_turn_as_a_call(self):
        usage = rollout_usage(self._sections(), "ckpt-1", {})
        assert usage["calls"] == 43

    def test_should_name_the_checkpoint_as_the_model(self):
        assert rollout_usage(self._sections(), "ckpt-1", {})["models"] == ["ckpt-1"]

    def test_should_keep_each_section_separately(self):
        by_task = rollout_usage(self._sections(), "ckpt-1", {})["by_task"]
        assert set(by_task) == {"front", "body", "back"}
        assert by_task["body"]["calls"] == 30
        assert by_task["body"]["merged_candidate_spans"] == 12

    def test_should_record_the_preprocessing_it_was_given(self):
        usage = rollout_usage(self._sections(), "ckpt-1",
                              {"n_md_lines": 900, "processing_time_ms": 36000})
        assert usage["markdown_lines"] == 900
        assert usage["preprocess_ms"] == 36000

    def test_should_not_invent_token_counts(self):
        # The service does not report them. A zero would be read as "spent
        # nothing" rather than "not known", and would sum into the totals.
        usage = rollout_usage(self._sections(), "ckpt-1", {})
        assert "input_tokens" not in usage
        assert "output_tokens" not in usage

    def test_should_survive_a_response_missing_its_counters(self):
        usage = rollout_usage({"front": {}}, "ckpt-1", {})
        assert usage["calls"] == 0
        assert usage["by_task"]["front"]["terminated_by"] is None

    def test_should_aggregate_through_llm_usage(self):
        # The point of the shape: combine_usage has to understand it.
        combined = combine_usage([rollout_usage(self._sections(), "ckpt-1", {})])
        assert combined["calls"] == 43
        assert combined["models"] == ["ckpt-1"]
        assert combined["by_task"]["body"]["calls"] == 30
