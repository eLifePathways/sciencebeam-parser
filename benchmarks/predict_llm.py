"""Predict with the trained JATS annotation model.

The model never sees the PDF: the service converts it to markdown, then
annotates it in three rollouts -- front, body, back -- each returning a complete
`<article>` with only its own section filled in. A prediction is those merged.

Generation is separate from the benchmark run, which has no GPU: predictions are
pushed to the store and read back like any baseline that does not generate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
import yaml
from lxml import etree

from benchmarks.fetch import fetch_data, resolved_sources
from benchmarks.predict import _append_manifest, _load_done, _Progress

LOGGER = logging.getLogger(__name__)

# JATS document order, which is also merge order.
SECTIONS: Tuple[str, ...] = ("front", "body", "back")

PREPROCESS_PATH = "/preprocess"
ANNOTATE_PATH = "/annotate"

# One section of a five-line document took 232s, so this is not the parser's scale.
DEFAULT_TIMEOUT_SECONDS = 1800
# Documents in flight. Each one holds three calls open, so this multiplies.
DEFAULT_CONCURRENCY = 2

# Not redistributable, and generation is the step that uploads them.
RESTRICTED_CORPORA_FOR_LLM = frozenset({"plos-manuscripts"})


def check_restricted_corpora(include: Optional[Iterable[str]]) -> None:
    restricted = sorted(RESTRICTED_CORPORA_FOR_LLM.intersection(include or ()))
    if restricted:
        raise SystemExit(
            f"refusing to annotate {', '.join(restricted)} with a served model:"
            " those manuscripts are not redistributable. Drop the corpus or point"
            " --endpoint at a self-hosted service."
        )


def _resolve_concurrency(concurrency: int) -> int:
    """0 means this module's default, not the parser's cpu-count one.

    A document is one call there and three here, against a service whose
    throughput is the bound -- local cores are the wrong thing to scale with.
    """
    return concurrency if concurrency > 0 else DEFAULT_CONCURRENCY


def _find_section(root: etree._Element, section: str) -> Optional[etree._Element]:
    """The named section of an article, whatever namespace it is declared in."""
    matches = root.xpath(f"*[local-name()='{section}']")
    return matches[0] if matches else None


def merge_section_documents(xml_by_section: Dict[str, str]) -> bytes:
    """One article from the three single-section documents the model returns.

    Every response carries a full skeleton, so merging replaces each section with
    the one from the response that annotated it. The front response is preferred
    as the skeleton: `article-type` and `xml:lang` come from the front matter.
    """
    present = [section for section in SECTIONS if xml_by_section.get(section)]
    if not present:
        raise ValueError("no section responses to merge")

    skeleton_source = xml_by_section.get("front") or xml_by_section[present[0]]
    merged = etree.fromstring(skeleton_source.encode("utf-8"))

    for section in SECTIONS:
        source = xml_by_section.get(section)
        if not source:
            continue
        annotated = _find_section(etree.fromstring(source.encode("utf-8")), section)
        if annotated is None:
            continue
        existing = _find_section(merged, section)
        if existing is not None:
            merged.replace(existing, annotated)
        else:
            merged.append(annotated)

    return etree.tostring(merged, xml_declaration=True, encoding="utf-8")


async def _preprocess(
    client: httpx.AsyncClient, endpoint: str, pdf_path: Path, timeout: int
) -> Dict[str, Any]:
    response = await client.post(
        f"{endpoint}{PREPROCESS_PATH}",
        files={"file": (pdf_path.name, pdf_path.read_bytes(), "application/pdf")},
        timeout=timeout,
    )
    response.raise_for_status()
    result = response.json()
    if not result.get("markdown"):
        raise RuntimeError(f"preprocess returned no markdown for {pdf_path.name}")
    return result


async def _annotate(
    client: httpx.AsyncClient,
    endpoint: str,
    section: str,
    preprocessed: Dict[str, Any],
    timeout: int,
) -> str:
    data = {
        "markdown": preprocessed["markdown"],
        "section": section,
        "doc_id": preprocessed.get("doc_id", ""),
    }
    # The inline/xref merge, which only the body rollout consumes.
    candidates = preprocessed.get("candidates")
    if section == "body" and candidates:
        data["candidates"] = json.dumps(candidates)

    response = await client.post(f"{endpoint}{ANNOTATE_PATH}", data=data, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    xml = payload.get("xml")
    if not xml:
        raise RuntimeError(f"annotate returned no xml for section {section!r}")
    return xml


async def _predict_one(
    client: httpx.AsyncClient, endpoint: str, pdf_path: Path, timeout: int
) -> bytes:
    """One document's JATS, or an exception.

    A failed section fails the document: a partial article scores as one whose
    references were not there, which a service error and a real result share.
    """
    preprocessed = await _preprocess(client, endpoint, pdf_path, timeout)
    rollouts = await asyncio.gather(
        *[_annotate(client, endpoint, section, preprocessed, timeout) for section in SECTIONS]
    )
    return merge_section_documents(dict(zip(SECTIONS, rollouts)))


async def _run_predict_async(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    records: List[Dict[str, Any]],
    done: set,
    run_dir: Path,
    endpoint: str,
    timeout: int,
    concurrency: int,
) -> Tuple[int, int]:
    to_process = [r for r in records if (r["corpus"], r["record_id"]) not in done]
    skipped = len(records) - len(to_process)
    if skipped:
        LOGGER.info("Skipping %d already-cached documents", skipped)
    LOGGER.info(
        "Annotating %d documents (concurrency=%d, %d calls in flight)",
        len(to_process), concurrency, concurrency * len(SECTIONS),
    )

    progress = _Progress(len(to_process))
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:

        async def _process_one(rec: Dict[str, Any]) -> None:
            corpus = rec["corpus"]
            record_id = rec["record_id"]
            out_path = run_dir / "predictions" / corpus / f"{record_id}.jats.xml"
            async with sem:
                t0 = time.monotonic()
                try:
                    merged = await _predict_one(
                        client, endpoint, Path(rec["pdf_path"]), timeout
                    )
                    elapsed_ms = round((time.monotonic() - t0) * 1000)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_bytes(merged)
                    _append_manifest(run_dir, {
                        "corpus": corpus, "record_id": record_id,
                        "status": "ok", "elapsed_ms": elapsed_ms,
                    })
                    progress.record_ok(corpus, record_id, elapsed_ms)
                except httpx.HTTPStatusError as exc:
                    msg = str(exc)
                    body = exc.response.text[:2000] if exc.response.text else ""
                    if body:
                        LOGGER.error("err %s/%s  %s\n%s", corpus, record_id, msg, body)
                    else:
                        LOGGER.error("err %s/%s  %s", corpus, record_id, msg)
                    _append_manifest(run_dir, {
                        "corpus": corpus, "record_id": record_id,
                        "status": "error", "error": msg, "error_body": body,
                    })
                    progress.record_err(corpus, record_id)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    msg = str(exc)
                    _append_manifest(run_dir, {
                        "corpus": corpus, "record_id": record_id,
                        "status": "error", "error": msg,
                    })
                    progress.record_err(corpus, record_id)
                    LOGGER.error("err %s/%s  %s", corpus, record_id, msg)

        await asyncio.gather(*[_process_one(rec) for rec in to_process])

    return progress.n_ok, progress.n_err


def run_predict_llm(  # noqa: E501  pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    config: dict,
    mode: str,
    split: str,
    data_dir: Path,
    run_dir: Path,
    endpoint: str,
    checkpoint: str,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    include: Optional[Iterable[str]] = None,
) -> None:
    check_restricted_corpora(include)
    records = fetch_data(config, mode, split, data_dir, include=include)
    sources = resolved_sources(config, split, include)
    done = _load_done(run_dir)

    t_start = time.monotonic()
    n_ok, n_err = asyncio.run(
        _run_predict_async(
            records, done, run_dir, endpoint.rstrip("/"), timeout,
            _resolve_concurrency(concurrency),
        )
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps({
        # `parser_image` is the key the report renders; here that is the checkpoint.
        "parser_image": checkpoint,
        "profile": "default",
        "endpoint": endpoint,
        "checkpoint": checkpoint,
        "sources": sources,
        "split": split,
        "mode": mode,
        "corpora": list(sources),
        "fields": config["fields"],
        "n_records": n_ok,
        "n_errors": n_err,
        "elapsed_s": round(time.monotonic() - t_start, 1),
    }, indent=2))

    LOGGER.info(
        "done  ok=%d  err=%d  elapsed=%.1fs",
        n_ok, n_err, time.monotonic() - t_start,
    )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Predict: annotate each PDF with the trained JATS model and save JATS XML"
    )
    parser.add_argument("--config", default="benchmarks/eval.yml")
    parser.add_argument("--mode", default="smoke")
    parser.add_argument("--split", default="train")
    parser.add_argument("--data", default="benchmarks/data", help="Data cache directory")
    parser.add_argument("--out", required=True, help="Run output dir")
    parser.add_argument(
        "--endpoint", required=True,
        help=(
            "Base URL of the annotation service. Required rather than defaulted:"
            " every document of the run is uploaded to it"
        ),
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Checkpoint identifier, recorded with the run and used as its store version",
    )
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help=f"Documents in flight, each holding {len(SECTIONS)} calls open"
             f" (default {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Per-call timeout in seconds (default {DEFAULT_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--include-corpus", action="append", default=None, dest="include_corpus",
        metavar="CORPUS", help="Also run an opt-in corpus, repeatable",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_predict_llm(
        config=config,
        mode=args.mode,
        split=args.split,
        data_dir=Path(args.data),
        run_dir=Path(args.out),
        endpoint=args.endpoint,
        checkpoint=args.checkpoint,
        concurrency=args.concurrency,
        timeout=args.timeout,
        include=args.include_corpus,
    )


if __name__ == "__main__":
    main()
