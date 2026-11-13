from benchmarks.selection_record import selection_record

SOURCES = {"corpus": {"sources": [{"repo_id": "org/a", "revision": "v1.0.0", "commit": "c"}]}}
RECORDS = [
    {"corpus": "corpus", "record_id": "a", "repo_id": "org/a", "xml_sha256": "x1",
     "pdf_sha256": "p1", "xml_path": "/tmp/a.jats.xml"},
    {"corpus": "corpus", "record_id": "b", "repo_id": "org/a", "xml_sha256": "x2",
     "pdf_sha256": "p2", "xml_path": "/tmp/b.jats.xml"},
]


def test_records_ids_and_checksums_but_not_local_paths():
    record = selection_record(SOURCES, RECORDS)
    assert record["documents"]["corpus"][0] == {
        "record_id": "a", "repo_id": "org/a", "xml_sha256": "x1", "pdf_sha256": "p1",
    }


def test_digest_is_stable_and_changes_with_content():
    digest = selection_record(SOURCES, RECORDS)["selection_digest"]
    assert digest == selection_record(SOURCES, RECORDS)["selection_digest"]
    corrected = [RECORDS[0], {**RECORDS[1], "xml_sha256": "x3"}]
    assert digest != selection_record(SOURCES, corrected)["selection_digest"]
    moved = {"corpus": {"sources": [{**SOURCES["corpus"]["sources"][0], "commit": "d"}]}}
    assert digest != selection_record(moved, RECORDS)["selection_digest"]
