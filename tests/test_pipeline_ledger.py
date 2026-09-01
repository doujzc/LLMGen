"""Regression tests for resumable immutable provider-work shard ledgers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmgen.pipeline.ledger import (
    EmbeddingRecord,
    JsonlShardLedger,
    LedgerIntegrityError,
    RequestRecord,
    ResponseRecord,
    stable_embedding_id,
    stable_request_id,
)


def _request(index: int) -> RequestRecord:
    return RequestRecord.from_prompt(
        "generate-queries",
        f"generate query {index}",
        request_key={"model": "test-model", "temperature": 0.2},
        metadata={"workflow_id": f"wf-{index}"},
    )


def _response(record: RequestRecord, status: str, payload: object = None) -> ResponseRecord:
    return ResponseRecord(
        request_id=record.request_id,
        prompt_hash=record.prompt_hash,
        status=status,  # type: ignore[arg-type]
        response=payload,
        error={"type": "temporary"} if status == "failed" else {},
    )


def test_stable_provider_identities_include_request_parameters() -> None:
    first = stable_request_id("generate", "same prompt", request_key={"model": "a"})
    again = stable_request_id("generate", "same prompt", request_key={"model": "a"})
    changed = stable_request_id("generate", "same prompt", request_key={"model": "b"})
    assert first == again
    assert first[1] == changed[1]
    assert first[0] != changed[0]

    embedding = stable_embedding_id("embed", "same text", item_key="skill-a", model="m1")
    changed_embedding = stable_embedding_id("embed", "same text", item_key="skill-a", model="m2")
    assert embedding[1] == changed_embedding[1]
    assert embedding[0] != changed_embedding[0]


def test_request_response_ledger_retries_failures_and_deduplicates_successes(
    tmp_path: Path,
) -> None:
    ledger = JsonlShardLedger(tmp_path / "ledger", batch_size=2)
    records = [_request(index) for index in range(3)]

    first = ledger.schedule_requests(records)
    assert [row.request_id for row in first.records] == [records[0].request_id, records[1].request_id]
    assert first.newly_recorded == 2
    assert first.retried == 0
    request_shard = first_result = ledger.manifest()["shards"][0]
    assert request_shard["kind"] == "requests"

    failed = ledger.record_responses([_response(records[0], "failed")])
    assert failed.accepted == 1
    retried = ledger.schedule_requests(records)
    assert [row.request_id for row in retried.records] == [records[0].request_id, records[1].request_id]
    assert retried.newly_recorded == 0
    assert retried.retried == 2

    success = ledger.record_responses(
        [
            _response(records[0], "succeeded", {"query": "one"}),
            _response(records[1], "succeeded", {"query": "two"}),
        ]
    )
    assert success.accepted == 2
    next_batch = ledger.schedule_requests(records)
    assert [row.request_id for row in next_batch.records] == [records[2].request_id]
    assert next_batch.newly_recorded == 1
    assert next_batch.skipped_succeeded == 2

    ledger.record_responses([_response(records[2], "succeeded", {"query": "three"})])
    finished = ledger.schedule_requests(records)
    assert finished.records == ()
    assert finished.skipped_succeeded == 3

    manifest = ledger.verify()
    assert manifest["stats"]["requests"] == {"rows": 3, "unique": 3}
    assert manifest["stats"]["responses"] == {
        "rows": 4,
        "succeeded": 3,
        "failed": 1,
        "success_unique": 3,
    }
    assert len(manifest["batches"]) == 5
    assert first_result["path"] == "requests/part-000001.jsonl"


def test_ledger_refuses_duplicate_input_corruption_and_orphan_shards(tmp_path: Path) -> None:
    ledger = JsonlShardLedger(tmp_path / "ledger", batch_size=4)
    record = _request(1)
    with pytest.raises(LedgerIntegrityError, match="duplicate request identity"):
        ledger.schedule_requests([record, record])

    ledger.schedule_requests([record])
    manifest = ledger.manifest()
    shard = ledger.root / manifest["shards"][0]["path"]
    shard.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(LedgerIntegrityError, match="hash changed"):
        ledger.verify()

    orphan = JsonlShardLedger(tmp_path / "orphan", batch_size=1)
    orphan.initialize()
    path = orphan.root / "requests" / "part-000001.jsonl"
    path.parent.mkdir()
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(LedgerIntegrityError, match="manifest/shard mismatch"):
        orphan.verify()


def test_ledger_recovers_valid_orphan_without_reissuing_request(
    tmp_path: Path,
) -> None:
    ledger = JsonlShardLedger(tmp_path / "recover", batch_size=2)
    request = _request(1)
    ledger.schedule_requests([request])
    response = ResponseRecord(
        request_id=request.request_id,
        prompt_hash=request.prompt_hash,
        status="succeeded",
        response={"query": "cached"},
        attempt=1,
    )
    row = response.to_row()
    orphan = ledger.root / "responses" / "part-000001.jsonl"
    orphan.parent.mkdir()
    orphan.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    recovered = ledger.recover()

    assert recovered["batches"][-1]["recovered"] is True
    assert ledger.schedule_requests([request]).records == ()
    cached = ledger.successful_response(request.request_id)
    assert cached is not None
    assert cached.response == {"query": "cached"}


def test_embedding_ledger_preserves_failed_attempt_and_reuses_success(tmp_path: Path) -> None:
    ledger = JsonlShardLedger.from_checkpointing(
        tmp_path / "embeddings",
        {"embedding_batch_records": 1, "llm_batch_records": 7},
        kind="embedding",
    )
    pending = EmbeddingRecord.from_text(
        "embed-candidates",
        "weather forecast",
        status="failed",
        item_key="weather",
        model="embedding-v1",
    )
    first = ledger.schedule_embeddings([pending])
    assert first.records == (pending,)
    assert first.retried == 0
    ledger.record_embeddings(first.records)

    retry = ledger.schedule_embeddings([pending])
    assert retry.records == (pending,)
    assert retry.retried == 1
    completed = EmbeddingRecord.from_text(
        "embed-candidates",
        "weather forecast",
        status="succeeded",
        item_key="weather",
        model="embedding-v1",
        vector=[0.25, -0.5],
    )
    result = ledger.record_embeddings([completed])
    assert result.accepted == 1
    assert result.batch_id == 2

    resumed = ledger.schedule_embeddings([pending])
    assert resumed.records == ()
    assert resumed.skipped_succeeded == 1
    manifest = ledger.verify()
    assert manifest["batch_size"] == 1
    assert manifest["stats"]["embeddings"] == {
        "rows": 2,
        "succeeded": 1,
        "failed": 1,
        "success_unique": 1,
    }
    cached = ledger.successful_embedding(completed.embedding_id)
    assert cached is not None
    assert cached.vector == (0.25, -0.5)


def test_llm_checkpointing_field_sets_provider_batch_size(tmp_path: Path) -> None:
    ledger = JsonlShardLedger.from_checkpointing(
        tmp_path / "llm",
        {"llm_batch_records": 2, "embedding_batch_records": 9},
        kind="llm",
    )
    batch = ledger.schedule_requests([_request(1), _request(2), _request(3)])
    assert ledger.batch_size == 2
    assert len(batch.records) == 2
    assert batch.newly_recorded == 2
