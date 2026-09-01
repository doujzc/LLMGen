"""Frozen Light and closed-set fixture regressions at the Runner boundary.

The checked-in Light snapshot is intentionally not copied wholesale in CI: its
training JSONL is tens of megabytes.  This test pins the candidate snapshot by
hash/count/anchor IDs and extracts one real multi-target query as an ordered-
qrels regression sample.  ``tests/fixtures/skillret`` is the repository's
small frozen closed-set fixture used to exercise code and SFT compatibility.
No network, Provider, GPU, or model download is needed.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
from typing import Any

from llmgen.pipeline.config import load_pipeline_config
from llmgen.pipeline.io import atomic_write_json, atomic_write_jsonl, read_json, read_jsonl, sha256_file
from llmgen.pipeline.runner import create_pipeline_run
from llmgen.pipeline.schema import ensure_ordered_qrels, validate_ordered_qrels
from llmgen.pipeline.stages import ArtifactOutput, StageResult, StageSpec
from llmgen.pipeline.stages.ingest import ingest
from llmgen.router_bundle import dump_router_decoder_artifacts


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "router_pipeline.yaml"
LIGHT_CANDIDATES = ROOT / "data_light" / "candidates.jsonl"
LIGHT_FINAL = ROOT / "data_light" / "final"
FROZEN = ROOT / "tests" / "fixtures" / "skillret"

LIGHT_CANDIDATES_SHA256 = "a714b1b01d97439331aa33d73cb67d758c85afbd493b22af64308f92c17a05c2"
FROZEN_HASHES = {
    "catalog_train.jsonl": "437c2e9d29e5c87e7d3e9d1d1673f0a2c758f78f44e7fd0761a0ac60881a1bbd",
    "queries_train.jsonl": "2fcc0a5cd0abce5dce27fe951a940651401b3e3260360ba5965c199cdbd677be",
    "qrels_train.jsonl": "4557bd217e818352cfa88748374fd1142d604a1e8f2423b3f53823a0fac01a23",
    "train_codes.jsonl": "6756c06e7bd859664ae1dbb9df675b91fea838187a1c81c00e603e0eee32291a",
    "train_registry.json": "b38979d48edb6572a8dba050078d30786412e402e9fc48f78720233c5f528ec8",
    "virtual_tokens.txt": "272011c9921dced4955964c5b7162a17be070b6aff0c604d58e87ae4e07a134f",
}


def _first_multiskill_light_sample() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read one historical Light query and only its qrels, not all 96k rows."""

    selected: dict[str, Any] | None = None
    with (LIGHT_FINAL / "queries_train.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if len(row.get("skill_ids") or []) > 1:
                selected = row
                break
    assert selected is not None
    query_id = selected["id"]
    qrels: list[dict[str, Any]] = []
    with (LIGHT_FINAL / "qrels_train.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("query_id") == query_id:
                qrels.append(row)
    return selected, qrels


def _fixture_stage(context) -> StageResult:
    """Materialize and verify the frozen code/SFT/bundle contract in one Stage."""

    output = context.output_dir
    dataset = output / "dataset"
    index = output / "index"
    router_data = output / "router_data"
    bundle = output / "bundle"
    dataset.mkdir(parents=True)
    index.mkdir()
    for name in FROZEN_HASHES:
        source = FROZEN / name
        destination = (dataset / name) if name in {"catalog_train.jsonl", "queries_train.jsonl", "qrels_train.jsonl"} else (index / name)
        shutil.copy2(source, destination)
    atomic_write_json(dataset / "manifest.json", {"artifacts": {name: {} for name in ("catalog_train.jsonl", "queries_train.jsonl", "qrels_train.jsonl")}})
    ensure_ordered_qrels(dataset)
    validate_ordered_qrels(dataset)
    context.run_command(
        [
            sys.executable,
            "scripts/build_router_data.py",
            "--catalog", str(dataset / "catalog_train.jsonl"),
            "--queries", str(dataset / "queries_train.jsonl"),
            "--qrels", str(dataset / "qrels_train.jsonl"),
            "--codes", str(index / "train_codes.jsonl"),
            "--virtual-tokens", str(index / "virtual_tokens.txt"),
            "--output-dir", str(router_data),
            "--memorization-validation-fraction", "0",
            "--retrieval-validation-fraction", "0",
            "--seed", "42",
        ],
        label="frozen-closedset-build-sft",
    )
    dump_router_decoder_artifacts(
        output_dir=bundle,
        catalog_path=dataset / "catalog_train.jsonl",
        codes_path=index / "train_codes.jsonl",
        registry_path=index / "train_registry.json",
        virtual_tokens_path=index / "virtual_tokens.txt",
        training_data_path=router_data / "retrieval_train.jsonl",
        supervision_phase="retrieval",
    )
    atomic_write_json(bundle / "config.json", {"model_type": "fixture"})
    (bundle / "model.safetensors").write_bytes(b"fixture weights")
    return StageResult(
        artifacts=(
            ArtifactOutput("fixture.dataset", dataset, "closedset_dataset/v3"),
            ArtifactOutput("fixture.codes", index, "skill_code_index/v1"),
            ArtifactOutput("fixture.sft", router_data, "router_sft_bundle/v1"),
            ArtifactOutput("fixture.bundle", bundle, "deployable_router/v1"),
        )
    )


def test_runner_preserves_light_input_and_closedset_fixture_contract(tmp_path: Path) -> None:
    """Protect the reusable historical inputs without requiring their full replay."""

    assert sha256_file(LIGHT_CANDIDATES) == LIGHT_CANDIDATES_SHA256
    assert {name: sha256_file(FROZEN / name) for name in FROZEN_HASHES} == FROZEN_HASHES
    base_model = tmp_path / "base-model"
    base_model.mkdir()
    atomic_write_json(base_model / "config.json", {"model_type": "fixture"})
    config = load_pipeline_config(
        CONFIG,
        candidates=LIGHT_CANDIDATES,
        output=tmp_path / "run",
        overrides=(
            f"router.base_model={base_model}",
            "runtime.device=cpu",
            "runtime.num_devices=1",
            "runtime.deepspeed=none",
        ),
        environment={"PIPELINE_PYTHON": str(sys.executable)},
    )
    specs = (
        StageSpec("ingest", "00_ingest", (), (), ingest, "freeze Light candidates"),
        StageSpec("fixture-regression", "01_fixture_regression", ("ingest",), ("candidates.manifest",), _fixture_stage, "build frozen closed-set SFT"),
    )
    runner = create_pipeline_run(config, stage_specs=specs, repo_root=ROOT)
    runner.run()

    normalized = read_jsonl(runner.registry.resolve("candidates.normalized"))
    assert len(normalized) == 301
    assert [row["skill_id"] for row in normalized[:3]] == [
        "tc-chengxin", "shareai-lab-learn-claude-code-pdf", "viral-product-copywriting-generator",
    ]
    assert normalized[-1]["skill_id"] == "superdesign"
    candidate_input = read_json(runner.run_dir / "config" / "candidate_input.json")
    assert candidate_input["sha256"] == LIGHT_CANDIDATES_SHA256

    dataset = runner.registry.resolve("fixture.dataset")
    qrels = read_jsonl(dataset / "qrels_train.jsonl")
    by_query: dict[str, list[dict[str, Any]]] = {}
    for row in qrels:
        by_query.setdefault(row["query_id"], []).append(row)
    assert [row["position"] for row in by_query["q2"]] == [0, 1]
    assert [row["skill_id"] for row in by_query["q2"]] == ["skill-b", "skill-c"]

    sft_manifest = read_json(runner.registry.resolve("fixture.sft") / "manifest.json")
    assert sft_manifest["counts"]["memorization"]["train_examples"] == 4
    assert sft_manifest["counts"]["retrieval"]["train_examples"] == 3
    bundle = runner.registry.resolve("fixture.bundle")
    import service

    service._validate_full_model_bundle(bundle)
    candidate_bundle = service._load_candidate_bundle(bundle)
    assert tuple(candidate_bundle.skills) == ("skill-a", "skill-b", "skill-c", "skill-d")
    output_manifest = read_json(runner.state.stage_dir("fixture-regression") / "output" / "manifest.json")
    assert {row["logical_name"] for row in output_manifest["artifacts"]} == {
        "fixture.dataset", "fixture.codes", "fixture.sft", "fixture.bundle",
    }


def test_light_historical_sample_retains_ordered_qrels_after_normalization(tmp_path: Path) -> None:
    """Exercise a real Light multi-skill sample without copying the full snapshot."""

    query, qrels = _first_multiskill_light_sample()
    root = tmp_path / "light-sample"
    root.mkdir()
    atomic_write_jsonl(root / "queries_train.jsonl", [query])
    atomic_write_jsonl(root / "qrels_train.jsonl", qrels)
    atomic_write_json(root / "manifest.json", {"artifacts": {"queries_train.jsonl": {}, "qrels_train.jsonl": {}}})
    ensure_ordered_qrels(root)
    validate_ordered_qrels(root)
    normalized = read_jsonl(root / "qrels_train.jsonl")
    assert [row["skill_id"] for row in normalized] == query["skill_ids"]
    assert [row["position"] for row in normalized] == list(range(len(query["skill_ids"])))
