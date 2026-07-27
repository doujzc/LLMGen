from __future__ import annotations

from argparse import Namespace
from http.server import ThreadingHTTPServer
import json
import threading
from urllib.request import Request, urlopen

import pytest

from llmgen.router import RouterDataError
from web_server.server import handler_class
from web_server.runtime import RouterRuntime


class StubRuntime:
    def health(self):
        return {
            "ready": True,
            "num_skills": 2,
            "num_paths": 2,
            "num_levels": 2,
            "max_code_paths": 8,
            "max_num_beams": 8,
            "max_batch_queries": 1000,
            "max_batch_size": 8,
        }

    def catalog(self, query, limit):
        return {
            "total": 1,
            "skills": [{"skill_id": "s1", "name": query or "天气"}],
        }

    def skill_detail(self, skill_id):
        return {
            "skill_id": skill_id,
            "name": "天气",
            "description": "获取天气预报",
            "text": "天气 | 获取天气预报",
            "code_text": "<L1_0><L2_0>",
        }

    def infer(
        self,
        query,
        *,
        max_code_paths,
        top_k,
        decoding_mode,
        num_beams,
    ):
        return {
            "query": query,
            "generated_text": "<L1_0><L2_0>",
            "paths": [],
            "candidates": [],
            "request": {
                "max_code_paths": max_code_paths,
                "top_k": top_k,
                "decoding_mode": decoding_mode,
                "num_beams": num_beams,
            },
        }

    def infer_batch(
        self,
        queries,
        *,
        batch_size,
        max_code_paths,
        top_k,
        decoding_mode,
        num_beams,
    ):
        return {
            "num_queries": len(queries),
            "latency_ms": 12.5,
            "queries_per_second": 160.0,
            "request": {
                "batch_size": batch_size,
                "max_code_paths": max_code_paths,
                "top_k": top_k,
                "decoding_mode": decoding_mode,
                "num_beams": num_beams,
            },
            "results": [
                {
                    "query_id": f"query-{index:06d}",
                    "query": query,
                    "paths": [],
                    "candidates": [],
                }
                for index, query in enumerate(queries, start=1)
            ],
        }


def _request(url, *, payload=None):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body else {},
    )
    with urlopen(request, timeout=2) as response:
        return response.status, json.loads(response.read())


def test_runtime_catalog_uses_the_only_candidate_set() -> None:
    runtime = RouterRuntime.__new__(RouterRuntime)
    runtime.skills = {
        "s1": {"skill_id": "s1", "name": "候选一"},
        "s2": {"skill_id": "s2", "name": "候选二"},
    }
    runtime.decode_map = {
        "skill_to_code": {
            "s1": {"code_text": "<L1_0><L2_0>"},
            "s2": {"code_text": "<L1_1><L2_1>"},
        }
    }

    result = runtime.catalog()

    assert result["total"] == 2
    assert {row["skill_id"] for row in result["skills"]} == {"s1", "s2"}


def test_runtime_catalog_can_return_the_complete_candidate_set() -> None:
    runtime = RouterRuntime.__new__(RouterRuntime)
    runtime.skills = {
        f"s{index:04d}": {
            "skill_id": f"s{index:04d}",
            "name": f"候选 {index:04d}",
        }
        for index in range(1002)
    }
    runtime.decode_map = {
        "skill_to_code": {
            skill_id: {"code_text": f"<L1_{index % 32}><L2_{index}>"}
            for index, skill_id in enumerate(runtime.skills)
        }
    }

    result = runtime.catalog(limit=len(runtime.skills))

    assert result["total"] == 1002
    assert len(result["skills"]) == 1002


def test_runtime_rejects_beam_width_above_server_limit() -> None:
    runtime = RouterRuntime.__new__(RouterRuntime)
    runtime.max_code_paths = 8
    runtime.max_num_beams = 8
    runtime.args = Namespace()

    with pytest.raises(RouterDataError, match="between 2 and 8"):
        runtime.infer(
            "帮我查天气",
            decoding_mode="beam_search",
            num_beams=16,
        )


def test_runtime_beam_search_forces_one_code_per_beam(monkeypatch) -> None:
    runtime = RouterRuntime.__new__(RouterRuntime)
    runtime.max_code_paths = 8
    runtime.max_num_beams = 8
    runtime.args = Namespace()
    runtime._lock = threading.Lock()
    runtime.tokenizer = object()
    runtime.model = object()
    runtime.torch = object()
    runtime.id_to_token = {}
    runtime.buckets = {}
    runtime.skills = {}
    runtime.decode_map = {"skill_to_code": {}}
    trie_requests = []
    runtime._trie = (
        lambda max_code_paths: trie_requests.append(max_code_paths) or object()
    )

    def fake_generate_batch(**kwargs):
        assert kwargs["args"].max_code_paths == 1
        return [
            {
                "query_id": "interactive",
                "query": "帮我查天气",
                "generated_text": "",
                "decoding": {
                    "mode": "beam_search",
                    "num_beams": 4,
                    "scope": "single_code_top_k",
                    "num_return_sequences": 4,
                },
                "paths": [],
                "candidates": [],
            }
        ]

    monkeypatch.setattr("web_server.runtime._generate_batch", fake_generate_batch)

    result = runtime.infer(
        "帮我查天气",
        max_code_paths=6,
        decoding_mode="beam_search",
        num_beams=4,
    )

    assert trie_requests == [1]
    assert result["request"]["max_code_paths"] == 1
    assert result["request"]["num_beams"] == 4


def test_runtime_batch_preserves_order_and_splits_model_batches(monkeypatch) -> None:
    runtime = RouterRuntime.__new__(RouterRuntime)
    runtime.max_code_paths = 8
    runtime.max_num_beams = 8
    runtime.max_batch_queries = 10
    runtime.max_batch_size = 4
    runtime.args = Namespace()
    runtime._lock = threading.Lock()
    runtime.tokenizer = object()
    runtime.model = object()
    runtime.torch = object()
    runtime.id_to_token = {}
    runtime.buckets = {}
    runtime.skills = {}
    runtime.decode_map = {"skill_to_code": {}}
    runtime._trie = lambda max_code_paths: object()
    model_batch_sizes = []

    def fake_generate_batch(**kwargs):
        batch = kwargs["batch"]
        model_batch_sizes.append(len(batch))
        return [
            {
                "query_id": row["id"],
                "query": row["query"],
                "generated_text": "",
                "decoding": {"mode": "greedy", "num_beams": 1},
                "paths": [],
                "candidates": [],
            }
            for row in batch
        ]

    monkeypatch.setattr("web_server.runtime._generate_batch", fake_generate_batch)

    result = runtime.infer_batch(
        ["第一个", "重复", "重复", "第四个", "第五个"],
        batch_size=2,
    )

    assert model_batch_sizes == [2, 2, 1]
    assert [row["query"] for row in result["results"]] == [
        "第一个",
        "重复",
        "重复",
        "第四个",
        "第五个",
    ]
    assert [row["batch_index"] for row in result["results"]] == list(range(5))
    assert result["request"]["batch_size"] == 2


def test_web_api_health_catalog_and_inference() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_class(StubRuntime()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        status, health = _request(base + "/api/health")
        assert status == 200
        assert health["num_skills"] == 2

        _, catalog = _request(base + "/api/catalog?q=%E5%A4%A9%E6%B0%94&limit=5")
        assert catalog["skills"][0]["name"] == "天气"

        _, detail = _request(base + "/api/skill?id=s1")
        assert detail["text"] == "天气 | 获取天气预报"

        _, result = _request(
            base + "/api/infer",
            payload={
                "query": "帮我查天气",
                "max_code_paths": 3,
                "top_k": 7,
                "decoding_mode": "beam_search",
                "num_beams": 4,
            },
        )
        assert result["query"] == "帮我查天气"
        assert result["request"] == {
            "max_code_paths": 3,
            "top_k": 7,
            "decoding_mode": "beam_search",
            "num_beams": 4,
        }

        _, greedy = _request(
            base + "/api/infer",
            payload={"query": "帮我查天气"},
        )
        assert greedy["request"]["decoding_mode"] == "greedy"
        assert greedy["request"]["num_beams"] == 1

        _, batch = _request(
            base + "/api/infer-batch",
            payload={
                "queries": ["查天气", "设置明天的提醒"],
                "batch_size": 2,
                "max_code_paths": 3,
                "top_k": 7,
            },
        )
        assert batch["num_queries"] == 2
        assert [row["query"] for row in batch["results"]] == [
            "查天气",
            "设置明天的提醒",
        ]
        assert batch["request"]["batch_size"] == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
