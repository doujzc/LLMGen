from __future__ import annotations

from http.server import ThreadingHTTPServer
import json
import threading
from urllib.request import Request, urlopen

from web_server.server import handler_class


class StubRuntime:
    def health(self):
        return {"ready": True, "num_skills": 2, "num_paths": 2, "num_levels": 2}

    def catalog(self, query, limit):
        return {"total": 1, "skills": [{"skill_id": "s1", "name": query or "天气"}]}

    def skill_detail(self, skill_id):
        return {
            "skill_id": skill_id,
            "name": "天气",
            "description": "获取天气预报",
            "text": "天气 | 获取天气预报",
            "code_text": "<L1_0><L2_0>",
        }

    def infer(self, query, *, max_code_paths, top_k):
        return {
            "query": query,
            "generated_text": "<L1_0><L2_0>",
            "paths": [],
            "candidates": [],
            "request": {"max_code_paths": max_code_paths, "top_k": top_k},
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
            payload={"query": "帮我查天气", "max_code_paths": 3, "top_k": 7},
        )
        assert result["query"] == "帮我查天气"
        assert result["request"] == {"max_code_paths": 3, "top_k": 7}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
