"""Long-lived model runtime shared by web inference requests."""

from __future__ import annotations

from argparse import Namespace
from functools import lru_cache
from pathlib import Path
import threading
import time
from typing import Any

from llmgen.router import (
    MultiPathTokenTrie,
    RouterDataError,
    code_token_id_map,
    load_virtual_tokens,
)
from llmgen.router_bundle import (
    BUNDLED_VIRTUAL_TOKENS_FILENAME,
    DECODE_MAP_FILENAME,
    load_skill_decode_map,
)
from scripts.infer_router import _generate_batch, _load_model_and_tokenizer


class RouterRuntime:
    """Load a trained router once and expose thread-safe interactive inference."""

    def __init__(
        self,
        *,
        model_dir: str | Path,
        base_model_name_or_path: str | None,
        device: str,
        dtype: str,
        max_code_paths: int,
        max_input_length: int | None,
        trust_remote_code: bool,
    ) -> None:
        if max_code_paths < 1:
            raise RouterDataError("max_code_paths must be positive")
        self.model_dir = Path(model_dir).expanduser().resolve()
        if not self.model_dir.is_dir():
            raise RouterDataError(f"model directory does not exist: {self.model_dir}")
        self.decode_map_path = self.model_dir / DECODE_MAP_FILENAME
        self.virtual_tokens_path = (
            self.model_dir / BUNDLED_VIRTUAL_TOKENS_FILENAME
        )
        if not self.decode_map_path.is_file() or not self.virtual_tokens_path.is_file():
            raise RouterDataError(
                "model dump is missing skill_decode_map.json or virtual_tokens.txt; "
                "run scripts/export_router_bundle.py first"
            )
        self.decode_map = load_skill_decode_map(self.decode_map_path)
        self.skills: dict[str, dict[str, Any]] = self.decode_map["skills"]
        self.supervision = self.decode_map.get("supervision")
        self.supervised_skill_ids = (
            set(self.supervision.get("supervised_skill_ids", ()))
            if isinstance(self.supervision, dict)
            else None
        )
        self.buckets = {
            tuple(path["tokens"]): tuple(path["skill_ids"])
            for path in self.decode_map["paths"]
        }
        self.max_code_paths = max_code_paths
        self._lock = threading.Lock()
        self.args = Namespace(
            model_name_or_path=str(self.model_dir),
            base_model_name_or_path=base_model_name_or_path,
            virtual_tokens=str(self.virtual_tokens_path),
            decode_map=str(self.decode_map_path),
            device=device,
            dtype=dtype,
            max_code_paths=max_code_paths,
            max_input_length=max_input_length,
            trust_remote_code=trust_remote_code,
            system_prompt=(
                "Select every Agent Skill needed for the user request in execution "
                "order. Output one hierarchical skill code per line, with no other text."
            ),
            top_k=20,
            _num_levels=int(self.decode_map["num_levels"]),
        )
        virtual_tokens = tuple(self.decode_map["virtual_tokens"])
        if load_virtual_tokens(self.virtual_tokens_path) != virtual_tokens:
            raise RouterDataError(
                "bundled virtual_tokens.txt disagrees with skill_decode_map.json"
            )
        self.torch, self.tokenizer, self.model, token_ids = _load_model_and_tokenizer(
            self.args, virtual_tokens
        )
        # Validate every bundled virtual token against the checkpoint tokenizer,
        # including currently unused namespace entries.
        code_token_id_map(self.tokenizer, virtual_tokens)
        self.token_ids = token_ids
        self.id_to_token = {token_id: token for token, token_id in token_ids.items()}
        try:
            self.active_token_paths = tuple(
                tuple(token_ids[token] for token in path) for path in self.buckets
            )
        except KeyError as exc:
            raise RouterDataError(
                f"bundled path uses unknown virtual token: {exc.args[0]!r}"
            ) from exc

    @lru_cache(maxsize=16)
    def _trie(self, max_code_paths: int) -> MultiPathTokenTrie:
        return MultiPathTokenTrie(
            self.active_token_paths,
            eos_token_id=int(self.tokenizer.eos_token_id),
            separator_token_ids=self.args._separator_token_ids,
            max_paths=max_code_paths,
        )

    def health(self) -> dict[str, Any]:
        return {
            "ready": True,
            "model_dir": str(self.model_dir),
            "device": str(self.args.device),
            "dtype": self.args.dtype,
            "num_skills": int(self.decode_map["num_skills"]),
            "num_supervised_skills": (
                len(self.supervised_skill_ids)
                if self.supervised_skill_ids is not None
                else None
            ),
            "supervision_phase": (
                self.supervision.get("phase")
                if isinstance(self.supervision, dict)
                else None
            ),
            "num_paths": int(self.decode_map["num_paths"]),
            "num_levels": int(self.decode_map["num_levels"]),
            "max_code_paths": self.max_code_paths,
        }

    def catalog(
        self,
        query: str = "",
        limit: int = 100,
        *,
        supervised_only: bool = False,
    ) -> dict[str, Any]:
        needle = query.casefold().strip()
        rows = []
        for skill_id, metadata in self.skills.items():
            if (
                supervised_only
                and self.supervised_skill_ids is not None
                and skill_id not in self.supervised_skill_ids
            ):
                continue
            searchable = " ".join(
                str(metadata.get(field, ""))
                for field in ("skill_id", "name", "description", "domain")
            ).casefold()
            if needle and needle not in searchable:
                continue
            code = self.decode_map["skill_to_code"][skill_id]
            rows.append({**metadata, **code})
        rows.sort(
            key=lambda row: (
                str(row.get("name", "")).casefold(),
                row["skill_id"],
            )
        )
        return {
            "total": len(rows),
            "skills": rows[: max(1, min(limit, 1000))],
            "supervised_only": supervised_only,
            "supervision_available": self.supervised_skill_ids is not None,
        }

    def skill_detail(self, skill_id: str) -> dict[str, Any]:
        metadata = self.skills.get(skill_id)
        if metadata is None:
            raise RouterDataError(f"unknown skill_id: {skill_id!r}")
        return {
            **metadata,
            **self.decode_map["skill_to_code"][skill_id],
        }

    def infer(
        self,
        query: str,
        *,
        max_code_paths: int = 4,
        top_k: int = 10,
    ) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise RouterDataError("query must not be empty")
        if len(query) > 20_000:
            raise RouterDataError("query is too long (maximum 20,000 characters)")
        if not 1 <= max_code_paths <= self.max_code_paths:
            raise RouterDataError(
                f"max_code_paths must be between 1 and {self.max_code_paths}"
            )
        if not 1 <= top_k <= 100:
            raise RouterDataError("top_k must be between 1 and 100")
        request_args = Namespace(**vars(self.args))
        request_args.max_code_paths = max_code_paths
        request_args.top_k = top_k
        started = time.perf_counter()
        with self._lock:
            result = _generate_batch(
                batch=[{"id": "interactive", "query": query}],
                tokenizer=self.tokenizer,
                model=self.model,
                torch=self.torch,
                trie=self._trie(max_code_paths),
                id_to_token=self.id_to_token,
                buckets=self.buckets,
                args=request_args,
            )[0]
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        for path in result["paths"]:
            path["skills"] = [self.skills[skill_id] for skill_id in path["skill_ids"]]
        for candidate in result["candidates"]:
            skill_id = candidate["skill_id"]
            candidate.update(self.skills[skill_id])
            candidate["code_text"] = self.decode_map["skill_to_code"][skill_id][
                "code_text"
            ]
        result["request"] = {
            "max_code_paths": max_code_paths,
            "top_k": top_k,
        }
        return result
