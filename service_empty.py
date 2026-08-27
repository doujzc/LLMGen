#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""No-op deployment service used to boot a host for manual debugging.

This file is deliberately self-contained.  It does not inspect deployment
paths, load model artifacts, import vLLM, or allocate accelerator resources.
"""

from __future__ import annotations

from typing import Any


class RetriverTest:
    """Framework-compatible service whose lifecycle always succeeds."""

    def __init__(self) -> None:
        pass

    def load(self) -> None:
        """Accept service initialization without doing any work."""

    def calc(self, req_data: Any = None) -> str:
        """Ignore every request and return an empty JSON result list."""

        del req_data
        return "[]"

    def close(self) -> None:
        """Accept service shutdown without doing any work."""


# Keep the historical framework class name while also offering a clear alias.
SkillRouterService = RetriverTest


if __name__ == "__main__":
    service = RetriverTest()
    service.load()
    print(service.calc(None))
    service.close()
