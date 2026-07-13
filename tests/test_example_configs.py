import json
import runpy
from pathlib import Path

from llmgen import TokenizerConfig, create_tokenizer


def test_example_configs_are_loadable() -> None:
    config_dir = Path(__file__).parents[1] / "examples" / "configs"

    for path in sorted(config_dir.glob("*.json")):
        config = TokenizerConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))
        tokenizer = create_tokenizer(config)

        assert tokenizer.config.num_levels == len(config.branching_factors)
        assert len(tokenizer.special_tokens) == sum(config.branching_factors)


def test_quickstart_is_executable() -> None:
    path = Path(__file__).parents[1] / "examples" / "quickstart.py"

    runpy.run_path(str(path), run_name="__main__")
