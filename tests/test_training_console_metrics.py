from __future__ import annotations

from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import urlencode
from urllib.request import urlopen

from training_console.metrics import LossMetricReader
from training_console.server import TrainingConsoleService, handler_class


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_loss_reader_extracts_phases_and_incremental_trainer_metrics(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "train.log"
    log_path.write_text(
        "\x1b[32m[05] train router memorization phase\x1b[0m\n"
        " 10%|##        | 10/100 [00:03<00:30]\r"
        "{'loss': 2.4, 'grad_norm': 8.5, "
        "'learning_rate': 1e-05, 'epoch': 0.1}\n"
        "{'loss': 2.4, 'grad_norm': 8.5, "
        "'learning_rate': 1e-05, 'epoch': 0.1}\n"
        "{'eval_loss': 2.1, 'eval_runtime': 3.0, 'epoch': 0.1}\n"
        "[06a] single-skill retrieval alignment curriculum\n"
        "5/50\n"
        "{'loss': 1.8, 'grad_norm': 4.0, "
        "'learning_rate': 8e-06, 'epoch': 0.2}\n",
        encoding="utf-8",
    )
    reader = LossMetricReader()

    first = reader.read(log_path, maximum_points=100)

    assert first["total_points"] == 3
    assert [point["kind"] for point in first["points"]] == [
        "train",
        "eval",
        "train",
    ]
    assert first["points"][0] == {
        "sequence": 1,
        "phase_id": "memorization",
        "phase": "05 Memorization",
        "kind": "train",
        "loss": 2.4,
        "step": 10,
        "total_steps": 100,
        "epoch": 0.1,
        "total_epochs": None,
        "learning_rate": 1e-05,
        "grad_norm": 8.5,
    }
    assert first["points"][1]["step"] == 10
    assert first["points"][2]["phase_id"] == "alignment"
    assert first["points"][2]["step"] == 5
    assert first["best_eval"]["loss"] == 2.1

    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(
            "[06b] multi-Skill retrieval\n"
            "7/70\n"
            '{"loss": 1.5, "grad_norm": 3.2, '
            '"learning_rate": 5e-06, "epoch": 0.1}\n'
        )

    second = reader.read(log_path, maximum_points=100)

    assert second["total_points"] == 4
    assert second["points"][-1]["phase_id"] == "retrieval"
    assert second["points"][-1]["step"] == 7
    assert [phase["id"] for phase in second["phases"]] == [
        "memorization",
        "alignment",
        "retrieval",
    ]


def test_loss_reader_supports_toolweaver_tokenizer_progress(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "train.log"
    log_path.write_text(
        '{"event": "stage1_progress", "epoch": 25, "epochs": 2000, '
        '"global_step": 250, "loss": 0.083, "collision_rate": 0.01}\n',
        encoding="utf-8",
    )

    payload = LossMetricReader().read(log_path)

    assert payload["points"][0]["phase_id"] == "tokenizer"
    assert payload["points"][0]["step"] == 250
    assert payload["points"][0]["epoch"] == 25
    assert payload["points"][0]["total_epochs"] == 2000
    assert payload["points"][0]["loss"] == 0.083


def test_loss_reader_rebuilds_after_log_truncation(tmp_path: Path) -> None:
    log_path = tmp_path / "train.log"
    log_path.write_text(
        "[05] memorization\n"
        + ("padding\n" * 30)
        + "{'loss': 4.0, 'epoch': 0.1}\n",
        encoding="utf-8",
    )
    reader = LossMetricReader()
    assert reader.read(log_path)["total_points"] == 1

    log_path.write_text(
        "[06b] retrieval\n"
        "{'loss': 0.75, 'epoch': 0.2}\n",
        encoding="utf-8",
    )

    rebuilt = reader.read(log_path)

    assert rebuilt["total_points"] == 1
    assert rebuilt["points"][0]["phase_id"] == "retrieval"
    assert rebuilt["points"][0]["loss"] == 0.75


def test_run_metrics_endpoint_and_loss_chart_assets(tmp_path: Path) -> None:
    service = TrainingConsoleService(
        repo_root=REPOSITORY_ROOT,
        state_root=tmp_path / "state",
        inference_url="",
        launch_enabled=False,
    )
    profile = service.store.save_profile(
        profile_id="loss-endpoint",
        dataset="clawhub",
        command="train-memorization",
        overrides={},
        resolved={},
    )
    run = service.store.create_run(
        profile["profile_id"],
        profile["version"],
        runtime_environment={},
    )
    Path(run["log_path"]).write_text(
        "[05] memorization\n"
        "3/30\n"
        "{'loss': 2.25, 'learning_rate': 1e-5, 'epoch': 0.1}\n",
        encoding="utf-8",
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_class(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        query = urlencode({"id": run["run_id"], "limit": 100})
        with urlopen(
            f"http://127.0.0.1:{server.server_port}/api/run-metrics?{query}",
            timeout=5,
        ) as response:
            payload = json.loads(response.read())
        assert payload["run_id"] == run["run_id"]
        assert payload["source"] == "train.log"
        assert payload["points"][0]["loss"] == 2.25
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    static_dir = REPOSITORY_ROOT / "training_console/static"
    page = (static_dir / "index.html").read_text(encoding="utf-8")
    app = (static_dir / "app.js").read_text(encoding="utf-8")
    styles = (static_dir / "styles.css").read_text(encoding="utf-8")
    assert 'id="loss-chart"' in page
    assert "loadMonitorMetrics" in app
    assert "renderLossMonitor" in app
    assert ".loss-monitor-panel" in styles
