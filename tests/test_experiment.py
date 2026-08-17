from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from llmgen.experiment import (
    RunStore,
    TrainingLogCallback,
    load_and_verify_model_artifact,
    make_training_log_callback,
    write_model_artifact_manifest,
)
from llmgen.top1 import Top1DataError, read_jsonl


class ExperimentTests(unittest.TestCase):
    def test_training_callback_inherits_transformers_lifecycle_defaults(self) -> None:
        class FakeTrainerCallback:
            def on_init_end(
                self,
                args: object,
                state: object,
                control: object,
                **kwargs: object,
            ) -> None:
                del args, state, control, kwargs

            def on_epoch_begin(
                self,
                args: object,
                state: object,
                control: object,
                **kwargs: object,
            ) -> None:
                del args, state, control, kwargs

        with tempfile.TemporaryDirectory() as temporary:
            store = RunStore.training(Path(temporary) / "run")
            callback = make_training_log_callback(store, FakeTrainerCallback)
            control = object()

            self.assertIsInstance(callback, TrainingLogCallback)
            self.assertIsInstance(callback, FakeTrainerCallback)
            self.assertIsNone(callback.on_init_end(object(), object(), control))
            self.assertIsNone(callback.on_epoch_begin(object(), object(), control))

    def test_training_run_manifest_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = RunStore.training(Path(temporary) / "run")
            manifest = {"run_signature": "same", "run_id": "run-1"}
            store.initialize(manifest)

            with self.assertRaisesRegex(Top1DataError, "already exists"):
                store.initialize(manifest)
            store.update_status("FAILED")
            store.initialize(manifest, resume=True)
            self.assertEqual(
                json.loads(store.status_path.read_text())["state"],
                "RESUMING",
            )
            with self.assertRaisesRegex(Top1DataError, "does not match"):
                store.initialize({"run_signature": "changed"}, resume=True)

    def test_model_artifact_detects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary) / "model"
            model_dir.mkdir()
            weights = model_dir / "weights.bin"
            weights.write_bytes(b"weights")
            artifact = write_model_artifact_manifest(
                model_dir,
                training_run_id="run-1",
            )

            loaded = load_and_verify_model_artifact(model_dir, verify_files=True)
            self.assertEqual(loaded["model_id"], artifact["model_id"])
            weights.write_bytes(b"changed")
            with self.assertRaisesRegex(Top1DataError, "changed"):
                load_and_verify_model_artifact(model_dir, verify_files=True)

    def test_training_callback_writes_events_and_checkpoint_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = RunStore.training(Path(temporary) / "run")
            store.initialize({"run_signature": "signature"})
            callback = TrainingLogCallback(store)
            state = SimpleNamespace(
                is_world_process_zero=True,
                global_step=5,
                epoch=0.5,
            )
            control = object()
            callback.on_log(
                SimpleNamespace(output_dir=str(store.root / "checkpoints")),
                state,
                control,
                logs={"loss": 1.25, "learning_rate": 1e-5},
            )
            callback.on_save(
                SimpleNamespace(output_dir=str(store.root / "checkpoints")),
                state,
                control,
            )

            events = read_jsonl(store.events_path)
            self.assertEqual(events[-2]["event"], "trainer_log")
            self.assertEqual(events[-2]["metrics"]["loss"], 1.25)
            pointer = json.loads(
                (store.root / "checkpoints" / "last_checkpoint.json").read_text()
            )
            self.assertEqual(pointer["step"], 5)


if __name__ == "__main__":
    unittest.main()
