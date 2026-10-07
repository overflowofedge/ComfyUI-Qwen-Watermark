import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


path = Path(__file__).resolve().parents[1] / "scripts/run_batch.py"
spec = importlib.util.spec_from_file_location("run_batch", path)
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def test_batch_continues_after_failure_and_records_success(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "a.png").write_bytes(b"invalid")
    (inputs / "b.png").write_bytes(b"valid fixture")
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    (workflows / "qwen21_watermark.api.json").write_text("{}")
    output = tmp_path / "results"
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_batch", str(inputs), "--output", str(output)])
    calls = []

    def execute(command, **kwargs):
        image = Path(command[2])
        calls.append(image.name)
        if image.name == "a.png":
            return SimpleNamespace(returncode=1)
        target = Path(command[command.index("--output") + 1])
        (target / "report.json").write_text('{"status":"needs_review"}')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(batch.subprocess, "run", execute)
    assert batch.main() == 1
    assert calls == ["a.png", "b.png"]
    assert [item["status"] for item in json.loads((output / "batch_report.json").read_text())] == ["failed", "needs_review"]


def test_resume_reuses_matching_input_and_invalidates_changed_input(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source = inputs / "image.png"
    source.write_bytes(b"first input")
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    workflow = workflows / "qwen21_watermark.api.json"
    workflow.write_bytes(b"{}")
    output = tmp_path / "results"
    target = output / source.name
    target.mkdir(parents=True)
    restored = target / "restored.png"
    restored.write_bytes(b"output")
    report = {"status": "needs_review", "files": {"restored": str(restored)},
              "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "workflow_sha256": hashlib.sha256(workflow.read_bytes()).hexdigest(),
              "options": {"skip_residual": False}}
    (target / "report.json").write_text(json.dumps(report))
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_batch", str(inputs), "--output", str(output), "--resume"])
    calls = []
    monkeypatch.setattr(batch.subprocess, "run", lambda *args, **kwargs: calls.append(args[0]) or SimpleNamespace(returncode=1))
    assert batch.main() == 0
    assert not calls
    source.write_bytes(b"changed input")
    assert batch.main() == 1
    assert len(calls) == 1
