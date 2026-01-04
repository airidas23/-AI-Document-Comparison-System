from __future__ import annotations

import queue
from pathlib import Path

import pytest
from PIL import Image


class _DummyProcess:
    def __init__(self):
        self._alive = True
        self.pid = 12345
        self.killed = False
        self.terminated = False
        self.joined = False

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        self.joined = True

    def terminate(self):
        self.terminated = True
        self._alive = False

    def kill(self):
        self.killed = True
        self._alive = False


class _QueueStub:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


class _OutputStub:
    def __init__(self, result=None, exc: Exception | None = None):
        self._result = result
        self._exc = exc

    def get(self, timeout=None):
        if self._exc is not None:
            raise self._exc
        return self._result


def test_get_and_shutdown_singleton(monkeypatch):
    from extraction import deepseek_persistent_worker as dpw

    # Reset singleton
    monkeypatch.setattr(dpw, "_GLOBAL_WORKER", None)

    class _FakeWorker:
        def __init__(self, model_path: str):
            self.model_path = model_path
            self.stopped = False

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(dpw, "DeepSeekPersistentWorker", _FakeWorker)

    w1 = dpw.get_persistent_worker("/m")
    w2 = dpw.get_persistent_worker("/m")
    assert w1 is w2

    dpw.shutdown_all_workers()
    assert w1.stopped is True
    assert dpw._GLOBAL_WORKER is None


def test_infer_success_path(monkeypatch, tmp_path: Path):
    from extraction.deepseek_persistent_worker import DeepSeekPersistentWorker

    img_path = tmp_path / "img.png"
    Image.new("RGB", (10, 10), color=(255, 255, 255)).save(img_path)

    worker = DeepSeekPersistentWorker("/model")

    # Prevent spawning
    monkeypatch.setattr(worker, "_start_worker", lambda: None)
    worker.input_queue = _QueueStub()
    worker.output_queue = _OutputStub(("SUCCESS", {"raw_text": "ok", "elapsed": 0.1, "peak_rss": 12.0}))

    raw, elapsed, rss = worker.infer(
        image_path=str(img_path),
        prompt_template="Prompt",
        base_size=0,
        image_size=0,
        use_grounding=True,
        timeout_sec=1,
    )

    assert raw == "ok"
    assert elapsed == 0.1
    assert rss == 12.0
    assert worker.is_warm is True

    # Ensure command queued and grounding prefix applied
    assert worker.input_queue.items
    cmd, payload = worker.input_queue.items[0]
    assert cmd == "INFER"
    assert payload["prompt_template"].startswith("<|grounding|>")


def test_infer_error_status_raises(monkeypatch, tmp_path: Path):
    from extraction.deepseek_persistent_worker import DeepSeekPersistentWorker

    img_path = tmp_path / "img.png"
    Image.new("RGB", (10, 10), color=(255, 255, 255)).save(img_path)

    worker = DeepSeekPersistentWorker("/model")
    monkeypatch.setattr(worker, "_start_worker", lambda: None)
    worker.input_queue = _QueueStub()
    worker.output_queue = _OutputStub(("ERROR", "bad"))

    with pytest.raises(RuntimeError):
        worker.infer(
            image_path=str(img_path),
            prompt_template="P",
            base_size=0,
            image_size=0,
            use_grounding=False,
            timeout_sec=1,
        )


def test_infer_timeout_triggers_restart(monkeypatch, tmp_path: Path):
    from extraction.deepseek_persistent_worker import DeepSeekPersistentWorker

    img_path = tmp_path / "img.png"
    Image.new("RGB", (10, 10), color=(255, 255, 255)).save(img_path)

    worker = DeepSeekPersistentWorker("/model")
    monkeypatch.setattr(worker, "_start_worker", lambda: None)
    worker.input_queue = _QueueStub()
    worker.output_queue = _OutputStub(exc=queue.Empty())

    restarted = {"n": 0}

    def fake_restart():
        restarted["n"] += 1

    monkeypatch.setattr(worker, "kill_and_restart", fake_restart)

    with pytest.raises(TimeoutError):
        worker.infer(
            image_path=str(img_path),
            prompt_template="P",
            base_size=0,
            image_size=0,
            use_grounding=False,
            timeout_sec=0,
        )

    assert restarted["n"] == 1


def test_stop_graceful_and_force_terminate(monkeypatch):
    from extraction.deepseek_persistent_worker import DeepSeekPersistentWorker

    worker = DeepSeekPersistentWorker("/model")
    worker.process = _DummyProcess()
    worker.input_queue = _QueueStub()

    # Simulate join timeout keeps process alive
    def still_alive():
        return True

    monkeypatch.setattr(worker.process, "is_alive", still_alive)

    worker.stop()

    assert worker.process is None
    assert worker.is_warm is False
    assert worker.input_queue.items[0][0] == "STOP"
