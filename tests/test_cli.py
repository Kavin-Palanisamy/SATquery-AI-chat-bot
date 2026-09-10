import subprocess
import pytest


def test_cli_models():
    res = subprocess.run([".\\.venv\\Scripts\\satquery.exe", "models"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "Registered Remote Sensing Models:" in res.stdout
    assert "Qwen2-VL-RS" in res.stdout
    assert "RS-SpectralHeuristic-Grounder-v1" in res.stdout


def test_cli_health():
    res = subprocess.run([".\\.venv\\Scripts\\satquery.exe", "health"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "HEALTHY" in res.stdout
    assert "v3.0.0" in res.stdout


def test_cli_train_status():
    res = subprocess.run([".\\.venv\\Scripts\\satquery.exe", "train-status"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "BigEarthNet.txt" in res.stdout


def test_cli_validate(sample_png_image):
    res = subprocess.run([".\\.venv\\Scripts\\satquery.exe", "validate", str(sample_png_image)], capture_output=True, text=True)
    assert res.returncode == 0
    assert '"valid": true' in res.stdout


def test_cli_ask(sample_png_image):
    res = subprocess.run([".\\.venv\\Scripts\\satquery.exe", "ask", str(sample_png_image), "Are there water bodies?"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "Task:  vqa" in res.stdout
