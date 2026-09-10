import re
import subprocess
from pathlib import Path


def test_index_html_has_all_required_dom_ids():
    index_html = Path("web/index.html").read_text(encoding="utf-8")
    required_ids = [
        "resAnswer",
        "resTask",
        "resTool",
        "resModel",
        "resMode",
        "resAdaptation",
        "resEvidence",
        "resInput",
        "resLatency",
        "resConfidence",
        "routingReasonBanner",
        "routingReasonText",
        "agentTraceContainer",
        "traceSteps",
        "metaCrs",
        "metaShape",
        "metaBands",
        "metaDriver",
        "labelPrimaryImage",
        "dropTextPrimary",
        "labelSecondaryImage",
        "dropTextSecondary",
        "groundingCanvas",
        "changeOverlay",
        "toggleGrounding",
        "toggleChange",
        "toggleChangeLayer",
        "layerControls",
        "provenanceFeedFull",
    ]
    missing = []
    for element_id in required_ids:
        pattern = rf'id=["\']{element_id}["\']'
        if not re.search(pattern, index_html):
            missing.append(element_id)

    assert not missing, f"Missing required DOM IDs in web/index.html: {missing}"


def test_app_js_defensive_dom_node_runner():
    res = subprocess.run(["node", "tests/test_frontend_dom.js"], capture_output=True, text=True)
    assert res.returncode == 0, f"Node frontend test failed: {res.stdout}\n{res.stderr}"
    assert "ALL FRONTEND DOM & RENDERING TESTS PASSED" in res.stdout


def test_e2e_live_ui_node_runner():
    res = subprocess.run(["node", "tests/test_e2e_live_ui.js"], capture_output=True, text=True)
    assert res.returncode == 0, f"Live E2E frontend test failed: {res.stdout}\n{res.stderr}"
    assert "ALL 4 WORKFLOWS RENDERED PERFECTLY WITH ZERO ERRORS" in res.stdout

