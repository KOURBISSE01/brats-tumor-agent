"""Serveur MCP : initialize, tools/list, tools/call, notifications — via subprocess."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def mcp():
    proc = subprocess.Popen(
        [sys.executable, "-u", str(ROOT / "mcp_server.py")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(ROOT),
    )

    def rpc(method, params=None, id=1, expect_response=True):
        msg = {"jsonrpc": "2.0", "id": id, "method": method}
        if params is not None:
            msg["params"] = params
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()
        if not expect_response:
            return None
        line = proc.stdout.readline()
        assert line, f"pas de réponse MCP — stderr: {proc.stderr.read()}"
        return json.loads(line)

    yield rpc
    proc.stdin.close()
    proc.wait(timeout=10)


def test_initialize(mcp):
    r = mcp("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "pytest", "version": "0"}})
    assert r["result"]["protocolVersion"] == "2024-11-05"
    assert r["result"]["serverInfo"]["name"] == "brats-tumor-agent"


def test_tools_list(mcp):
    r = mcp("tools/list", id=2)
    names = {t["name"] for t in r["result"]["tools"]}
    assert {"segment", "extract_biomarkers", "classify_risk", "evaluate_dice", "query_knowledge", "render_slice"} <= names
    # chaque outil a un JSON schema valide
    for t in r["result"]["tools"]:
        assert t["inputSchema"]["type"] == "object"
        assert "properties" in t["inputSchema"]


def test_notification_no_response_then_ping(mcp):
    # notification sans id : pas de réponse (spécification MCP)
    mcp("notifications/initialized", id=None, expect_response=False)
    r = mcp("ping", id=99)
    assert "result" in r


def test_segment_and_biomarkers_roundtrip(mcp, tmp_path):
    vol_path = tmp_path / "vol.npy"
    np.save(vol_path, np.random.default_rng(1).random((4, 32, 32, 32)).astype(np.float32))
    out_path = tmp_path / "labels.npy"

    r = mcp("tools/call", {"name": "segment", "arguments": {"volume_path": str(vol_path), "output_path": str(out_path)}}, id=3)
    content = r["result"]
    assert content["isError"] is False
    res = json.loads(content["content"][0]["text"])
    assert res["backend"] == "mock"
    assert Path(res["labels_path"]).is_file()
    assert res["voxels_wt"] > 0

    r = mcp("tools/call", {"name": "extract_biomarkers", "arguments": {"labels_path": str(out_path)}}, id=4)
    b = json.loads(r["result"]["content"][0]["text"])
    assert b["volume_wt_cm3"] > 0
    assert 0 < b["sphericity_wt"] <= 1.1
    assert b["n_components"] >= 1


def test_evaluate_dice_tool(mcp, tmp_path):
    gt = np.zeros((24, 24, 24), dtype=np.uint8)
    gt[6:18, 6:18, 6:18] = 2
    p = gt.copy()
    p[6:12, :, :] = 0  # enlève la moitié
    gp, pp = tmp_path / "gt.npy", tmp_path / "pred.npy"
    np.save(gp, gt)
    np.save(pp, p)
    r = mcp("tools/call", {"name": "evaluate_dice", "arguments": {"pred_path": str(pp), "gt_path": str(gp)}}, id=5)
    d = json.loads(r["result"]["content"][0]["text"])
    assert 0.0 < d["dice_wt"] < 1.0
    assert r["result"]["isError"] is False


def test_query_knowledge_tool(mcp):
    r = mcp("tools/call", {"name": "query_knowledge", "arguments": {"query": "classification risque LOW HIGH confiance règles", "k": 2}}, id=6)
    hits = json.loads(r["result"]["content"][0]["text"])["hits"]
    assert hits and hits[0]["source"] == "regles_risque.md"


def test_unknown_tool_is_error(mcp):
    r = mcp("tools/call", {"name": "s_ascii_art", "arguments": {}}, id=7)
    assert r["result"]["isError"] is True


def test_unknown_method(mcp):
    r = mcp("resources/list", id=8)
    assert r["error"]["code"] == -32601


def test_render_slice_png(mcp, tmp_path):
    vol_path = tmp_path / "v.npy"
    np.save(vol_path, np.random.rand(32, 32, 32).astype(np.float32))
    out = tmp_path / "s.png"
    r = mcp("tools/call", {"name": "render_slice", "arguments": {"volume_path": str(vol_path), "output_path": str(out)}}, id=9)
    assert r["result"]["isError"] is False
    assert out.read_bytes()[:4] == b"\x89PNG"
