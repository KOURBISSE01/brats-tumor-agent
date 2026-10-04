#!/usr/bin/env python3
"""Serveur MCP (Model Context Protocol) — stdio, JSON-RPC2.0, zéro dépendance.

Protocole : `2024-11-05` · transport newline-delimited JSON sur stdin/stdout.
Méthodes implémentées : initialize, notifications/initialized, ping,
tools/list, tools/call. Les journaux partent sur stderr (stdout est réservé
au protocole).

Outils exposés (fichiers .npy / .nii.gz — étatless, passation par chemin) :
    segment, extract_biomarkers, classify_risk, evaluate_dice,
    query_knowledge, render_slice

Connexion (Claude Code / client MCP) :
    { "mcpServers": { "brats": {
        "command": "python3", "args": ["/chemin/vers/mcp_server.py"] } } }
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_VERSION = "2024-11-05"

TOOLS = [
    {
        "name": "segment",
        "description": (
            "Segmente un volume IRM BraTS (.npy 3D/4D ou .nii.gz) en labels {0,1,2,3} "
            "avec hiérarchie WT ⊇ TC ⊇ ET. Écrit labels.npy et retourne le résumé."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "volume_path": {"type": "string", "description": "Chemin du volume d'entrée"},
                "output_path": {"type": "string", "description": "Chemin .npy de sortie (défaut: <volume>_labels.npy)"},
                "spacing": {"type": "array", "items": {"type": "number"}, "description": "Spacing (z,y,x) mm, défaut (1,1,1)"},
                "model_path": {"type": "string", "description": "Poids TorchScript optionnels (sinon mode MOCK)"},
            },
            "required": ["volume_path"],
        },
    },
    {
        "name": "extract_biomarkers",
        "description": "Calcule les 11 biomarqueurs BraTS (volumes, ratios, sphéricité, composantes) depuis une carte de labels.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "labels_path": {"type": "string", "description": "Carte de labels .npy"},
                "spacing": {"type": "array", "items": {"type": "number"}, "description": "Spacing (z,y,x) mm"},
            },
            "required": ["labels_path"],
        },
    },
    {
        "name": "classify_risk",
        "description": "Classe le risque LOW/MED/HIGH (règles Q33/Q66) depuis une carte de labels.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "labels_path": {"type": "string"},
                "spacing": {"type": "array", "items": {"type": "number"}},
            },
            "required": ["labels_path"],
        },
    },
    {
        "name": "evaluate_dice",
        "description": "Dice WT/TC/ET + moyenne entre labels prédits et ground truth.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pred_path": {"type": "string", "description": "Labels prédits .npy"},
                "gt_path": {"type": "string", "description": "Ground truth .npy"},
            },
            "required": ["pred_path", "gt_path"],
        },
    },
    {
        "name": "query_knowledge",
        "description": "Interroge la base de connaissances BraTS (BM25 sur rag/docs/*.md) — protocoles, définitions, règles de risque.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "description": "Nombre de sections (défaut 3)", "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "render_slice",
        "description": "Rend un slice IRM en PNG avec overlay des labels tumorales (sortie RGB).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "volume_path": {"type": "string", "description": "Volume .npy (3D ou 4D — canal0 si 4D)"},
                "labels_path": {"type": "string", "description": "Labels .npy optionnels pour l'overlay"},
                "slice_index": {"type": "integer", "description": "Index du slice axial (défaut: centre)"},
                "output_path": {"type": "string", "description": "Chemin .png de sortie"},
            },
            "required": ["volume_path", "output_path"],
        },
    },
]


def _spacing(args: dict) -> tuple[float, float, float]:
    s = args.get("spacing", [1.0, 1.0, 1.0])
    return (float(s[0]), float(s[1]), float(s[2]))


def call_tool(name: str, args: dict) -> dict:
    """Dispatch d'un outil → dict JSON-serializable."""
    from tools.nifti_io import load_labels, load_volume, save_labels

    if name == "segment":
        from tools.segmentation import segment_volume

        volume, sp = load_volume(args["volume_path"])
        if "spacing" in args:
            sp = _spacing(args)
        labels, meta = segment_volume(volume, spacing=sp, model_path=args.get("model_path"))
        out = args.get("output_path") or str(Path(args["volume_path"]).with_suffix("")) + "_labels.npy"
        save_labels(labels, out)
        return {
            "labels_path": out,
            "backend": meta.get("backend"),
            "warnings": meta.get("warnings", []),
            "voxels_wt": int((labels > 0).sum()),
            "shape": list(labels.shape),
        }

    if name == "extract_biomarkers":
        from tools.biomarkers import compute_biomarkers

        labels = load_labels(args["labels_path"])
        b = compute_biomarkers(labels, spacing=_spacing(args))
        return b.model_dump()

    if name == "classify_risk":
        from tools.biomarkers import compute_biomarkers
        from tools.risk import classify_risk

        labels = load_labels(args["labels_path"])
        b = compute_biomarkers(labels, spacing=_spacing(args))
        r = classify_risk(b)
        return r.model_dump()

    if name == "evaluate_dice":
        from tools.evaluation import evaluate_dice

        pred = load_labels(args["pred_path"])
        gt = load_labels(args["gt_path"])
        return evaluate_dice(pred, gt).model_dump()

    if name == "query_knowledge":
        from rag.knowledge_base import query as rag_query

        hits = rag_query(args["query"], k=int(args.get("k", 3)))
        return {"hits": hits}

    if name == "render_slice":
        import numpy as np

        from tools.png_util import slice_png

        volume, _ = load_volume(args["volume_path"], channel_first=True)
        vol3d = volume[0] if volume.ndim == 4 else volume
        n = vol3d.shape[0]
        idx = int(args.get("slice_index", n // 2))
        idx = max(0, min(idx, n - 1))
        labels = None
        if args.get("labels_path"):
            labels = load_labels(args["labels_path"])
        png = slice_png(np.asarray(vol3d[idx]), None if labels is None else labels[idx])
        out = args["output_path"]
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(png)
        return {"output_path": out, "slice_index": idx, "bytes": len(png)}

    raise ValueError(f"outil inconnu : {name}")


# ---------------------------------------------------------------------------
# Boucle JSON-RPCstdio
# ---------------------------------------------------------------------------
def _result(msg_id, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(msg_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def handle(msg: dict) -> dict | None:
    """Traite un message JSON-RPC. None = notification (pas de réponse)."""
    method = msg.get("method")
    msg_id = msg.get("id")
    params = msg.get("params") or {}

    if method is None:
        return _error(msg_id, -32600, "Requête invalide : champ method manquant")

    if method.startswith("notifications/"):
        return None  # notifications : accusé de réception silencieux

    if method == "initialize":
        return _result(msg_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "brats-tumor-agent", "version": "1.0.0"},
        })

    if method == "ping":
        return _result(msg_id, {})

    if method == "tools/list":
        return _result(msg_id, {"tools": TOOLS})

    if method == "tools/call":
        name = (params or {}).get("name")
        args = (params or {}).get("arguments") or {}
        try:
            result = call_tool(name, args)
            return _result(msg_id, {
                "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, default=str)}],
                "isError": False,
            })
        except Exception as exc:
            tb = traceback.format_exc(limit=3)
            print(f"[mcp] erreur outil {name}: {tb}", file=sys.stderr)
            return _result(msg_id, {
                "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                "isError": True,
            })

    return _error(msg_id, -32601, f"Méthode non supportée : {method}")


def main() -> None:
    print("[mcp] serveur brats-tumor-agent prêt (stdio)", file=sys.stderr)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            resp = _error(None, -32700, f"JSON invalide : {exc}")
            print(json.dumps(resp, ensure_ascii=False), flush=True)
            continue
        try:
            resp = handle(msg)
        except Exception as exc:  # garde-fou : le serveur ne doit pas mourir
            print(f"[mcp] interne: {traceback.format_exc(limit=3)}", file=sys.stderr)
            resp = _error(msg.get("id"), -32603, f"Erreur interne : {exc}")
        if resp is not None:
            print(json.dumps(resp, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    if not os.getenv("BRATS_NO_ANSI"):
        pass  # pas de couleur : stdout protocole
    main()
