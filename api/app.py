"""API REST BraTS — squelette complet des8 endpoints du projet (FastAPI).

Endpoints (alignés sur le PDF d'origine) :
    GET  /api/health                      — statut + backend modèle
    GET  /api/patients                    — liste des cas de data/patients/
    GET  /api/patients/{id}               — résumé (biomarqueurs + risque si calculés)
    GET  /api/patients/{id}/slice/{n}     — PNG axial avec overlay labels
    POST /api/inference/predict           — lance un job (thread) → {job_id}
    GET  /api/jobs/{job_id}               — statut + AgentResult
    GET  /api/jobs/{job_id}/slice/{n}     — PNG du résultat
    GET  /api/jobs/{job_id}/rapport       — rapport markdown de l'agent

Dépendances optionnelles : fastapi, uvicorn (pip install -r requirements.txt).
"""
from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path

import numpy as np

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import Response
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "fastapi/uvicorn non installés — `pip install fastapi uvicorn` pour lancer l'API"
    ) from exc

from agent.orchestrator import Case, run_agent
from tools.nifti_io import load_labels, load_volume, save_labels
from tools.png_util import slice_png

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "patients"
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def _list_patients() -> list[str]:
    if not DATA_DIR.is_dir():
        return []
    return sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())


def _load_case(patient_id: str) -> Case:
    """Charge data/patients/<id>/volume.npy (+ optional gt_labels.npy, spacing.json)."""
    base = DATA_DIR / patient_id
    vol_path = base / "volume.npy"
    if not vol_path.is_file():
        candidates = list(base.glob("*.npy")) if base.is_dir() else []
        vol_path = next((c for c in candidates if "label" not in c.name and "gt" not in c.name), None)
    if vol_path is None or not Path(vol_path).is_file():
        raise FileNotFoundError(f"patient introuvable ou volume manquant : {patient_id}")
    volume, spacing = load_volume(vol_path)
    gt = None
    gt_path = base / "gt_labels.npy"
    if gt_path.is_file():
        gt = load_labels(gt_path)
    return Case(case_id=patient_id, volume=volume, spacing=spacing, gt_labels=gt)


def create_app() -> FastAPI:
    app = FastAPI(title="BraTS Tumor Agent API", version="1.0.0")

    # 1. health ------------------------------------------------------------
    @app.get("/api/health")
    def health() -> dict:
        import os

        return {
            "status": "ok",
            "model": os.getenv("BRATS_MODEL_PATH", "mock"),
            "patients": len(_list_patients()),
            "time": time.time(),
        }

    # 2. liste patients ----------------------------------------------------
    @app.get("/api/patients")
    def list_patients() -> dict:
        return {"patients": _list_patients()}

    # 3. détail patient ----------------------------------------------------
    @app.get("/api/patients/{patient_id}")
    def patient_detail(patient_id: str) -> dict:
        try:
            case = _load_case(patient_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        result = run_agent("analyse standard", case)
        return {
            "id": patient_id,
            "workflow": result.workflow.value,
            "biomarkers": result.biomarkers.model_dump() if result.biomarkers else None,
            "risk": result.risk.model_dump() if result.risk else None,
            "dice": result.dice.model_dump() if result.dice else None,
            "backend": next((s.name for s in result.steps if s.name == "segmentation"), None),
            "warnings": result.warnings,
        }

    # 4. slice PNG ---------------------------------------------------------
    @app.get("/api/patients/{patient_id}/slice/{slice_index}")
    def patient_slice(patient_id: str, slice_index: int) -> Response:
        try:
            case = _load_case(patient_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        from tools.segmentation import segment_volume

        labels, _ = segment_volume(case.volume, spacing=case.spacing)
        vol3d = case.volume[0] if case.volume.ndim == 4 else case.volume
        n = vol3d.shape[0]
        if not (0 <= slice_index < n):
            raise HTTPException(404, f"slice {slice_index} hors bornes [0,{n})")
        png = slice_png(np.asarray(vol3d[slice_index]), labels[slice_index])
        return Response(content=png, media_type="image/png")

    # 5. lancer inférence --------------------------------------------------
    @app.post("/api/inference/predict")
    def predict(payload: dict) -> dict:
        patient_id = payload.get("patient_id")
        if not patient_id:
            raise HTTPException(400, "patient_id requis")
        try:
            case = _load_case(patient_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

        job_id = uuid.uuid4().hex[:12]
        with JOBS_LOCK:
            JOBS[job_id] = {"status": "pending", "patient_id": patient_id, "started": time.time()}

        def _run() -> None:
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "running"
            try:
                result = run_agent(payload.get("text", "analyse complète"), case)
                with JOBS_LOCK:
                    JOBS[job_id].update(
                        status="done",
                        result=result.model_dump(),
                        finished=time.time(),
                    )
            except Exception as exc:  # jamais de job bloqué en l'état
                with JOBS_LOCK:
                    JOBS[job_id].update(status="error", error=str(exc), finished=time.time())

        threading.Thread(target=_run, daemon=True).start()
        return {"job_id": job_id, "status": "pending"}

    # 6. statut job --------------------------------------------------------
    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "job inconnu")
        return job

    # 7. slice du résultat -------------------------------------------------
    @app.get("/api/jobs/{job_id}/slice/{slice_index}")
    def job_slice(job_id: str, slice_index: int) -> Response:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "job inconnu")
        if job.get("status") != "done":
            raise HTTPException(409, f"job non terminé ({job.get('status')})")
        case = _load_case(job["patient_id"])
        labels_path = DATA_DIR / job["patient_id"] / "pred_labels.npy"
        labels = None
        if labels_path.is_file():
            labels = load_labels(labels_path)
        else:
            from tools.segmentation import segment_volume

            labels, _ = segment_volume(case.volume, spacing=case.spacing)
        vol3d = case.volume[0] if case.volume.ndim == 4 else case.volume
        n = vol3d.shape[0]
        if not (0 <= slice_index < n):
            raise HTTPException(404, f"slice {slice_index} hors bornes [0,{n})")
        png = slice_png(np.asarray(vol3d[slice_index]), labels[slice_index])
        return Response(content=png, media_type="image/png")

    # 8. rapport -----------------------------------------------------------
    @app.get("/api/jobs/{job_id}/rapport")
    def job_report(job_id: str) -> Response:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "job inconnu")
        if job.get("status") != "done":
            raise HTTPException(409, f"job non terminé ({job.get('status')})")
        report = (job.get("result") or {}).get("report", "")
        return Response(content=report, media_type="text/markdown; charset=utf-8")

    return app


app = create_app()

if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
