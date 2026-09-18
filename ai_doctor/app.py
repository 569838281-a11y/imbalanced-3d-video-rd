"""FastAPI server for the cascade AI doctor."""

from __future__ import annotations

import logging
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ai_doctor.doctor import IMAGE_EXTS, VIDEO_EXTS, get_doctor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ai_doctor")

STATIC_DIR = Path(__file__).resolve().parent / "static"
UPLOAD_ROOT = Path(tempfile.gettempdir()) / "erdes_ai_doctor_uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ERDES AI Doctor", version="0.1.0")
app.mount("/assets", StaticFiles(directory=str(STATIC_DIR)), name="assets")


@app.on_event("startup")
def _warmup() -> None:
    log.info("Warming up AI Doctor models...")
    get_doctor()
    log.info("Warmup done.")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    doctor = get_doctor()
    return {
        "status": "ok",
        "device": doctor.device,
        "rd_threshold": doctor.rd_threshold,
        "macula_threshold": doctor.macula_threshold,
        "stage3_threshold": doctor.stage3_threshold,
        "stage3_enabled": doctor.enable_stage3,
        "stage3_intact_ready": doctor.stage3_intact_ckpt is not None,
        "stage3_detached_ready": doctor.stage3_detached_ckpt is not None,
        "pipeline": ["rd", "macula", "subtype"],
    }


@app.post("/api/diagnose")
async def diagnose(files: List[UploadFile] = File(...)) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="请上传至少一个超声视频或图像文件。")

    session = UPLOAD_ROOT / uuid.uuid4().hex
    session.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    try:
        for uf in files:
            name = Path(uf.filename or "upload.bin").name
            ext = Path(name).suffix.lower()
            if ext not in VIDEO_EXTS and ext not in IMAGE_EXTS:
                raise HTTPException(
                    status_code=400,
                    detail=f"不支持的文件类型: {ext}。请上传 mp4/avi/mov 或 jpg/png 等图像。",
                )
            dest = session / name
            with dest.open("wb") as f:
                shutil.copyfileobj(uf.file, f)
            saved.append(dest)

        doctor = get_doctor()
        videos = [p for p in saved if p.suffix.lower() in VIDEO_EXTS]
        images = [p for p in saved if p.suffix.lower() in IMAGE_EXTS]

        if videos and images:
            raise HTTPException(status_code=400, detail="请不要同时上传视频与图像，择一即可。")
        if len(videos) > 1:
            raise HTTPException(status_code=400, detail="一次请只上传一个超声视频。")
        if videos:
            result = doctor.diagnose_file(videos[0])
        else:
            result = doctor.diagnose_images(images)
        return {"ok": True, "result": result}
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Diagnosis failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        shutil.rmtree(session, ignore_errors=True)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "ai_doctor.app:app",
        host="127.0.0.1",
        port=7860,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
