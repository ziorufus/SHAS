import argparse
import hmac
import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

ROOT = Path(__file__).resolve().parent
SUPERVISED_HYBRID_ROOT = ROOT / "src" / "supervised_hybrid"
if str(SUPERVISED_HYBRID_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERVISED_HYBRID_ROOT))

from segment import get_default_device, load_segmentation_models, segment_single_wav, yaml_dump

CHECKPOINT_ENV = "SHAS_CHECKPOINT"
BEARER_TOKEN_ENV = "SHAS_BEARER_TOKEN"
JOBS_DIR_ENV = "JOBS_DIR"
HOST_ENV = "UVICORN_HOST"
PORT_ENV = "UVICORN_PORT"
ROOT_PATH_ENV = "UVICORN_ROOT_PATH"
RELOAD_ENV = "SHAS_RELOAD"

STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_ERROR = "error"

load_dotenv(ROOT / ".env")
bearer_scheme = HTTPBearer(auto_error=False)


def resolve_jobs_dir(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def get_job_dir(job_id: str) -> Path:
    return app.state.jobs_dir / job_id


def get_status_path(job_dir: Path) -> Path:
    return job_dir / "status.json"


def get_output_path(job_dir: Path) -> Path:
    return job_dir / "output.yaml"


def get_error_path(job_dir: Path) -> Path:
    return job_dir / "error.txt"


def infer_job_status_payload(job_dir: Path) -> dict:
    output_path = get_output_path(job_dir)
    if output_path.exists():
        return {"status": STATUS_COMPLETED, "progress": 100.0}

    error_path = get_error_path(job_dir)
    if error_path.exists():
        error_message = error_path.read_text().strip() or "Segmentation job failed"
        return {"status": STATUS_ERROR, "error": error_message}

    request_path = job_dir / "request.json"
    if request_path.exists():
        return {"status": STATUS_RUNNING}

    raise HTTPException(status_code=404, detail="Unknown job ID")


def set_job_status(
    job_dir: Path,
    status: str,
    progress: float | None = None,
    error: str | None = None,
) -> None:
    payload = {"status": status}
    if progress is not None:
        payload["progress"] = round(progress, 1)
    if error:
        payload["error"] = error
    write_json(get_status_path(job_dir), payload)


def get_job_status_payload(job_dir: Path) -> dict:
    status_path = get_status_path(job_dir)
    if not status_path.exists():
        return infer_job_status_payload(job_dir)

    payload = read_json(status_path)
    if payload.get("status") == STATUS_COMPLETED and not get_output_path(job_dir).exists():
        return infer_job_status_payload(job_dir)

    if payload.get("status") == STATUS_ERROR and not get_error_path(job_dir).exists():
        return infer_job_status_payload(job_dir)
    return payload


def require_bearer_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
):
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if credentials.scheme.lower() != "bearer" or not credentials.credentials:
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    expected_token = app.state.bearer_token
    if not hmac.compare_digest(credentials.credentials, expected_token):
        raise HTTPException(
            status_code=401,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def run_segmentation_job(
    job_dir: Path,
    input_path: Path,
    original_name: str,
    inference_batch_size: int,
    inference_segment_length: int,
    inference_times: int,
    dac_max_segment_length: float,
    dac_min_segment_length: float,
    dac_threshold: float,
    not_strict: bool,
) -> None:
    def update_progress(progress: float) -> None:
        set_job_status(job_dir, STATUS_RUNNING, progress=progress)

    try:
        yaml_content = segment_single_wav(
            input_path,
            app.state.wav2vec_model,
            app.state.sfc_model,
            app.state.device,
            inference_batch_size=inference_batch_size,
            inference_segment_length=inference_segment_length,
            inference_times=inference_times,
            dac_max_segment_length=dac_max_segment_length,
            dac_min_segment_length=dac_min_segment_length,
            dac_threshold=dac_threshold,
            not_strict=not_strict,
            dataloader_num_workers=0,
            progress_callback=update_progress,
        )
        for item in yaml_content:
            item["wav"] = original_name

        get_output_path(job_dir).write_text(yaml_dump(yaml_content))
        set_job_status(job_dir, STATUS_COMPLETED, progress=100.0)
    except Exception as exc:
        get_error_path(job_dir).write_text(traceback.format_exc())
        set_job_status(job_dir, STATUS_ERROR, error=str(exc))


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint_path = os.environ.get(CHECKPOINT_ENV)
    if not checkpoint_path:
        raise RuntimeError(
            f"Missing {CHECKPOINT_ENV}. Start the server with --checkpoint."
        )
    bearer_token = os.environ.get(BEARER_TOKEN_ENV)
    if not bearer_token:
        raise RuntimeError(
            f"Missing {BEARER_TOKEN_ENV}. Start the server with --bearer-token."
        )

    jobs_dir = resolve_jobs_dir(os.environ.get(JOBS_DIR_ENV, "./jobs"))
    jobs_dir.mkdir(parents=True, exist_ok=True)

    device = get_default_device()
    wav2vec_model, sfc_model, checkpoint, device = load_segmentation_models(
        checkpoint_path, device
    )
    app.state.wav2vec_model = wav2vec_model
    app.state.sfc_model = sfc_model
    app.state.checkpoint = checkpoint
    app.state.device = device
    app.state.checkpoint_path = checkpoint_path
    app.state.bearer_token = bearer_token
    app.state.jobs_dir = jobs_dir
    app.state.executor = ThreadPoolExecutor(max_workers=1)

    yield

    app.state.executor.shutdown(wait=False, cancel_futures=False)


app = FastAPI(title="SHAS Segmentation API", lifespan=lifespan)


@app.post("/segment-start")
async def segment_start(
    _authorized: None = Depends(require_bearer_token),
    wav_file: UploadFile = File(...),
    inference_batch_size: int = Form(12),
    inference_segment_length: int = Form(20),
    inference_times: int = Form(1),
    dac_max_segment_length: float = Form(18),
    dac_min_segment_length: float = Form(0.2),
    dac_threshold: float = Form(0.5),
    not_strict: bool = Form(False),
):
    job_id = str(uuid4())
    job_dir = get_job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=False)

    original_name = Path(wav_file.filename or "input.wav").name
    suffix = Path(original_name).suffix or ".wav"
    input_path = job_dir / f"input{suffix}"
    input_path.write_bytes(await wav_file.read())

    write_json(
        job_dir / "request.json",
        {
            "job_id": job_id,
            "original_name": original_name,
            "inference_batch_size": inference_batch_size,
            "inference_segment_length": inference_segment_length,
            "inference_times": inference_times,
            "dac_max_segment_length": dac_max_segment_length,
            "dac_min_segment_length": dac_min_segment_length,
            "dac_threshold": dac_threshold,
            "not_strict": not_strict,
        },
    )
    set_job_status(job_dir, STATUS_RUNNING, progress=0.0)

    app.state.executor.submit(
        run_segmentation_job,
        job_dir,
        input_path,
        original_name,
        inference_batch_size,
        inference_segment_length,
        inference_times,
        dac_max_segment_length,
        dac_min_segment_length,
        dac_threshold,
        not_strict,
    )

    return {"job_id": job_id, "status": STATUS_RUNNING}


@app.get("/segment-status")
def segment_status(
    job_id: str = Query(..., description="Segmentation job ID"),
    _authorized: None = Depends(require_bearer_token),
):
    job_dir = get_job_dir(job_id)
    payload = get_job_status_payload(job_dir)
    return {"job_id": job_id, **payload}


@app.get("/segment-out")
def segment_out(
    job_id: str = Query(..., description="Segmentation job ID"),
    _authorized: None = Depends(require_bearer_token),
):
    job_dir = get_job_dir(job_id)
    payload = get_job_status_payload(job_dir)
    if payload["status"] == STATUS_RUNNING:
        raise HTTPException(status_code=409, detail="Segmentation still running")
    if payload["status"] == STATUS_ERROR:
        raise HTTPException(
            status_code=409,
            detail=payload.get("error", "Segmentation job failed"),
        )

    output_path = get_output_path(job_dir)
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output YAML not found")

    return FileResponse(
        output_path,
        media_type="application/x-yaml",
        filename=f"{job_id}.yaml",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get(CHECKPOINT_ENV),
        help="absolute path to the audio-frame-classifier checkpoint",
    )
    parser.add_argument(
        "--bearer-token",
        default=os.environ.get(BEARER_TOKEN_ENV),
        help="bearer token required to call the API",
    )
    parser.add_argument(
        "--jobs-dir",
        default=os.environ.get(JOBS_DIR_ENV, "./jobs"),
        help="directory used to store per-job inputs, status, and outputs",
    )
    parser.add_argument("--host", default=os.environ.get(HOST_ENV, "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get(PORT_ENV, "8000"))
    )
    parser.add_argument("--root-path", default=os.environ.get(ROOT_PATH_ENV, ""))
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    if not args.reload:
        args.reload = os.environ.get(RELOAD_ENV, "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    if not args.checkpoint:
        parser.error(
            f"--checkpoint is required unless {CHECKPOINT_ENV} is set in the environment or .env"
        )
    if not args.bearer_token:
        parser.error(
            f"--bearer-token is required unless {BEARER_TOKEN_ENV} is set in the environment or .env"
        )

    os.environ[CHECKPOINT_ENV] = args.checkpoint
    os.environ[BEARER_TOKEN_ENV] = args.bearer_token
    os.environ[JOBS_DIR_ENV] = args.jobs_dir
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        root_path=args.root_path,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
