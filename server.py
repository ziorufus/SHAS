import argparse
import hmac
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
SUPERVISED_HYBRID_ROOT = ROOT / "src" / "supervised_hybrid"
if str(SUPERVISED_HYBRID_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERVISED_HYBRID_ROOT))

from segment import get_default_device, load_segmentation_models, segment_single_wav, yaml_dump

CHECKPOINT_ENV = "SHAS_CHECKPOINT"
BEARER_TOKEN_ENV = "SHAS_BEARER_TOKEN"
HOST_ENV = "UVICORN_HOST"
PORT_ENV = "UVICORN_PORT"
RELOAD_ENV = "SHAS_RELOAD"

load_dotenv(ROOT / ".env")


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

    yield


app = FastAPI(title="SHAS Segmentation API", lifespan=lifespan)
bearer_scheme = HTTPBearer(auto_error=False)


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


@app.post("/segment")
async def segment(
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
    suffix = Path(wav_file.filename or "input.wav").suffix or ".wav"
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / f"input{suffix}"
        input_path.write_bytes(await wav_file.read())

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
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    original_name = Path(wav_file.filename or "input.wav").name
    for item in yaml_content:
        item["wav"] = original_name

    yaml_body = yaml_dump(yaml_content)
    output_name = f"{Path(original_name).stem}.yaml"
    return Response(
        content=yaml_body,
        media_type="application/x-yaml",
        headers={"Content-Disposition": f'attachment; filename="{output_name}"'},
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
    parser.add_argument("--host", default=os.environ.get(HOST_ENV, "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get(PORT_ENV, "8000"))
    )
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    if not args.reload:
        args.reload = os.environ.get(RELOAD_ENV, "").lower() in {"1", "true", "yes", "on"}

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
    uvicorn.run("server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
