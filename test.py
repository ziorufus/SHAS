import argparse
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from uuid import uuid4


def build_url(base_url: str, endpoint: str) -> str:
    return f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"


def make_headers(token: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {token}"}
    if extra:
        headers.update(extra)
    return headers


def encode_multipart_formdata(fields: dict[str, str], file_field: str, file_path: Path):
    boundary = f"----SHASBoundary{uuid4().hex}"
    body = bytearray()
    crlf = b"\r\n"

    for name, value in fields.items():
        body.extend(f"--{boundary}".encode())
        body.extend(crlf)
        body.extend(
            f'Content-Disposition: form-data; name="{name}"'.encode()
        )
        body.extend(crlf)
        body.extend(crlf)
        body.extend(str(value).encode())
        body.extend(crlf)

    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    body.extend(f"--{boundary}".encode())
    body.extend(crlf)
    body.extend(
        (
            f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{file_path.name}"'
        ).encode()
    )
    body.extend(crlf)
    body.extend(f"Content-Type: {mime_type}".encode())
    body.extend(crlf)
    body.extend(crlf)
    body.extend(file_path.read_bytes())
    body.extend(crlf)
    body.extend(f"--{boundary}--".encode())
    body.extend(crlf)

    content_type = f"multipart/form-data; boundary={boundary}"
    return bytes(body), content_type


def request_json(
    url: str,
    method: str,
    token: str,
    data: bytes | None = None,
    content_type: str | None = None,
) -> dict:
    headers = make_headers(token)
    if content_type:
        headers["Content-Type"] = content_type

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        details = exc.read().decode(errors="replace")
        raise RuntimeError(f"{method} {url} failed: HTTP {exc.code} {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc


def download_file(url: str, token: str, output_path: Path) -> None:
    request = urllib.request.Request(url, headers=make_headers(token), method="GET")
    try:
        with urllib.request.urlopen(request) as response:
            output_path.write_bytes(response.read())
    except urllib.error.HTTPError as exc:
        details = exc.read().decode(errors="replace")
        raise RuntimeError(f"GET {url} failed: HTTP {exc.code} {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"GET {url} failed: {exc.reason}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload an audio file to the SHAS API and save the output YAML."
    )
    parser.add_argument("audio_file", help="path to the input WAV/audio file")
    parser.add_argument(
        "-o",
        "--output",
        help="path to save the output YAML (default: <audio-stem>.yaml)",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="server base URL, for example http://127.0.0.1:8000 or https://host/shas",
    )
    parser.add_argument(
        "--bearer-token",
        required=True,
        help="bearer token required by the API",
    )
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="maximum seconds to wait; 0 means no timeout",
    )
    parser.add_argument("--inference-batch-size", type=int, default=12)
    parser.add_argument("--inference-segment-length", type=int, default=20)
    parser.add_argument("--inference-times", type=int, default=1)
    parser.add_argument("--dac-max-segment-length", type=float, default=18.0)
    parser.add_argument("--dac-min-segment-length", type=float, default=0.2)
    parser.add_argument("--dac-threshold", type=float, default=0.5)
    parser.add_argument(
        "--not-strict",
        action="store_true",
        help="set the API not_strict flag",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    audio_path = Path(args.audio_file).expanduser().resolve()
    if not audio_path.is_file():
        print(f"Input audio file not found: {audio_path}", file=sys.stderr)
        return 1

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else audio_path.with_suffix(".yaml")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    form_fields = {
        "inference_batch_size": str(args.inference_batch_size),
        "inference_segment_length": str(args.inference_segment_length),
        "inference_times": str(args.inference_times),
        "dac_max_segment_length": str(args.dac_max_segment_length),
        "dac_min_segment_length": str(args.dac_min_segment_length),
        "dac_threshold": str(args.dac_threshold),
        "not_strict": "true" if args.not_strict else "false",
    }

    body, content_type = encode_multipart_formdata(form_fields, "wav_file", audio_path)
    start_url = build_url(args.base_url, "/segment-start")
    status_url = build_url(args.base_url, "/segment-status")
    output_url = build_url(args.base_url, "/segment-out")

    print(f"Submitting {audio_path} to {start_url}")
    start_payload = request_json(
        start_url,
        method="POST",
        token=args.bearer_token,
        data=body,
        content_type=content_type,
    )
    job_id = start_payload["job_id"]
    print(f"Job started: {job_id}")

    started_at = time.monotonic()
    last_progress = None

    while True:
        if args.timeout and (time.monotonic() - started_at) > args.timeout:
            raise TimeoutError(
                f"Timed out after {args.timeout} seconds while waiting for job {job_id}"
            )

        query = urllib.parse.urlencode({"job_id": job_id})
        status_payload = request_json(
            f"{status_url}?{query}",
            method="GET",
            token=args.bearer_token,
        )
        status = status_payload["status"]
        progress = status_payload.get("progress")

        if progress != last_progress:
            if progress is None:
                print(f"Status: {status}")
            else:
                print(f"Status: {status} ({progress:.1f}%)")
            last_progress = progress

        if status == "completed":
            break
        if status == "error":
            raise RuntimeError(
                f"Segmentation job {job_id} failed: {status_payload.get('error', 'unknown error')}"
            )

        time.sleep(max(args.poll_interval, 0.1))

    query = urllib.parse.urlencode({"job_id": job_id})
    download_file(f"{output_url}?{query}", args.bearer_token, output_path)
    print(f"Saved YAML output to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
