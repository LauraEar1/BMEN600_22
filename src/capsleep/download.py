from __future__ import annotations

from pathlib import Path
import re
import shutil
import uuid

import requests


class DownloadError(RuntimeError):
    pass


def fetch_edf_header(url: str, timeout: int) -> tuple[bytes, int]:
    response = requests.get(
        url, headers={"Range": "bytes=0-65535"}, timeout=timeout
    )
    response.raise_for_status()
    if response.status_code != 206:
        raise DownloadError(f"Server did not honor an EDF-header byte range for {url}")
    match = re.fullmatch(
        r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
    )
    if not match or match.group(1) != "0":
        raise DownloadError(f"Invalid EDF Content-Range response for {url}")
    content = response.content
    file_size = int(match.group(3))
    if len(content) != int(match.group(2)) + 1:
        raise DownloadError(f"Truncated ranged EDF-header response for {url}")
    if len(content) < 256:
        raise DownloadError(f"EDF fixed header is truncated for {url}")
    header_size = int(content[184:192].decode("ascii").strip())
    signal_count = int(content[252:256].decode("ascii").strip())
    if not 0 < signal_count <= 4096 or not 256 + signal_count * 256 <= header_size <= 256 + 4096 * 256:
        raise DownloadError(f"Invalid EDF header dimensions for {url}")
    if header_size > len(content):
        extra = requests.get(
            url,
            headers={"Range": f"bytes={len(content)}-{header_size - 1}"},
            timeout=timeout,
        )
        extra.raise_for_status()
        if extra.status_code != 206:
            raise DownloadError(f"Server did not honor the full EDF-header range for {url}")
        extra_range = re.fullmatch(
            r"bytes (\d+)-(\d+)/(\d+)", extra.headers.get("Content-Range", "")
        )
        if (
            not extra_range
            or int(extra_range.group(1)) != len(content)
            or int(extra_range.group(3)) != file_size
            or len(extra.content) != int(extra_range.group(2)) - len(content) + 1
        ):
            raise DownloadError(f"Invalid EDF signal-header range response for {url}")
        content += extra.content
    if len(content) < header_size:
        raise DownloadError(f"EDF signal header is truncated for {url}")
    return content[:header_size], file_size


def remote_size(url: str, timeout: int) -> int:
    response = requests.head(url, allow_redirects=True, timeout=timeout)
    response.raise_for_status()
    value = response.headers.get("Content-Length")
    if value is None:
        raise DownloadError(f"PhysioNet did not report a file size for {url}")
    return int(value)


def download_recording(
    url: str,
    temporary_directory: Path,
    minimum_free_gb: float,
    timeout: int,
) -> Path:
    temporary_directory.mkdir(parents=True, exist_ok=True)
    size = remote_size(url, timeout)
    free = shutil.disk_usage(temporary_directory).free
    required = size + int(minimum_free_gb * 1024**3)
    if free < required:
        raise DownloadError(
            f"Insufficient disk space for {url}: {free / 1024**3:.2f} GiB free, "
            f"{required / 1024**3:.2f} GiB required"
        )
    name = Path(url).name
    destination = temporary_directory / f".capsleep_{name[:-4]}_{uuid.uuid4().hex}.edf"
    partial = destination.with_suffix(".edf.part")
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with partial.open("xb") as stream:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        stream.write(block)
        if partial.stat().st_size != size:
            raise DownloadError(
                f"Short download for {url}: expected {size} bytes, got {partial.stat().st_size}"
            )
        partial.replace(destination)
        return destination
    except Exception:
        partial.unlink(missing_ok=True)
        raise
