from __future__ import annotations

from datetime import UTC, datetime
import hashlib
from pathlib import Path
import re
import subprocess
import tarfile
import uuid


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return cleaned or "job"


def generate_job_id(name: str) -> str:
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{uuid.uuid4().hex[:8]}-{safe_name(name)[:80]}"


def compute_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_file_list(code_dir: Path) -> list[Path]:
    proc = subprocess.run(
        ["git", "ls-files", "-z", "-c", "-o", "--exclude-standard"],
        cwd=code_dir,
        check=False,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Code directory must be a git repository: {code_dir}")
    rels = [Path(p.decode()) for p in proc.stdout.split(b"\0") if p]
    return [p for p in rels if p.parts and p.parts[0] != ".git"]


def create_code_tarball(code_dir: Path, output_path: Path) -> Path:
    code_dir = code_dir.resolve()
    files = git_file_list(code_dir)
    with tarfile.open(output_path, "w:gz") as tar:
        for rel in files:
            src = code_dir / rel
            if src.is_file():
                tar.add(src, arcname=str(rel))
    return output_path
