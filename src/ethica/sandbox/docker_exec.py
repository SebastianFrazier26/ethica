"""Optional Docker executor for code tasks only (everything else runs in the simulated World).

Files travel as a tar on stdin and are unpacked into a tmpfs inside the container, so no host
directory is ever mounted.
"""

from __future__ import annotations

import contextlib
import io
import shutil
import subprocess
import tarfile
import threading
import time
import uuid
from dataclasses import dataclass
from typing import IO

# Mutable tag by default so a fresh checkout works; pass `image_digest="sha256:..."` to pin.
DEFAULT_IMAGE = "python:3.12-slim"
MAX_OUTPUT_CHARS = 20_000
TIMEOUT_EXIT_CODE = 124
OUTPUT_LIMIT_EXIT_CODE = 153
_DOCKER_CALL_TIMEOUT_S = 30
_POLL_S = 0.05

# Runs inside the container: unpack stdin tar into /work, then exec the entry script.
_BOOTSTRAP = (
    "import os,sys,tarfile;"
    "tarfile.open(fileobj=sys.stdin.buffer,mode='r|').extractall('/work',filter='data');"
    "os.chdir('/work');os.execvp('python',['python',sys.argv[1]])"
)


@dataclass(frozen=True)
class DockerExecutor:
    image: str = DEFAULT_IMAGE
    docker_bin: str = "docker"
    image_digest: str = ""

    @property
    def image_ref(self) -> str:
        return f"{self.image}@{self.image_digest}" if self.image_digest else self.image

    def is_available(self) -> bool:
        if shutil.which(self.docker_bin) is None:
            return False
        try:
            r = subprocess.run(
                [self.docker_bin, "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                timeout=10,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError):
            return False
        return r.returncode == 0

    def image_present(self) -> bool:
        try:
            r = subprocess.run(
                [self.docker_bin, "image", "inspect", self.image_ref],
                capture_output=True,
                check=False,
                timeout=_DOCKER_CALL_TIMEOUT_S,
            )
        except (subprocess.TimeoutExpired, OSError):
            return False
        return r.returncode == 0

    def _argv(self, name: str, entry: str) -> list[str]:
        return [
            self.docker_bin, "run", "--rm", "-i", "--name", name,
            "--pull", "never",
            "--network", "none",
            "--ipc", "none",
            "--read-only",
            "--tmpfs", "/work:size=64m,mode=1777,exec",
            "--memory", "512m",
            "--memory-swap", "512m",
            "--cpus", "1",
            "--pids-limit", "128",
            "--ulimit", "nofile=256",
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",
            "--user", "65534",
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            self.image_ref, "python", "-c", _BOOTSTRAP, entry,
        ]  # fmt: skip

    def _docker(self, *args: str) -> None:
        with contextlib.suppress(subprocess.TimeoutExpired, OSError):
            subprocess.run(
                [self.docker_bin, *args],
                capture_output=True,
                check=False,
                timeout=_DOCKER_CALL_TIMEOUT_S,
            )

    def run(
        self, code: str, files: dict[str, str] | None = None, timeout_s: float = 30.0
    ) -> tuple[int, str, str]:
        """Run `code` as main.py beside `files`. Returns (exit_code, stdout, stderr).

        Output is read incrementally; a stream passing MAX_OUTPUT_CHARS kills the container
        (exit code OUTPUT_LIMIT_EXIT_CODE) so a print loop cannot fill host memory."""
        payload = _make_tar({**(files or {}), "main.py": code})
        name = f"ethica-{uuid.uuid4().hex[:12]}"
        proc = subprocess.Popen(
            self._argv(name, "main.py"),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        overflow = threading.Event()
        bufs: dict[str, bytearray] = {"out": bytearray(), "err": bytearray()}
        threads = [
            threading.Thread(target=_drain, args=(stream, bufs[k], overflow), daemon=True)
            for k, stream in (("out", proc.stdout), ("err", proc.stderr))
        ]
        threads.append(threading.Thread(target=_feed, args=(proc.stdin, payload), daemon=True))
        for t in threads:
            t.start()

        deadline = time.monotonic() + timeout_s
        note = ""
        code_override: int | None = None
        while proc.poll() is None:
            if overflow.is_set():
                note, code_override = (
                    "output limit exceeded; container killed",
                    OUTPUT_LIMIT_EXIT_CODE,
                )
                break
            if time.monotonic() >= deadline:
                note, code_override = (
                    f"timeout after {timeout_s}s; container killed",
                    TIMEOUT_EXIT_CODE,
                )
                break
            overflow.wait(_POLL_S)
        if code_override is not None:
            # Killing the docker CLI alone leaves the container running; kill it by name.
            self._docker("kill", name)
            proc.kill()
            self._docker("rm", "-f", name)
        proc.wait()
        for t in threads:
            t.join(timeout=5)
        out, err = _clip(bytes(bufs["out"])), _clip(bytes(bufs["err"]))
        if code_override is not None:
            return code_override, out, err + f"\n[{note}]"
        return proc.returncode, out, err


def _drain(stream: IO[bytes] | None, buf: bytearray, overflow: threading.Event) -> None:
    if stream is None:
        return
    while chunk := stream.read(65536):
        if len(buf) <= MAX_OUTPUT_CHARS:
            buf.extend(chunk)
        if len(buf) > MAX_OUTPUT_CHARS:
            overflow.set()


def _feed(stream: IO[bytes] | None, payload: bytes) -> None:
    if stream is None:
        return
    try:
        stream.write(payload)
        stream.close()
    except OSError:
        pass


def _make_tar(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for path, content in files.items():
            parts = path.replace("\\", "/").split("/")
            if not path or path.startswith(("/", "\\")) or ".." in parts or "\x00" in path:
                raise ValueError(f"unsafe file name for sandbox tar: {path!r}")
            data = content.encode()
            info = tarfile.TarInfo(path)
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _clip(b: bytes) -> str:
    s = b.decode(errors="replace")
    return s if len(s) <= MAX_OUTPUT_CHARS else s[:MAX_OUTPUT_CHARS] + "\n[output truncated]"
