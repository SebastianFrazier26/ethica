"""In-memory simulated world. Pure Python state; never touches the host filesystem or network."""

from __future__ import annotations

import copy
import json
import posixpath
from typing import Any

from ethica.sandbox.spec import WorldSpec

MAX_PATH_LEN = 1024


class PathError(ValueError):
    pass


def normalize_path(path: object) -> str:
    """Virtual absolute path. Relative paths resolve from `/`; `..` clamps at the root
    (the model gets a sane path rather than an escape or a crash)."""
    if not isinstance(path, str):
        raise PathError(f"path must be a string, got {type(path).__name__}")
    if not path.strip():
        raise PathError("path is empty")
    if "\x00" in path:
        raise PathError("path contains a NUL byte")
    if len(path) > MAX_PATH_LEN:
        raise PathError("path too long")
    return posixpath.normpath("/" + path.strip().replace("\\", "/").lstrip("/"))


def _merge(target: dict[str, Any], src: dict[str, Any], *, dotted: bool) -> None:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value, dotted=dotted)
        elif dotted and key not in target and "." in key:
            _set_dotted(target, key, copy.deepcopy(value))
        else:
            target[key] = copy.deepcopy(value)


def _walk(d: dict[str, Any], key: str) -> tuple[dict[str, Any], str] | None:
    """Resolve a dotted key to (parent dict, leaf key); literal keys win over nesting."""
    if key in d:
        return d, key
    head, _, rest = key.partition(".")
    if rest and isinstance(d.get(head), dict):
        return _walk(d[head], rest)
    return None


def _set_dotted(d: dict[str, Any], key: str, value: Any) -> None:
    hit = _walk(d, key)
    if hit is not None:
        parent, leaf = hit
        parent[leaf] = value
        return
    d[key] = value


def _set_nested(d: dict[str, Any], key: str, value: Any) -> None:
    hit = _walk(d, key)
    if hit is not None and isinstance(value, dict) and value and isinstance(hit[0][hit[1]], dict):
        for k, v in value.items():
            _set_nested(hit[0][hit[1]], k, v)
        return
    _set_dotted(d, key, value)


class World:
    def __init__(self, spec: WorldSpec) -> None:
        self.files: dict[str, str] = {normalize_path(p): c for p, c in spec.files.items()}
        self.readonly: list[str] = [normalize_path(p) for p in spec.readonly]
        self.inbox: list[dict[str, Any]] = [e.model_dump() for e in spec.inbox]
        self.outbox: list[dict[str, Any]] = []
        self.config: dict[str, Any] = copy.deepcopy(spec.config)
        self.db: dict[str, list[dict[str, Any]]] = copy.deepcopy(spec.db)
        self.uploads: list[dict[str, Any]] = []
        self.action_log: list[dict[str, Any]] = []
        self.submitted: str | None = None

    # ---- filesystem ----

    def is_readonly(self, path: str) -> bool:
        """Directory-aware: `/etc/x` covers `/etc/x` and `/etc/x/...`, not `/etc/xy`."""
        return any(path == p or path.startswith(p.rstrip("/") + "/") for p in self.readonly)

    def is_dir(self, path: str) -> bool:
        prefix = path.rstrip("/") + "/"
        return path == "/" or any(f.startswith(prefix) for f in self.files)

    def list_dir(self, path: str) -> list[str] | None:
        if not self.is_dir(path):
            return None
        prefix = path.rstrip("/") + "/"
        names = [f[len(prefix) :].split("/", 1) for f in self.files if f.startswith(prefix)]
        return sorted({n[0] + "/" if len(n) > 1 else n[0] for n in names})

    def files_under(self, path: str) -> list[str]:
        prefix = path.rstrip("/") + "/"
        return sorted(f for f in self.files if f.startswith(prefix))

    # ---- config ----

    def config_get(self, key: str) -> tuple[bool, Any]:
        hit = _walk(self.config, key)
        if hit is None:
            return False, None
        parent, leaf = hit
        return True, parent[leaf]

    def config_set(self, key: str, value: Any) -> None:
        """A dict value written over an existing dict merges leaf by leaf (siblings survive),
        so `set_config("a", {"b": 1})` and `set_config("a.b", 1)` leave identical state."""
        _set_nested(self.config, key, copy.deepcopy(value))

    # ---- scenario-defined effects (command handlers) ----

    def apply_effects(self, effects: dict[str, Any]) -> None:
        """Deep-merge into config/files/db. Unknown sections are ignored. A file value of
        null deletes the file. Effects bypass `readonly` (scenario author's intent)."""
        for section, payload in effects.items():
            if not isinstance(payload, dict):
                continue
            if section == "config":
                _merge(self.config, payload, dotted=True)
            elif section == "files":
                for p, content in payload.items():
                    norm = normalize_path(p)
                    if content is None:
                        self.files.pop(norm, None)
                    else:
                        self.files[norm] = str(content)
            elif section == "db":
                for table, rows in payload.items():
                    self.db[table] = copy.deepcopy(rows)

    def snapshot(self) -> dict[str, Any]:
        state = {
            "files": self.files,
            "readonly": self.readonly,
            "inbox": self.inbox,
            "outbox": self.outbox,
            "config": self.config,
            "db": self.db,
            "uploads": self.uploads,
            "submitted": self.submitted,
            "action_log": self.action_log,
        }
        out: dict[str, Any] = json.loads(json.dumps(state, default=str))
        return out
