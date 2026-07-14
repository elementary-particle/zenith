"""Atomic, checksummed trainer checkpoints with transactional restore validation."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile


class CheckpointError(RuntimeError): pass


def _hash(path):
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1 << 20): digest.update(chunk)
    return digest.hexdigest()


def _sync_file(path):
    with path.open("rb") as file: os.fsync(file.fileno())


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def publish(root, state, *, compatibility, metadata=None):
    import torch
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".checkpoint-staging-", dir=root))
    try:
        torch.save(state["model"], staging / "model.pt")
        torch.save(state.get("optimizer", {}), staging / "optimizer.pt")
        torch.save(state.get("trainer", {}), staging / "trainer-state.pt")
        json_state = state.get("state", {})
        (staging / "state.json").write_text(json.dumps(json_state, sort_keys=True, default=_json), encoding="utf-8")
        members = ["model.pt", "optimizer.pt", "trainer-state.pt", "state.json"]
        for member in members: _sync_file(staging / member)
        hashes = {member: _hash(staging / member) for member in members}
        (staging / "CHECKSUMS").write_text("".join(f"{digest}  {name}\n" for name, digest in hashes.items()), encoding="utf-8")
        _sync_file(staging / "CHECKSUMS")
        manifest = {"schema": 1, "compatibility": asdict(compatibility), "members": hashes,
                    "metadata": metadata or {}}
        immutable = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        checkpoint_id = sha256(immutable).hexdigest()
        manifest["checkpoint_id"] = checkpoint_id
        (staging / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        _sync_file(staging / "manifest.json"); _sync_directory(staging)
        validate(staging, expected=compatibility)
        destination = root / checkpoint_id
        if destination.exists(): shutil.rmtree(staging)
        else: os.replace(staging, destination); _sync_directory(root)
        temporary = root / ".latest.tmp"
        temporary.write_text(checkpoint_id + "\n", encoding="ascii"); _sync_file(temporary)
        os.replace(temporary, root / "latest"); _sync_directory(root)
        return checkpoint_id
    except Exception:
        if staging.exists(): shutil.rmtree(staging, ignore_errors=True)
        raise


def validate(path, *, expected=None):
    from .compatibility import CompatibilitySet
    path = Path(path)
    try: manifest = json.loads((path / "manifest.json").read_text())
    except Exception as exc: raise CheckpointError(f"invalid checkpoint manifest: {exc}") from exc
    members = manifest.get("members", {})
    required = {"model.pt", "optimizer.pt", "trainer-state.pt", "state.json"}
    if set(members) != required:
        raise CheckpointError(
            f"checkpoint member set mismatch: {sorted(members)} != {sorted(required)}"
        )
    claimed_id = manifest.get("checkpoint_id")
    immutable = dict(manifest)
    immutable.pop("checkpoint_id", None)
    actual_id = sha256(
        json.dumps(immutable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if claimed_id != actual_id:
        raise CheckpointError("checkpoint manifest identity mismatch")
    for member, digest in members.items():
        target = path / member
        if not target.is_file() or _hash(target) != digest:
            raise CheckpointError(f"checkpoint member checksum mismatch: {member}")
    actual = CompatibilitySet.from_mapping(manifest["compatibility"])
    if expected is not None: expected.require(actual, context="checkpoint")
    return manifest


def restore(path, *, expected):
    import torch
    path = Path(path); manifest = validate(path, expected=expected)
    temporary = {
        "model": torch.load(path / "model.pt", map_location="cpu", weights_only=False),
        "optimizer": torch.load(path / "optimizer.pt", map_location="cpu", weights_only=False),
        "trainer": torch.load(path / "trainer-state.pt", map_location="cpu", weights_only=False),
        "state": json.loads((path / "state.json").read_text()), "manifest": manifest,
    }
    return temporary


def reproducibility_metadata(*, input_config, resolved_config, source_revision, dirty,
                             dependency_lock_digest, capabilities, level):
    if level not in {"exact_same_target", "numerical_compatible", "weights_only"}:
        raise ValueError("invalid reproducibility level")
    return {"input_config": str(input_config), "resolved_config": resolved_config,
        "source_revision": source_revision, "source_dirty": bool(dirty),
        "dependency_lock_digest": dependency_lock_digest, "capabilities": capabilities,
        "reproducibility_level": level}


def resolve_latest(root):
    root = Path(root); value = (root / "latest").read_text(encoding="ascii").strip()
    path = root / value
    if not path.is_dir(): raise CheckpointError("latest points to a missing checkpoint")
    return path


def publish_evaluation_records(root, outcomes, rating_transactions):
    """Atomically replace the two recomputable evaluation ledgers."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("outcomes.jsonl", outcomes),
        ("rating-transactions.jsonl", rating_transactions),
    ):
        temporary = root / f".{name}.tmp"
        with temporary.open("w", encoding="utf-8") as output:
            for row in rows:
                output.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, root / name)
    _sync_directory(root)


def _json(value):
    if is_dataclass(value): return asdict(value)
    if hasattr(value, "tolist"): return value.tolist()
    raise TypeError(type(value).__name__)
