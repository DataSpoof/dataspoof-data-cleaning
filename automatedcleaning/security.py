"""Model signing: prove a saved model came from your training pipeline and wasn't altered.

A saved model directory is signed with an Ed25519 private key (kept by the training job / CI);
deployments verify it with the public key **before** anything is unpickled::

    from automatedcleaning.security import generate_keypair, sign_model, verify_model
    generate_keypair("keys/model_signing.pem", "keys/model_signing.pub")
    sign_model("models/churn", "keys/model_signing.pem")          # writes models/churn/signature.json
    verify_model("models/churn", "keys/model_signing.pub")        # raises SignatureError if tampered

Set ``AC_MODEL_PUBLIC_KEY`` (a path or PEM text) and every ``load_model`` — the REST API, batch
scoring, the registry — refuses models that aren't signed by the matching key.
Needs the ``cryptography`` package.
"""

from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import json
import os
from pathlib import Path

SIGNATURE_FILE = "signature.json"


class SignatureError(Exception):
    """The model is unsigned, signed by another key, or was modified after signing."""


def _crypto():
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
    except ImportError as exc:
        raise ImportError("Model signing needs the cryptography package: pip install cryptography") from exc
    return serialization, Ed25519PrivateKey, Ed25519PublicKey


def _read_key_material(key) -> bytes:
    if isinstance(key, bytes):
        return key
    text = str(key)
    if text.lstrip().startswith("-----BEGIN"):
        return text.encode()
    return Path(text).read_bytes()


def generate_keypair(private_path, public_path, *, password: str | None = None) -> tuple:
    """Write a new Ed25519 key pair as PEM files (the private key optionally encrypted)."""
    serialization, Ed25519PrivateKey, _ = _crypto()
    key = Ed25519PrivateKey.generate()
    enc = (serialization.BestAvailableEncryption(password.encode()) if password
           else serialization.NoEncryption())
    private_path, public_path = Path(private_path), Path(public_path)
    private_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, enc))
    try:
        os.chmod(private_path, 0o600)
    except OSError:
        pass
    public_path.write_bytes(key.public_key().public_bytes(serialization.Encoding.PEM,
                                                          serialization.PublicFormat.SubjectPublicKeyInfo))
    return private_path, public_path


def _digest_files(path: Path) -> dict:
    files = {}
    for f in sorted(p for p in path.rglob("*") if p.is_file() and p.name != SIGNATURE_FILE):
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        files[f.relative_to(path).as_posix()] = h.hexdigest()
    return files


def _payload(files: dict, meta: dict) -> bytes:
    return json.dumps({"files": files, **meta}, sort_keys=True, separators=(",", ":")).encode()


def _fingerprint(public_bytes: bytes) -> str:
    return hashlib.sha256(public_bytes).hexdigest()[:16]


def sign_model(path, private_key, *, password: str | None = None, signer: str | None = None) -> Path:
    """Sign every file in a saved model directory; writes ``signature.json`` next to them."""
    serialization, _, _ = _crypto()
    path = Path(path)
    key = serialization.load_pem_private_key(_read_key_material(private_key),
                                             password=password.encode() if password else None)
    public = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    meta = {"algorithm": "ed25519", "key_id": _fingerprint(public), "signer": signer,
            "signed_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    files = _digest_files(path)
    if not files:
        raise ValueError(f"{path} has no files to sign")
    signature = key.sign(_payload(files, meta))
    out = path / SIGNATURE_FILE
    out.write_text(json.dumps({**meta, "files": files, "signature": base64.b64encode(signature).decode()},
                              indent=2), encoding="utf-8")
    return out


def verify_model(path, public_key) -> dict:
    """Check ``signature.json`` against ``public_key`` and every file's hash. Returns the
    signature metadata; raises :class:`SignatureError` on any problem (missing, wrong key,
    modified, added or removed files)."""
    serialization, _, Ed25519PublicKey = _crypto()
    from cryptography.exceptions import InvalidSignature
    path = Path(path)
    sig_file = path / SIGNATURE_FILE
    if not sig_file.exists():
        raise SignatureError(f"{path} is not signed (no {SIGNATURE_FILE})")
    data = json.loads(sig_file.read_text(encoding="utf-8"))
    pub = serialization.load_pem_public_key(_read_key_material(public_key))
    if not isinstance(pub, Ed25519PublicKey):
        raise SignatureError("The public key is not an Ed25519 key")
    raw = pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    if data.get("key_id") != _fingerprint(raw):
        raise SignatureError(f"{path} was signed by a different key ({data.get('key_id')})")
    meta = {k: data[k] for k in ("algorithm", "key_id", "signer", "signed_at")}
    try:
        pub.verify(base64.b64decode(data["signature"]), _payload(data["files"], meta))
    except InvalidSignature:
        raise SignatureError(f"{path}: the signature does not match its file list") from None
    actual = _digest_files(path)
    if actual != data["files"]:
        changed = sorted(set(actual) ^ set(data["files"]) |
                         {f for f in actual if f in data["files"] and actual[f] != data["files"][f]})
        raise SignatureError(f"{path} was modified after signing: {changed[:5]}")
    return meta


def verify_if_required(path, public_key=None) -> dict | None:
    """Verify when a key is given or ``AC_MODEL_PUBLIC_KEY`` is set (used by ``load_model``)."""
    key = public_key or os.environ.get("AC_MODEL_PUBLIC_KEY")
    if not key:
        return None
    return verify_model(path, key)
