"""Accounts: scrypt password hashes in SQLite plus HS256 signed tokens (stdlib only)."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import config

TOKEN_TTL_SECONDS = 7 * 24 * 3600
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
MIN_PASSWORD_LEN = 8


class AuthError(Exception):
    pass


@dataclass(frozen=True)
class User:
    id: str
    username: str

    @property
    def data_dir(self) -> Path:
        return config.USERS_ROOT / self.id / "data"

    @property
    def storage_dir(self) -> Path:
        return config.USERS_ROOT / self.id / "storage"


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _secret() -> bytes:
    env = os.getenv("AUTH_SECRET")
    if env:
        return env.encode()
    path = config.USERS_ROOT / ".auth_secret"
    config.USERS_ROOT.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(secrets.token_hex(32))
        path.chmod(0o600)
    return path.read_text().strip().encode()


def issue_token(user: User, ttl: int = TOKEN_TTL_SECONDS) -> str:
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64(json.dumps({"sub": user.id, "name": user.username, "exp": int(time.time()) + ttl}).encode())
    sig = _b64(hmac.new(_secret(), f"{header}.{payload}".encode(), hashlib.sha256).digest())
    return f"{header}.{payload}.{sig}"


def decode_token(token: str) -> User:
    try:
        header, payload, sig = token.split(".")
        expected = _b64(hmac.new(_secret(), f"{header}.{payload}".encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            raise AuthError("invalid token")
        data = json.loads(_unb64(payload))
    except AuthError:
        raise
    except Exception as exc:
        raise AuthError("invalid token") from exc
    if data.get("exp", 0) < time.time():
        raise AuthError("token expired")
    return User(id=data["sub"], username=data["name"])


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)


class UserStore:
    def __init__(self):
        self._lock = threading.Lock()

    def _conn(self) -> sqlite3.Connection:
        config.USERS_ROOT.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(config.USERS_ROOT / "users.db")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS users ("
            "id TEXT PRIMARY KEY, username TEXT UNIQUE COLLATE NOCASE NOT NULL, "
            "salt BLOB NOT NULL, hash BLOB NOT NULL, created REAL NOT NULL)"
        )
        return conn

    def register(self, username: str, password: str) -> User:
        username = (username or "").strip()
        if not _USERNAME_RE.match(username):
            raise AuthError("Username must be 3-32 characters: letters, numbers, . _ -")
        if len(password or "") < MIN_PASSWORD_LEN:
            raise AuthError(f"Password must be at least {MIN_PASSWORD_LEN} characters")
        user = User(id=uuid.uuid4().hex, username=username)
        salt = secrets.token_bytes(16)
        with self._lock, self._conn() as conn:
            try:
                conn.execute(
                    "INSERT INTO users VALUES (?, ?, ?, ?, ?)",
                    (user.id, username, salt, _hash_password(password, salt), time.time()),
                )
            except sqlite3.IntegrityError as exc:
                raise AuthError("That username is taken") from exc
        self._seed(user)
        return user

    def login(self, username: str, password: str) -> User:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, username, salt, hash FROM users WHERE username = ?", ((username or "").strip(),)
            ).fetchone()
        # Hash even when the user is unknown so timing doesn't reveal which names exist.
        salt = row[2] if row else b"\0" * 16
        candidate = _hash_password(password or "", salt)
        if not row or not hmac.compare_digest(candidate, row[3]):
            raise AuthError("Wrong username or password")
        return User(id=row[0], username=row[1])

    @staticmethod
    def _seed(user: User) -> None:
        """Give a new account the shipped sample doc so the first question works."""
        sample = config.DATA_DIR / "sample.md"
        if sample.exists():
            user.data_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(sample, user.data_dir / "sample.md")


users = UserStore()
