"""
Auth: per-device API keys + JWT for dashboard users.

Choices and why:
* Device API key = 32 random bytes, shown ONCE at registration, stored only as
  SHA-256. Devices are provisioned programmatically, so "show once" costs
  nothing and a database dump exposes no credentials.
* Plain SHA-256 (no salt/stretching) is correct for 256-bit random keys —
  salting defends low-entropy secrets like passwords; a random 256-bit key
  cannot be dictionary-attacked. Passwords, which ARE low-entropy, get scrypt.
* scrypt from hashlib (stdlib) instead of passlib/bcrypt: one less native
  dependency, memory-hard, recommended parameters from the Python docs.
* JWT via PyJWT, HS256, 24 h expiry. Fine for a single-backend MVP; switch to
  RS256 + refresh tokens if a mobile app ever appears.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import time

import jwt

JWT_SECRET = os.environ.get("JWT_SECRET", "dev-secret-change-me")
JWT_TTL_S = 24 * 3600


# -- device API keys ----------------------------------------------------------

def new_api_key() -> tuple[str, str]:
    """(plaintext_key, stored_hash). Plaintext leaves this process exactly once."""
    key = secrets.token_urlsafe(32)
    return key, hash_api_key(key)


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


# -- user passwords -----------------------------------------------------------

_SCRYPT = dict(n=2 ** 14, r=8, p=1)     # ~16 MB memory cost: slow for attackers,
                                        # imperceptible for a login

def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT)
    return salt.hex() + ":" + dk.hex()


def verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, dk_hex = stored.split(":")
        dk = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), **_SCRYPT)
        return secrets.compare_digest(dk.hex(), dk_hex)
    except (ValueError, AttributeError):
        return False


# -- JWT ----------------------------------------------------------------------

def make_token(user_id: str) -> str:
    now = int(time.time())
    return jwt.encode({"sub": user_id, "iat": now, "exp": now + JWT_TTL_S},
                      JWT_SECRET, algorithm="HS256")


def verify_token(token: str) -> str | None:
    """Returns user_id, or None if invalid/expired."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])["sub"]
    except jwt.PyJWTError:
        return None
