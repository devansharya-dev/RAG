"""Cryptographic and stable content hashing utilities."""

import hashlib


def compute_sha256(content: str | bytes) -> str:
    """Compute the SHA-256 hexadecimal digest for given string or bytes content.

    Args:
        content: String or bytes to hash.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()
