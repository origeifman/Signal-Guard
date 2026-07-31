"""Broker-credential envelope: how an API key/secret pair is stored and read.

The key and secret are serialised to one JSON object and AES-GCM encrypted as a
single blob (§12). This lives in one module so the write side (saving an account)
and the read side (building an adapter) can never disagree about the format.

Plaintext exists only for the moment of a call and is never logged or returned by
the API — `BrokerAccountOut` deliberately has no credential field.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from signalguard.crypto import decrypt_credential, encrypt_credential


@dataclass(frozen=True)
class BrokerCredentials:
    api_key: str
    api_secret: str


def seal_credentials(
    creds: BrokerCredentials, master_key_b64: str
) -> tuple[bytes, bytes]:
    """Return (ciphertext, nonce) for storage in `broker_accounts`."""
    blob = json.dumps({"api_key": creds.api_key, "api_secret": creds.api_secret})
    return encrypt_credential(blob, master_key_b64)


def open_credentials(
    ciphertext: bytes, nonce: bytes, master_key_b64: str
) -> BrokerCredentials:
    """Decrypt stored credentials. Call at the moment of use, never at load time."""
    data: dict[str, str] = json.loads(
        decrypt_credential(ciphertext, nonce, master_key_b64)
    )
    return BrokerCredentials(api_key=data["api_key"], api_secret=data["api_secret"])
