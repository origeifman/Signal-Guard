"""Build a broker adapter for a stored account.

Credentials are decrypted here, in memory, at the moment an adapter is needed —
never at load time, never cached in plaintext (§12). One place constructs
adapters so the mapping from a stored `broker` string to a concrete adapter class
lives in exactly one spot.
"""

from __future__ import annotations

from signalguard.db.models import BrokerAccount
from signalguard.execution.base import BrokerAdapter
from signalguard.execution.binance_testnet import BinanceTestnetAdapter
from signalguard.execution.credentials import open_credentials

# The only broker implemented in v1 (CLAUDE.md §9 — design the interface, build
# exactly one). Unknown values fail closed rather than defaulting to something.
_SUPPORTED = {"binance_spot_testnet"}


def build_adapter(account: BrokerAccount, master_key_b64: str) -> BrokerAdapter:
    if account.broker not in _SUPPORTED:
        raise ValueError(f"unsupported broker: {account.broker!r}")
    creds = open_credentials(
        account.encrypted_credentials, account.credentials_nonce, master_key_b64
    )
    return BinanceTestnetAdapter(creds.api_key, creds.api_secret)
