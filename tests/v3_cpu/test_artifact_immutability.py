"""Verify all strictly protected historical artifacts remain byte-for-byte unchanged."""
from __future__ import annotations

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_PROTECTED_SHA256 = {
    "docs/v3/EXPERIMENTS.md": "c34d89ecafff6889f6e676eb34f639854e68d5321ccb8c31849f1011bb0a4b63",
    "docs/v3/GATE_REPORT.md": "f7d6f9f51e2cc9a19888cad81e02bc4cb077ff848cb8635dbcfc131e00401961",
    "docs/v3/EVIDENCE_PACK.md": "a33e9ecb5a2cbe2fc57f61d775975d89f2750776a36e86882c404c285d0516c7",
    "docs/v3/PROTOCOL.md": "46ccd1419799a6f2430b8af41b55c0c4bf53f07e8e94ca3ad022f08cd1fe2abd",
    "docs/v3/DATA_MANIFEST.jsonl": "b4a1fc522492868c7360e921b7e02e787f94b91821c9bc7c3768e8b47e382b4d",
    "results/v3-new/e0_e7_full_final.json": "0a4b6c596747e0a540e2573a7fe209d7d9b9100d1a66883d1636fb97b19fbab3",
    "configs/v3/p0_d2_steps64.json": "b4695d61f96efa8411e1fe360483456069a16a4497ffa735fd98838ea54a9681",
    "configs/v3/p2_dev12_orth_e6.json": "74d29687a82a82671d20b24e21030ed64475642ba2f82e6f145e9c33cdd7f9c0",
    "configs/v3/p2_dev36_full.json": "d89fe1c6a8b867edbc00a42563c91bc88e063d0f1d0dd268e6aa4e9b05062fa9",
    "configs/v3/p2_cal36_full.json": "ea354cf94be5337c4c43a803e0f5cf5ab84a35f6228c397e49771ef9852b4749",
    "configs/v3/p2_test48_frozen.json": "08b5831a55329bd71bf1e33f509b18a0be84d88ead17d2ca7190de7a3ba6013f",
}


def test_protected_historical_artifacts_unchanged():
    for rel, expected in EXPECTED_PROTECTED_SHA256.items():
        p = REPO_ROOT / rel
        assert p.exists(), f"Missing protected file: {rel}"
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        assert actual == expected, f"Protected file modified: {rel} ({actual} != {expected})"
