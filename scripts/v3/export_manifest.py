"""Generate docs/v3/DATA_MANIFEST.jsonl from krmm.v3.knowledge_registry (no gold answers leaked to public manifest)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from krmm.v3.knowledge_registry import manifest, public_units


def main() -> None:
    out = Path("docs/v3/DATA_MANIFEST.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    m = manifest()
    lines = [json.dumps({"manifest_header": m}, ensure_ascii=False)]
    for u in public_units():
        lines.append(json.dumps(u.to_dict(), ensure_ascii=False))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)-1} public units to {out} (sha256={m['catalog_sha256']})")


if __name__ == "__main__":
    main()
