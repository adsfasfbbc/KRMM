# Complete terminal log archive

The byte-preserved stdout/stderr logs from all 20 model-trial launches are stored in [terminal-logs.tar.gz](../artifacts/terminal-logs.tar.gz). Compression avoids committing tens of megabytes of repeated framework warnings. No failure line is removed from the archive.

A readable per-launch preview is in [terminal-preview.txt](../reports/terminal-preview.txt); complete run metadata remains under .autoresearch/runs/. The public manifest records both the archive checksum and every original uncompressed log checksum.

To inspect a single log without changing original evidence:

```bash
tar -tzf artifacts/terminal-logs.tar.gz
tar -xOzf artifacts/terminal-logs.tar.gz .autoresearch/runs/RUN_ID/stderr.log
```
