"""
download_demo_data.py
─────────────────────────────────────────────────────────────────────────────
Pre-download MIT-BIH Arrhythmia Database records for offline demo use.

Run this ONCE on the RPi before presentation day to cache records locally
so the demo never needs a network connection during the actual presentation.

Usage:
    python download_demo_data.py

Records downloaded:
    100 — Normal sinus rhythm (healthy patient reference)
    119 — Premature Ventricular Contractions (PVCs) — default simulator record
    200 — Ventricular bigeminy (every other beat is PVC — most dramatic)
    201 — Atrial fibrillation + PVCs (highest BPM variability)

Output: backend/demo_data/  (.dat + .hea files for each record)
─────────────────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path

# ── Ensure demo_data directory exists ────────────────────────────────────────
ROOT_DIR  = Path(__file__).resolve().parent
DATA_DIR  = ROOT_DIR / "demo_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEMO_RECORDS = {
    "100": "Normal sinus rhythm — healthy patient reference",
    "119": "PVCs — Premature Ventricular Contractions (default demo)",
    "200": "Ventricular bigeminy — PVC every other beat (most dramatic for demo)",
    "201": "Atrial fibrillation + PVCs — highest BPM variability",
}

def main():
    try:
        import wfdb
    except ImportError:
        print("ERROR: wfdb is not installed.")
        print("Install it with: pip install wfdb>=4.1.0")
        sys.exit(1)

    print("=" * 60)
    print("  ECG Demo Data Pre-Downloader")
    print(f"  Output directory: {DATA_DIR}")
    print("=" * 60)

    failed = []
    for record_id, description in DEMO_RECORDS.items():
        dest = DATA_DIR / record_id
        # Check if already cached (both .dat and .hea must exist)
        if (DATA_DIR / f"{record_id}.dat").exists() and \
           (DATA_DIR / f"{record_id}.hea").exists():
            print(f"  [SKIP] {record_id} — already cached  ({description})")
            continue

        print(f"\n  [DOWNLOAD] {record_id} — {description}")
        try:
            wfdb.dl_database(
                "mitdb",
                dl_dir=str(DATA_DIR),
                records=[record_id],
            )
            print(f"  [OK]  {record_id} downloaded → {DATA_DIR}")
        except Exception as e:
            print(f"  [FAIL] {record_id}: {e}")
            failed.append(record_id)

    print("\n" + "=" * 60)
    if failed:
        print(f"  WARNING: Failed to download: {', '.join(failed)}")
        print("  Simulator will fall back to synthetic ECG for those records.")
    else:
        print("  All records cached successfully. Demo is ready offline.")
    print("=" * 60)


if __name__ == "__main__":
    main()
