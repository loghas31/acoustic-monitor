"""
record_session.py — CHECK 4/5 / the week-2 dataset recorder.

Records timestamped wav + accelerometer csv pairs in exactly the format that
`capture.FileSource` reads and `ml/realdata/analyse_recording.py` analyses, plus
a JSON sidecar of metadata.

    python firmware/bench/record_session.py --machine "grinder" --label healthy \
        --minutes 120 --rpm 2850 --bearing 6204 --out data/sessions

WHY THE METADATA SIDECAR MATTERS MORE THAN THE AUDIO
----------------------------------------------------------------------------
In six weeks you will have forty recordings and no memory of which motor,
which bearing, which mounting position, or whether the fault was in yet. A
recording without provenance is not data, it is noise you paid for. Every
run therefore captures: machine name, healthy/faulty label, bearing
designation, nominal RPM, mounting position, operator, and free-text notes.

The `--label` field is the single most important one: it is the ground truth
that turns a pile of recordings into a labelled dataset you can compute an
ROC curve from. Be strict about it. "healthy" means you know it was healthy.

CHUNKING: long recordings are written in segments (default 5 minutes) rather
than one enormous file, so that (a) a crash loses one segment not two hours,
(b) files stay small enough to move around, and (c) each segment is already
close to the analysis window length.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench_common import Report, run_guarded  # noqa: E402

LABELS = ("healthy", "faulty", "unknown", "seeded-outer", "seeded-inner")


def write_segment(out_dir: Path, stem: str, audio: np.ndarray, fs_audio: float,
                  accel: np.ndarray | None, fs_accel: float, meta: dict) -> Path:
    from scipy.io import wavfile
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{stem}.wav"

    # Fixed full-scale, NOT per-file normalisation: normalising each file would
    # destroy the amplitude relationship between healthy and faulty recordings,
    # which is part of the signal we are trying to detect.
    pcm = np.clip(audio, -1.0, 1.0)
    wavfile.write(wav_path, int(fs_audio), (pcm * 32767).astype(np.int16))

    if accel is not None and len(accel):
        t = np.arange(len(accel)) / fs_accel
        cols = accel if accel.ndim == 2 else accel[:, None]
        np.savetxt(out_dir / f"{stem}.csv",
                   np.column_stack([t, cols]), delimiter=",",
                   header="t_s," + ",".join("xyz"[:cols.shape[1]]),
                   comments="", fmt="%.6f")

    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    return wav_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--machine", required=True, help="e.g. 'bench grinder #1'")
    p.add_argument("--label", required=True, choices=LABELS,
                   help="GROUND TRUTH. Be honest; this is what ROC curves rest on.")
    p.add_argument("--minutes", type=float, default=10.0)
    p.add_argument("--segment-minutes", type=float, default=5.0)
    p.add_argument("--out", type=Path, default=Path("data/sessions"))
    p.add_argument("--rpm", type=float, default=None, help="nominal shaft RPM if known")
    p.add_argument("--bearing", default=None, help="designation, e.g. 6204")
    p.add_argument("--position", default="motor end-shield",
                   help="where the sensor is mounted")
    p.add_argument("--operator", default="")
    p.add_argument("--notes", default="")
    p.add_argument("--fs-audio", type=int, default=16000)
    p.add_argument("--fs-accel", type=int, default=6400)
    p.add_argument("--no-accel", action="store_true", help="mic-only recording")
    args = p.parse_args(argv)

    report = Report("CHECK 4/5 — SESSION RECORDER")
    report.header()

    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out / f"{args.machine.replace(' ', '_')}_{args.label}_{session}"
    n_segments = max(1, int(round(args.minutes / args.segment_minutes)))
    seg_s = args.segment_minutes * 60.0

    report.info(f"machine   : {args.machine}")
    report.info(f"label     : {args.label}   <-- ground truth")
    report.info(f"duration  : {args.minutes:.0f} min in {n_segments} segment(s)")
    report.info(f"output    : {out_dir}")

    from capture import HardwareSource
    src = HardwareSource(window_s=seg_s, fs_audio=args.fs_audio,
                         fs_accel=args.fs_accel,
                         require_accel=not args.no_accel)

    base_meta = {
        "session": session, "machine": args.machine, "label": args.label,
        "bearing": args.bearing, "rpm": args.rpm, "position": args.position,
        "operator": args.operator, "notes": args.notes,
        "fs_audio": args.fs_audio, "fs_accel": args.fs_accel,
        "schema": "acoustic-monitor/recording/1",
    }

    written = 0
    t0 = time.monotonic()
    for i, (audio, accel) in enumerate(src.windows()):
        if i >= n_segments:
            break
        stem = f"seg{i:03d}"
        meta = dict(base_meta, segment=i,
                    recorded_utc=datetime.now(timezone.utc).isoformat(),
                    duration_s=len(audio) / args.fs_audio)
        write_segment(out_dir, stem, np.asarray(audio, float), args.fs_audio,
                      None if args.no_accel else np.asarray(accel, float),
                      args.fs_accel, meta)
        written += 1
        report.info(f"  wrote {stem} ({(i+1)*args.segment_minutes:.0f}/"
                    f"{args.minutes:.0f} min)")

    elapsed = (time.monotonic() - t0) / 60.0
    report.check("segments written", written == n_segments,
                 f"{written}/{n_segments} in {elapsed:.1f} min")
    report.advise(f"Analyse a healthy/faulty pair with:\n"
                  f"  python ml/realdata/analyse_recording.py "
                  f"--healthy <healthy>/seg000.wav --faulty <faulty>/seg000.wav"
                  + (f" --bearing {args.bearing}" if args.bearing else "")
                  + (f" --rpm {args.rpm:.0f}" if args.rpm else ""))
    return report.finish()


if __name__ == "__main__":
    raise SystemExit(run_guarded(main))
