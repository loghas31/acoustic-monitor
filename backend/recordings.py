"""
recordings.py — turn an uploaded phone recording into a verdict, using the
pipeline that already exists.

The rule this module follows: **no new signal processing lives here.** It
converts the upload to a WAV, hands it to `tools/ingest.py`'s reader and
`tools/phone_monitor.py`'s `analyse()`, and stores what comes back. If a
number appears in a verdict, it was produced by the same code the firmware
runs — there is no separate "phone" analysis path to keep in step.

Processing is deliberately out-of-band. A 28-minute recording takes roughly a
minute to analyse; a phone on mobile data will not hold an HTTP request open
that long, and a Shortcut that appears to hang is a Shortcut nobody uses. So
upload returns `202 queued` with an id, and the phone polls.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for sub in ("firmware", "ml", "tools"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.append(p)

# Where uploads land. Under /tmp by default because the repo mount forbids
# deletion and SQLite/large files there have bitten this project before
# (F12). Override with ACOUSTIC_UPLOAD_DIR in production.
import os

UPLOAD_DIR = Path(os.environ.get("ACOUSTIC_UPLOAD_DIR",
                                 f"/tmp/acoustic_uploads_{os.getuid()}"))

# iOS "Record Audio" produces m4a. Everything else a phone might send is
# listed so the failure message can be specific rather than "unsupported".
_NEEDS_CONVERSION = {".m4a", ".mp3", ".aac", ".caf", ".mp4", ".ogg", ".opus"}
TARGET_SR = 16000


def _to_wav(src: Path) -> Path:
    """Return a 16 kHz mono WAV. Shells out to ffmpeg for compressed formats.

    ffmpeg is an explicit dependency of the phone route and nothing else, so
    its absence is reported as a deployment problem with the fix in the
    message, rather than surfacing as an opaque decode error.
    """
    if src.suffix.lower() not in _NEEDS_CONVERSION:
        return src
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"cannot decode {src.suffix} — ffmpeg is not installed on the "
            f"server. Install it (apt install ffmpeg / brew install ffmpeg), "
            f"or have the phone upload WAV instead.")
    dst = src.with_suffix(".converted.wav")
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src),
         "-ac", "1", "-ar", str(TARGET_SR), str(dst)],
        capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {src.name}: "
                           f"{proc.stderr.decode()[:300]}")
    return dst


def analyse_recording(path: Path, learn_windows: int = 48,
                      window_s: float = 30.0) -> dict:
    """Run the real pipeline. Returns phone_monitor's own summary dict.

    Raises with a legible message when the recording is too short — which is
    the single most likely user error, because a phone recording feels long
    and 48 windows of 30 s is 24 minutes before ANY window can be scored.
    """
    import importlib.util
    wav = _to_wav(path)

    spec = importlib.util.spec_from_file_location(
        "phone_monitor", ROOT / "tools" / "phone_monitor.py")
    pm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pm)

    rows, summary = pm.analyse(wav, None, window_s=window_s,
                               learn_windows=learn_windows)
    summary = dict(summary)
    summary["rows"] = rows[-24:]          # tail only; the phone shows a chart
    summary["n_scored"] = len(rows)
    summary["learn_windows"] = learn_windows
    # DOC_STATUS: below 48 learn windows the held-out false-alarm rate is
    # 55-59 %, i.e. noise. Anything shorter is a plumbing check, and the
    # verdict must say so rather than looking like a health verdict.
    summary["learn_period_too_short"] = learn_windows < 48
    return summary


def process(recording_id: str, session_factory, learn_windows: int = 48) -> None:
    """Background worker. Never raises — a crashed worker that leaves a row
    stuck at 'running' forever is worse than one that records why it failed.
    """
    db = session_factory()
    try:
        import models
        rec = db.query(models.Recording).filter_by(id=recording_id).first()
        if rec is None:
            return
        rec.status = "running"
        db.commit()
        try:
            rec.verdict = analyse_recording(Path(rec.path),
                                            learn_windows=learn_windows)
            rec.status = "done"
            rec.error = ""
        except Exception as e:                       # noqa: BLE001
            rec.status = "failed"
            # The message the phone sees. Keep the first line human; the
            # traceback goes to the server log, not to a Shortcut alert.
            rec.error = f"{type(e).__name__}: {e}"[:800]
            print(f"[recordings] {recording_id} failed:\n"
                  f"{traceback.format_exc()}", file=sys.stderr)
        db.commit()
    finally:
        db.close()
