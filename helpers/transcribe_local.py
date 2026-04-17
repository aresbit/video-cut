"""Transcribe a video with local faster-whisper (small model).

Extracts mono 16kHz audio via ffmpeg, transcribes with word-level timestamps,
and writes a Scribe-compatible JSON to <edit_dir>/transcripts/<video_stem>.json.

Cached: if the output file already exists, transcription is skipped.

Usage:
    python helpers/transcribe_local.py <video_path>
    python helpers/transcribe_local.py <video_path> --edit-dir /custom/edit
    python helpers/transcribe_local.py <video_path> --language en
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from faster_whisper import WhisperModel

MODEL_SIZE = "small"


def extract_audio(video_path: Path, dest: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(dest),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def get_model() -> WhisperModel:
    return WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")


def transcribe_one(
    video: Path,
    edit_dir: Path,
    model: WhisperModel,
    language: str | None = None,
    verbose: bool = True,
) -> Path:
    transcripts_dir = edit_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    out_path = transcripts_dir / f"{video.stem}.json"

    if out_path.exists():
        if verbose:
            print(f"cached: {out_path.name}")
        return out_path

    if verbose:
        print(f"  extracting audio from {video.name}", flush=True)

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / f"{video.stem}.wav"
        extract_audio(video, audio)
        size_mb = audio.stat().st_size / (1024 * 1024)
        if verbose:
            print(f"  transcribing {video.stem}.wav ({size_mb:.1f} MB) with faster-whisper {MODEL_SIZE}", flush=True)

        segments, info = model.transcribe(
            str(audio),
            language=language,
            word_timestamps=True,
            condition_on_previous_text=True,
        )

        # Convert generator to list so we can iterate multiple times if needed
        segments = list(segments)

    words: list[dict] = []
    for segment in segments:
        seg_words = segment.words or []
        for i, w in enumerate(seg_words):
            words.append({
                "type": "word",
                "text": w.word.strip(),
                "start": w.start,
                "end": w.end,
                "speaker_id": "speaker_0",
            })
            # Insert spacing entry between words to match Scribe gap format
            if i < len(seg_words) - 1:
                next_w = seg_words[i + 1]
                words.append({
                    "type": "spacing",
                    "text": " ",
                    "start": w.end,
                    "end": next_w.start,
                })

    # Build Scribe-compatible payload
    payload: dict = {
        "language_code": info.language,
        "language_probability": info.language_probability,
        "words": words,
        "segments": [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text.strip(),
            }
            for s in segments
        ],
    }

    out_path.write_text(json.dumps(payload, indent=2))
    dt = time.time() - t0

    if verbose:
        kb = out_path.stat().st_size / 1024
        print(f"  saved: {out_path.name} ({kb:.1f} KB) in {dt:.1f}s")
        print(f"    words: {len([w for w in words if w.get('type') == 'word'])}")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe a video with local faster-whisper")
    ap.add_argument("video", type=Path, help="Path to video file")
    ap.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Edit output directory (default: <video_parent>/edit)",
    )
    ap.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional ISO language code (e.g., 'en'). Omit to auto-detect.",
    )
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")

    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()
    model = get_model()

    transcribe_one(
        video=video,
        edit_dir=edit_dir,
        model=model,
        language=args.language,
    )


if __name__ == "__main__":
    main()
