# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
# Activate the local venv first
source .venv/bin/activate

# Start the server (default port 8019; override with PORT env var)
python app.py

# Or via uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8019
```

Open http://localhost:8019 in a browser.

`start.sh` automates venv creation and dep installation, but references the now-obsolete `webapp.py` — use the commands above instead.

**External requirement:** `ffmpeg` and `ffprobe` must be on `PATH` (standard builds include the ebur128, scene, and tile filters). `openai-whisper` is optional (speech signal only).

## Installing Dependencies

```bash
source .venv/bin/activate
pip install -r requirements.txt
# Optional speech signal:
pip install openai-whisper
```

## Environment Variables

| Variable | Default | Notes |
|---|---|---|
| `MODEL_NAME` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | HuggingFace vision-language model |
| `WHISPER_MODEL` | `base` | Whisper size (`tiny`, `base`, `small`, `medium`, `large`) |
| `HIGHLIGHT_OUTPUT_DIR` | system temp | Where reels, chapters JSON/VTT, and contact sheets are written |
| `PORT` | `8019` | Uvicorn bind port |

## Architecture

The codebase has a strict two-layer split:

**`highlight_core.py`** — framework-agnostic analysis engine. Everything ML and ffmpeg lives here: model loading, video decoding, segment scoring, audio loudness, Whisper transcription, segment selection and merging, crossfade concatenation, VTT/JSON export, contact sheet generation. No FastAPI imports.

**`app.py`** — thin FastAPI orchestration layer. Handles file upload, job lifecycle, SSE progress streaming, and file serving. All heavy work is delegated to `highlight_core`. Job state (`_JOBS`, `_VIDEOS`) lives in process memory (no persistent store).

**`static/index.html`** — self-contained single-page frontend. Vanilla JS, no build step. Canvas-based score timeline, drag-and-drop upload, live SSE progress, Q&A panel.

### Key Design Decisions

**Single-inference lock** (`_INFERENCE_LOCK` in `highlight_core.py`): only one job runs at a time. A second request gets an immediate error telling the user to retry; there is no queue. This is intentional for single-user use.

**Per-video cache** (`_VIDEO_CACHE`, keyed by `video_hash()`): analysis results (video description, auto-detected highlights, loudness timeline, scene cuts, Whisper transcript) are stored in memory keyed by a hash of file size + first/last 1 MB. Re-runs with different selection settings reuse cached analysis and are fast.

**Fast-frames path** (`fast_frames=True` default): segments are scored by sampling `FRAMES_PER_SEGMENT` (8) JPEG frames via ffmpeg rather than cutting intermediate MP4s. This is the default for speed. The full-clip path (`fast_frames=False`) sends an MP4 to the model instead.

**`cut_segment()` fallback**: stream-copy is attempted first for speed; if the resulting file isn't decodable (common at non-keyframe boundaries), it falls back to a full libx264 re-encode.

**Score fusion formula**:
```
fused = (w_visual·visual + w_audio·audio + w_speech·speech) / (sum of active weights)
```
Inactive signals (audio/speech disabled) contribute zero weight to the denominator.

### API Endpoints

| Method | Path | Returns |
|---|---|---|
| `POST` | `/api/upload` | `{video_id, duration, video_url}` |
| `POST` | `/api/process` | `{job_id}` — starts background job |
| `GET` | `/api/events/{job_id}` | SSE stream: `progress`, `done`, `error` events |
| `POST` | `/api/ask` | `{job_id}` — Q&A, also streamed via SSE |
| `GET` | `/api/video/{video_id}` | Raw video (Range-request capable) |
| `GET` | `/files/{name}` | Generated reel MP4, chapters JSON/VTT, contact sheet JPG |
| `GET` | `/` | `static/index.html` |

### Tunable Constants in `highlight_core.py`

| Constant | Default | Meaning |
|---|---|---|
| `MAX_VIDEO_SECONDS` | `1800` | 30-minute hard cap |
| `DEFAULT_SCORE_THRESHOLD` | `6.0` | Threshold mode default |
| `MAX_HIGHLIGHT_RATIO` | `0.6` | Max fraction of video kept as highlights |
| `SCENE_THRESHOLD` | `0.4` | ffmpeg scene-change sensitivity (higher = fewer cuts) |
| `FRAMES_PER_SEGMENT` | `8` | Frames sampled per segment for fast-frames scoring |

### Note on `.pi/repo-knowledge/`

These files describe a completely different project (SpeqForge) and should be ignored — they do not reflect this codebase.
