# Highlight Studio

SmolVLM2 video highlight generator. The model watches a video in segments, scores each
for visual drama, audio peaks, and (optionally) what's said, then cuts the strong moments
into one reel. Now a FastAPI backend + a single static HTML page (no Gradio).

## Layout

```
app.py             FastAPI: upload, background job, SSE progress, Q&A, file serving
highlight_core.py  All analysis logic — model, ffmpeg, scoring, selection, export.
                   Framework-agnostic; no web code.
static/index.html  Single-page frontend (vanilla JS, clean light theme, canvas timeline)
```

The split means the core can be imported and driven from a script, a worker, or a
different web layer without touching the analysis code.

## Run

```bash
pip install fastapi uvicorn pydantic python-multipart \
            torch transformers decord
# optional, only if you enable the speech signal:
pip install openai-whisper
# ffmpeg must be on PATH (with the ebur128, scene, and tile filters — standard builds have these)

python app.py            # serves on http://0.0.0.0:8000
# or: uvicorn app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000.

## Environment variables

| var | default | meaning |
|-----|---------|---------|
| `MODEL_NAME` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | vision-language model |
| `WHISPER_MODEL` | `base` | Whisper size for the speech signal |
| `HIGHLIGHT_OUTPUT_DIR` | system temp | where reels/exports are written |
| `PORT` | `8000` | server port |

## How scoring works

Each segment gets a fused 0–10 score:

```
fused = (w_visual·visual + w_audio·audio + w_speech·speech) / (sum of active weights)
```

- **visual** — SmolVLM2 rates the footage against the highlight criteria
- **audio** — peak loudness (ffmpeg ebur128) normalized across the video
- **speech** — keyword overlap between the criteria and the Whisper transcript

Weights are adjustable live in the UI. Selection is by score threshold, best N moments,
or best N seconds. Kept segments are merged, padded, ratio-capped, and concatenated
(with optional crossfade).

## Outputs

Every run produces a reel MP4, a `chapters.json` (timestamps + scores), a `chapters.vtt`
(loads into any player), and a contact-sheet JPG. All downloadable from the results panel.

## Notes

- One job runs at a time (single in-process inference lock); a second request is told to
  wait rather than queued. Fine for single-user / small-team use; swap in a real task queue
  (Celery/RQ) if you need concurrency.
- Per-video analysis (description, transcript, loudness, scene cuts) is cached by content
  hash, so re-runs with different selection settings and Q&A are fast.
- Max video length is 30 minutes (`MAX_VIDEO_SECONDS` in `highlight_core.py`).
