#!/usr/bin/env python3
"""FastAPI backend for the SmolVLM2 highlight generator.

Serves a single-page frontend (static/index.html) and exposes:
  POST /api/upload          -> {video_id}
  POST /api/process         -> {job_id}     (starts background job)
  GET  /api/events/{job_id} -> SSE stream of progress + final result
  POST /api/ask             -> {job_id}     (Q&A, also streamed via SSE)
  GET  /files/...           -> generated reel / exports / contact sheet

All heavy lifting lives in highlight_core.py; this module only orchestrates.
"""

import json
import logging
import os
import queue
import tempfile
import threading
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import highlight_core as hc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

app = FastAPI(title="Highlight Studio")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "highlight_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(hc.OUTPUT_DIR, exist_ok=True)

# Job registry: job_id -> {"queue": Queue, "thread": Thread}
_JOBS = {}
_JOBS_LOCK = threading.Lock()
# Uploaded videos: video_id -> path
_VIDEOS = {}


def _new_job():
    job_id = uuid.uuid4().hex
    q = queue.Queue()
    with _JOBS_LOCK:
        _JOBS[job_id] = {"queue": q}
    return job_id, q


def _emit(q, event, **data):
    """Push an SSE event onto the job queue."""
    q.put({"event": event, "data": data})


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ProcessRequest(BaseModel):
    video_id: str
    highlight_prompt: str = ""
    mode: str = "threshold"  # threshold | best_n | best_seconds
    threshold: float = hc.DEFAULT_SCORE_THRESHOLD
    n_moments: int = 5
    n_seconds: float = 60.0
    w_visual: float = 1.0
    w_audio: float = 0.35
    w_speech: float = 0.5
    use_audio: bool = True
    use_speech: bool = False
    segment_length: float = 10.0
    padding: float = 0.5
    crossfade: float = 0.4
    use_scenes: bool = True
    overlap: bool = False
    fast_frames: bool = True


class AskRequest(BaseModel):
    video_id: str
    question: str
    segment_length: float = 10.0
    use_speech: bool = False


# ---------------------------------------------------------------------------
# Core job: highlight generation
# ---------------------------------------------------------------------------


def _run_process(job_id: str, q: queue.Queue, req: ProcessRequest):
    video_path = _VIDEOS.get(req.video_id)
    try:
        if not video_path or not os.path.exists(video_path):
            _emit(q, "error", message="Video not found. Re-upload and try again.")
            return

        if not hc._INFERENCE_LOCK.acquire(blocking=False):
            _emit(q, "error", message="The model is busy with another request. Try again shortly.")
            return

        try:
            duration = hc.get_video_duration_seconds(video_path)
            if duration > hc.MAX_VIDEO_SECONDS:
                _emit(q, "error", message="That video is over 30 minutes. Trim it and try again.")
                return

            vhash = hc.video_hash(video_path)
            with hc._VIDEO_CACHE_LOCK:
                cache = hc._VIDEO_CACHE.setdefault(vhash, {})

            processor, model = hc.get_or_load_model_and_processor(hc.MODEL_NAME)

            _emit(q, "progress", message="Reading the video…", pct=2)
            if "desc" not in cache:
                cache["desc"] = hc.analyze_video_content(processor, model, video_path)
            video_desc = cache["desc"]

            if req.highlight_prompt.strip():
                criteria = req.highlight_prompt.strip()
                criteria_source = "your prompt"
            else:
                _emit(q, "progress", message="Deciding what to look for…", pct=6)
                criteria = cache.get("auto_highlights") or hc.determine_highlights(processor, model, video_desc)
                cache["auto_highlights"] = criteria
                criteria_source = "auto-detected"
            keywords = hc.extract_keywords(criteria)

            loudness, lf, lc = [], 0.0, 0.0
            if req.use_audio:
                _emit(q, "progress", message="Measuring audio…", pct=10)
                if "loudness" not in cache:
                    cache["loudness"] = hc.compute_loudness_timeline(video_path)
                loudness = cache["loudness"]
                if loudness:
                    vals = [v for _, v in loudness]
                    lf, lc = min(vals), max(vals)

            transcript = []
            if req.use_speech:
                _emit(q, "progress", message="Transcribing speech…", pct=14)
                if "transcript" not in cache:
                    cache["transcript"] = hc.transcribe_video(video_path)
                transcript = cache["transcript"]

            if req.use_scenes:
                _emit(q, "progress", message="Finding scene cuts…", pct=18)
                scene_times = cache.get("scenes")
                if scene_times is None:
                    scene_times = hc.detect_scene_times(video_path)
                    cache["scenes"] = scene_times
                windows = hc.snap_windows_to_scenes(duration, req.segment_length, scene_times)
            else:
                windows = hc.build_windows(duration, req.segment_length, req.overlap)
            total = max(1, len(windows))

            scored_windows = []
            with tempfile.TemporaryDirectory(prefix="seg_") as temp_dir:
                for i, (start_time, end_time) in enumerate(windows):
                    pct = 20 + int((i / total) * 60)
                    _emit(q, "progress", message=f"Scoring segments… {i}/{total}", pct=pct)
                    try:
                        if req.fast_frames:
                            fdir = os.path.join(temp_dir, f"frames_{i}")
                            os.makedirs(fdir, exist_ok=True)
                            frames = hc.extract_frames(video_path, start_time, end_time, fdir)
                            v_score, _ = (hc.score_segment_frames(processor, model, frames, criteria)
                                          if frames else (0.0, ""))
                        else:
                            seg_path = os.path.join(temp_dir, f"seg_{i}.mp4")
                            hc.cut_segment(video_path, start_time, end_time - start_time, seg_path)
                            v_score, _ = hc.score_segment_file(processor, model, seg_path, criteria)
                    except Exception as seg_err:
                        logger.warning("Segment %d failed: %s", i, seg_err)
                        continue

                    a_score = (hc.audio_score_for_window(loudness, start_time, end_time, lf, lc)
                               if req.use_audio else 0.0)
                    s_score = (hc.speech_score_for_window(transcript, keywords, start_time, end_time)[0]
                               if req.use_speech else 0.0)

                    ev = req.w_visual
                    ea = req.w_audio if req.use_audio else 0.0
                    es = req.w_speech if req.use_speech else 0.0
                    wsum = ev + ea + es
                    fused = (ev * v_score + ea * a_score + es * s_score) / max(wsum, 1e-6)
                    scored_windows.append((start_time, end_time, fused))

            if not scored_windows:
                _emit(q, "error", message="Couldn't score any segments — the decoder may have failed.")
                return

            # selection
            if req.mode == "best_n":
                kept = hc.select_topk_count(scored_windows, int(req.n_moments))
                sel_label = f"top {int(req.n_moments)} moments"
            elif req.mode == "best_seconds":
                kept = hc.select_topk_seconds(scored_windows, float(req.n_seconds))
                sel_label = f"best {int(req.n_seconds)}s"
            else:
                kept = hc.select_by_threshold(scored_windows, req.threshold)
                sel_label = f"score ≥ {req.threshold:.1f}"

            merged = hc.merge_adjacent_segments([(s, e) for s, e, _ in kept])
            padded = hc.apply_padding(merged, duration, req.padding)
            scored_merged = []
            for s, e in padded:
                ov = [sc for ws, we, sc in scored_windows if we > s and ws < e]
                scored_merged.append((s, e, max(ov) if ov else 0.0))
            capped = hc.cap_segments_by_ratio(scored_merged, duration)
            final_segments = hc.merge_adjacent_segments([(s, e) for s, e, *_ in capped])
            kept_ranges = hc.merge_adjacent_segments([(s, e) for s, e, *_ in capped])

            export_segments = []
            for s, e in final_segments:
                ov = [sc for ws, we, sc in scored_windows if we > s and ws < e]
                export_segments.append((s, e, max(ov) if ov else 0.0, sel_label))

            final_kept = sum(e - s for s, e in final_segments)
            final_pct = (final_kept / duration) * 100 if duration > 0 else 0.0

            _emit(q, "progress", message="Cutting the reel…", pct=84)
            base = f"highlights_{vhash[:10]}"
            reel_path = os.path.join(hc.OUTPUT_DIR, f"{base}.mp4")
            hc.concatenate_scenes(video_path, final_segments, reel_path, crossfade=req.crossfade)

            _emit(q, "progress", message="Writing chapters & contact sheet…", pct=94)
            meta = {"source_duration_s": round(duration, 2), "selection": sel_label,
                    "kept_percent": round(final_pct, 1), "segments": len(final_segments)}
            exports = hc.write_exports(base, export_segments, meta)
            sheet_path = os.path.join(hc.OUTPUT_DIR, f"{base}_sheet.jpg")
            sheet = hc.build_contact_sheet(video_path, final_segments, sheet_path)

            signals = ["visual"]
            if req.use_audio and loudness:
                signals.append("audio")
            if req.use_speech and transcript:
                signals.append("speech")

            result = {
                "reel_url": f"/files/{os.path.basename(reel_path)}",
                "json_url": f"/files/{os.path.basename(exports['json'])}",
                "vtt_url": f"/files/{os.path.basename(exports['vtt'])}",
                "sheet_url": f"/files/{os.path.basename(sheet)}" if sheet else None,
                "windows": hc.windows_to_payload(scored_windows, kept_ranges),
                "duration": round(duration, 2),
                "threshold": req.threshold,
                "kept_percent": round(final_pct, 1),
                "segment_count": len(final_segments),
                "selection": sel_label,
                "signals": signals,
                "criteria": criteria,
                "criteria_source": criteria_source,
                "summary": video_desc,
            }
            _emit(q, "done", **result)

        finally:
            hc._INFERENCE_LOCK.release()
            try:
                import torch

                if hc.DEVICE == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                pass

    except Exception as e:
        logger.exception("Processing failed")
        _emit(q, "error", message=f"Something went wrong: {e}")
    finally:
        _emit(q, "_close")


def _run_ask(job_id: str, q: queue.Queue, req: AskRequest):
    video_path = _VIDEOS.get(req.video_id)
    try:
        if not video_path or not os.path.exists(video_path):
            _emit(q, "error", message="Video not found. Re-upload and try again.")
            return
        if not req.question.strip():
            _emit(q, "error", message="Type a question to ask.")
            return
        if not hc._INFERENCE_LOCK.acquire(blocking=False):
            _emit(q, "error", message="The model is busy generating highlights. Try again shortly.")
            return
        try:
            duration = hc.get_video_duration_seconds(video_path)
            if duration > hc.MAX_VIDEO_SECONDS:
                _emit(q, "error", message="That video is over 30 minutes. Trim it and try again.")
                return
            vhash = hc.video_hash(video_path)
            with hc._VIDEO_CACHE_LOCK:
                cache = hc._VIDEO_CACHE.setdefault(vhash, {})
            processor, model = hc.get_or_load_model_and_processor(hc.MODEL_NAME)

            transcript = []
            speech_mode = req.use_speech or hc.is_speech_focused_question(req.question)
            if speech_mode:
                _emit(q, "progress", message="Transcribing speech…", pct=5)
                if "transcript" not in cache:
                    cache["transcript"] = hc.transcribe_video(video_path)
                transcript = cache["transcript"]

            if speech_mode and transcript:
                _emit(q, "progress", message="Answering from speech transcript…", pct=35)
                final = hc.answer_from_transcript(processor, model, req.question.strip(), transcript)
                _emit(q, "answer", text=final)
                return

            windows = hc.build_windows(duration, req.segment_length, overlap=False)
            total = max(1, len(windows))
            segment_answers = {}
            with tempfile.TemporaryDirectory(prefix="qa_") as temp_dir:
                for i, (start_time, end_time) in enumerate(windows):
                    seg_path = os.path.join(temp_dir, f"qa_{i}.mp4")
                    try:
                        hc.cut_segment(video_path, start_time, end_time - start_time, seg_path)
                        ans = hc.answer_segment_question(processor, model, seg_path, req.question.strip())
                    except Exception as seg_err:
                        logger.warning("Q&A segment %d failed: %s", i, seg_err)
                        ans = "(segment could not be analyzed)"
                    segment_answers[(start_time, end_time)] = ans
                    _emit(q, "progress", message=f"Reading segment {i + 1}/{total}…",
                          pct=int((i + 1) / total * 90))

            final = hc.summarize_answers(processor, model, req.question.strip(), segment_answers, transcript)
            _emit(q, "answer", text=final)
        finally:
            hc._INFERENCE_LOCK.release()
            try:
                import torch

                if hc.DEVICE == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                pass
    except Exception as e:
        logger.exception("Q&A failed")
        _emit(q, "error", message=f"Something went wrong: {e}")
    finally:
        _emit(q, "_close")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    video_id = uuid.uuid4().hex
    ext = os.path.splitext(file.filename or "")[1] or ".mp4"
    dest = os.path.join(UPLOAD_DIR, f"{video_id}{ext}")
    with open(dest, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    _VIDEOS[video_id] = dest
    try:
        dur = hc.get_video_duration_seconds(dest)
    except Exception:
        dur = None
    return {"video_id": video_id, "duration": dur, "video_url": f"/api/video/{video_id}"}


@app.post("/api/process")
def process(req: ProcessRequest):
    if req.video_id not in _VIDEOS:
        raise HTTPException(status_code=404, detail="Unknown video_id")
    job_id, q = _new_job()
    t = threading.Thread(target=_run_process, args=(job_id, q, req), daemon=True)
    with _JOBS_LOCK:
        _JOBS[job_id]["thread"] = t
    t.start()
    return {"job_id": job_id}


@app.post("/api/ask")
def ask(req: AskRequest):
    if req.video_id not in _VIDEOS:
        raise HTTPException(status_code=404, detail="Unknown video_id")
    job_id, q = _new_job()
    t = threading.Thread(target=_run_ask, args=(job_id, q, req), daemon=True)
    with _JOBS_LOCK:
        _JOBS[job_id]["thread"] = t
    t.start()
    return {"job_id": job_id}


@app.get("/api/events/{job_id}")
def events(job_id: str):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    q = job["queue"]

    def stream():
        while True:
            item = q.get()
            if item["event"] == "_close":
                with _JOBS_LOCK:
                    _JOBS.pop(job_id, None)
                break
            payload = json.dumps(item["data"])
            yield f"event: {item['event']}\ndata: {payload}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/video/{video_id}")
def serve_video(video_id: str):
    path = _VIDEOS.get(video_id)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Unknown video_id")
    # FileResponse honors Range requests, so the player can seek/scrub.
    return FileResponse(path, media_type="video/mp4")


@app.get("/files/{name}")
def serve_file(name: str):
    # Prevent path traversal; only serve from OUTPUT_DIR by basename.
    safe = os.path.basename(name)
    path = os.path.join(hc.OUTPUT_DIR, safe)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path)


@app.get("/", response_class=HTMLResponse)
def index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path) as f:
        return HTMLResponse(f.read())


# Mount remaining static assets (css/js if split out later).
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8019"))
    uvicorn.run(app, host="0.0.0.0", port=port)