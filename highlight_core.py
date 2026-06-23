#!/usr/bin/env python3
"""SmolVLM2 video highlight generator — analysis core (framework-agnostic).

Imported by app.py (FastAPI). Contains no web-framework code.

Signals fused per segment:
  - Visual relevance (SmolVLM2, 0-10 score)
  - Audio loudness peaks (ffmpeg ebur128)
  - Speech relevance (Whisper transcript keyword match) [optional]

Features:
  - Single-pass scored scanning with tunable per-signal weights
  - Scene-cut-aware segment boundaries (ffmpeg scene detection)
  - Threshold mode and Top-K mode (best N moments / best N seconds)
  - Optional sliding-window overlap; pre/post-roll padding; crossfades
  - Natural-language highlight prompt (overrides auto-generated archetypes)
  - Score-over-time timeline, per-segment results table, contact sheet
  - JSON + WebVTT chapter export
  - Direct-frame scoring path (skips intermediate MP4s) for speed
  - Per-video caching keyed by content hash
"""

import hashlib
import json
import logging
import os
import re
import subprocess
import tempfile
import threading

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.video_utils import load_video

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "HuggingFaceTB/SmolVLM-500M-Instruct")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
MAX_HIGHLIGHT_RATIO = 0.6
MAX_VIDEO_SECONDS = 1800  # 30 minutes
DEFAULT_SCORE_THRESHOLD = 6.0
SCENE_THRESHOLD = 0.4  # ffmpeg scene-change sensitivity (0-1, higher = fewer cuts)
FRAMES_PER_SEGMENT = 8  # frames sampled for direct-frame scoring
QA_SEGMENT_MAX_TOKENS = 420
QA_SUMMARY_MAX_TOKENS = 900
QA_TRANSCRIPT_MAX_TOKENS = 900

_MODEL_CACHE = None
_MODEL_CACHE_LOCK = threading.Lock()
_WHISPER_CACHE = None
_WHISPER_CACHE_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()
_VIDEO_CACHE = {}
_VIDEO_CACHE_LOCK = threading.Lock()

OUTPUT_DIR = os.getenv("HIGHLIGHT_OUTPUT_DIR", tempfile.gettempdir())


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _generate(processor, model, messages, max_new_tokens, do_sample=False, temperature=None, device=DEVICE):
    """Run a chat-template generation and return only the model's new text."""
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device, dtype=DTYPE)

    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    if do_sample and temperature is not None:
        gen_kwargs["temperature"] = temperature

    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    return processor.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()


def _parse_score(text: str, default: float = 0.0) -> tuple:
    """Extract a 0-10 score and a short reason from a model reply."""
    m = re.search(r"(?:score\s*[:=]?\s*)?(\d{1,2}(?:\.\d+)?)\s*(?:/\s*10)?", text, re.IGNORECASE)
    if not m:
        return default, text.strip()
    try:
        score = float(m.group(1))
    except ValueError:
        return default, text.strip()
    score = max(0.0, min(10.0, score))
    reason = text[m.end():].lstrip(" :.-\n") or text.strip()
    return score, reason


# ---------------------------------------------------------------------------
# Window construction
# ---------------------------------------------------------------------------


def build_windows(duration: float, segment_length: float, overlap: bool):
    """Return list of (start, end) windows, optionally with 50% overlap."""
    stride = segment_length / 2 if overlap else segment_length
    start = 0.0
    windows = []
    while start < duration:
        end = min(start + segment_length, duration)
        windows.append((start, end))
        if end >= duration:
            break
        start += stride
    return windows


def detect_scene_times(video_path: str) -> list:
    """Return timestamps (s) of detected scene cuts via ffmpeg, or [] on failure."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-i", video_path, "-filter:v",
             f"select='gt(scene,{SCENE_THRESHOLD})',showinfo", "-f", "null", "-"],
            capture_output=True, text=True,
        )
        times = []
        for line in proc.stderr.splitlines():
            m = re.search(r"pts_time:(\d+\.?\d*)", line)
            if m:
                times.append(float(m.group(1)))
        return sorted(set(times))
    except Exception as e:
        logger.warning("Scene detection failed: %s", e)
        return []


def snap_windows_to_scenes(duration: float, segment_length: float, scene_times: list):
    """Build windows whose boundaries prefer scene cuts; fall back to grid."""
    if not scene_times:
        return build_windows(duration, segment_length, overlap=False)
    cuts = [0.0] + sorted(t for t in scene_times if 0 < t < duration) + [duration]
    windows = []
    i = 0
    while i < len(cuts) - 1:
        start, end = cuts[i], cuts[i + 1]
        j = i + 1
        while (end - start) < segment_length and j < len(cuts) - 1:
            j += 1
            end = cuts[j]
        if (end - start) > segment_length * 2:
            t = start
            while t < end:
                windows.append((t, min(t + segment_length, end)))
                t += segment_length
        else:
            windows.append((start, end))
        i = j
    return windows


# ---------------------------------------------------------------------------
# Segment / ratio utilities
# ---------------------------------------------------------------------------


def merge_adjacent_segments(segments: list, gap_tolerance: float = 0.5) -> list:
    """Sort and merge overlapping or contiguous segments to avoid seams."""
    if not segments:
        return segments
    ordered = sorted(segments, key=lambda s: s[0])
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + gap_tolerance:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def apply_padding(segments: list, duration: float, pad: float) -> list:
    """Add pre/post-roll to each segment, clamped to [0, duration]."""
    if pad <= 0:
        return segments
    return [(max(0.0, s - pad), min(duration, e + pad)) for s, e in segments]


def select_by_threshold(scored_windows: list, threshold: float) -> list:
    kept = [(s, e, sc) for s, e, sc in scored_windows if sc >= threshold]
    if not kept:
        kept = [max(scored_windows, key=lambda w: w[2])]
    return kept


def select_topk_count(scored_windows: list, k: int) -> list:
    ranked = sorted(scored_windows, key=lambda w: w[2], reverse=True)[:max(1, k)]
    return sorted(ranked, key=lambda w: w[0])


def select_topk_seconds(scored_windows: list, budget: float) -> list:
    ranked = sorted(scored_windows, key=lambda w: w[2], reverse=True)
    out, used = [], 0.0
    for s, e, sc in ranked:
        out.append((s, e, sc))
        used += e - s
        if used >= budget:
            break
    return sorted(out, key=lambda w: w[0])


def cap_segments_by_ratio(segments: list, duration: float, max_ratio: float = MAX_HIGHLIGHT_RATIO) -> list:
    """Cap total retained highlight duration. Keeps highest-scoring if scored."""
    if not segments or duration <= 0:
        return segments
    total_kept = sum(e - s for s, e, *_ in segments)
    budget = duration * max_ratio
    if total_kept <= budget:
        return segments

    scored = all(len(seg) >= 3 for seg in segments)
    if scored:
        ranked = sorted(segments, key=lambda s: s[2], reverse=True)
        selected, used = [], 0.0
        for seg in ranked:
            length = seg[1] - seg[0]
            if used + length > budget and selected:
                continue
            selected.append(seg)
            used += length
        return sorted(selected, key=lambda s: s[0])

    avg_len = total_kept / len(segments)
    keep_count = max(1, int(budget / max(avg_len, 1e-6)))
    if keep_count >= len(segments):
        return segments
    if keep_count == 1:
        return [segments[len(segments) // 2]]
    last_idx = len(segments) - 1
    selected, seen = [], set()
    for i in range(keep_count):
        idx = round(i * last_idx / (keep_count - 1))
        seg = segments[idx]
        if seg not in seen:
            selected.append(seg)
            seen.add(seg)
    return selected


def normalize_video_input(video):
    if isinstance(video, str):
        return video
    if isinstance(video, dict):
        return video.get("path") or video.get("video")
    return None


def video_hash(video_path: str) -> str:
    h = hashlib.sha256()
    size = os.path.getsize(video_path)
    h.update(str(size).encode())
    with open(video_path, "rb") as f:
        h.update(f.read(1 << 20))
        if size > (2 << 20):
            f.seek(-(1 << 20), os.SEEK_END)
            h.update(f.read(1 << 20))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def pick_attention_backend(device: str) -> str:
    if device != "cuda":
        return "eager"
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:
        logger.warning("flash_attn not installed; falling back to sdpa attention.")
        return "sdpa"


def pick_video_backend(preferred: str = "decord") -> str:
    """Pick an available backend, honoring a preferred backend when possible."""
    order = [preferred] + [b for b in ("decord", "pyav") if b != preferred]
    for backend in order:
        module_name = "decord" if backend == "decord" else "av"
        try:
            __import__(module_name)
            return backend
        except Exception:
            continue
    raise RuntimeError("No video decoder backend available. Install decord or av.")


def configure_video_fetch_backend(proc) -> str:
    backend = pick_video_backend("decord")
    fallback_backend = pick_video_backend("pyav") if backend == "decord" else pick_video_backend("decord")
    if fallback_backend == backend:
        fallback_backend = None

    def _fetch(video_url_or_urls, sample_indices_fn=None):
        if isinstance(video_url_or_urls, list):
            return list(zip(*[_fetch(x, sample_indices_fn=sample_indices_fn) for x in video_url_or_urls]))
        try:
            return load_video(video_url_or_urls, backend=backend, sample_indices_fn=sample_indices_fn)
        except Exception as primary_err:
            if not fallback_backend:
                raise
            logger.warning(
                "Primary video backend '%s' failed for %s (%s); retrying with '%s'.",
                backend,
                video_url_or_urls,
                primary_err,
                fallback_backend,
            )
            return load_video(video_url_or_urls, backend=fallback_backend, sample_indices_fn=sample_indices_fn)

    proc.video_processor.fetch_videos = _fetch
    return backend


# ---------------------------------------------------------------------------
# ffprobe / ffmpeg helpers
# ---------------------------------------------------------------------------


def get_video_duration_seconds(video_path: str) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
            capture_output=True, text=True, check=True,
        )
        return float(json.loads(result.stdout)["format"]["duration"])
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, ValueError) as e:
        raise RuntimeError(f"Could not read video duration: {e}") from e


def video_has_audio(video_path: str) -> bool:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=index", "-of", "csv=p=0", video_path],
        capture_output=True, text=True, check=True,
    )
    return bool(result.stdout.strip())


def has_decodable_video(video_path: str) -> bool:
    """True if ffprobe finds at least one video stream that reports frames.

    Stream-copy cuts can produce a container ffmpeg writes happily but whose
    video stream is empty/unindexed, which makes decord raise
    'cannot find video stream'. This guards against that.
    """
    try:
        if not os.path.exists(video_path) or os.path.getsize(video_path) <= 0:
            return False
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_packets", "-show_entries", "stream=nb_read_packets",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True, check=True,
        )
        out = result.stdout.strip()
        return bool(out) and out != "N/A" and int(out or 0) > 0
    except (subprocess.CalledProcessError, ValueError):
        return False


def cut_segment(input_video: str, start_time: float, segment_length: float, segment_path: str):
    """Cut a short segment for analysis.

    Tries fast stream-copy, but verifies the result actually contains a
    decodable video stream (decord rejects copies that land on a bad boundary).
    Falls back to a clean re-encode otherwise.
    """
    base = ["ffmpeg", "-y", "-ss", str(start_time), "-i", input_video, "-t", str(segment_length)]
    try:
        subprocess.run(
            base + ["-map", "0:v:0?", "-map", "0:a:0?",
                    "-avoid_negative_ts", "make_zero", "-c", "copy",
                    "-movflags", "+faststart", segment_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if os.path.getsize(segment_path) > 0 and has_decodable_video(segment_path):
            return
    except subprocess.CalledProcessError:
        pass

    # Re-encode: frame-accurate, always decodable. -ss after -i for accuracy.
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_video, "-ss", str(start_time), "-t", str(segment_length),
         "-map", "0:v:0?", "-map", "0:a:0?",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", segment_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def extract_frames(input_video: str, start: float, end: float, out_dir: str, n: int = FRAMES_PER_SEGMENT) -> list:
    """Sample n evenly-spaced JPEG frames from [start,end]. Returns sorted paths."""
    length = max(0.1, end - start)
    fps = max(1.0, n / length)
    pattern = os.path.join(out_dir, "f_%04d.jpg")
    cmd = ["ffmpeg", "-y", "-ss", str(start), "-i", input_video, "-t", str(length),
           "-vf", f"fps={fps},scale=384:-2", "-frames:v", str(n), "-q:v", "4", pattern]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return sorted(
        os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.startswith("f_") and f.endswith(".jpg")
    )


def compute_loudness_timeline(video_path: str) -> list:
    if not video_has_audio(video_path):
        return []
    proc = subprocess.run(
        ["ffmpeg", "-i", video_path, "-af", "ebur128=metadata=1", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    timeline = []
    cur_t = None
    for line in proc.stderr.splitlines():
        tm = re.search(r"t:\s*(\d+\.?\d*)", line)
        if tm:
            cur_t = float(tm.group(1))
        lm = re.search(r"M:\s*(-?\d+\.?\d*)", line)
        if lm and cur_t is not None:
            val = float(lm.group(1))
            if val > -120:
                timeline.append((cur_t, val))
    return timeline


def audio_score_for_window(loudness: list, start: float, end: float, floor: float, ceil: float) -> float:
    if not loudness or ceil <= floor:
        return 0.0
    vals = [v for t, v in loudness if start <= t < end]
    if not vals:
        return 0.0
    peak = max(vals)
    return max(0.0, min(10.0, (peak - floor) / (ceil - floor) * 10.0))


def concatenate_scenes(video_path: str, scene_times: list, output_path: str, crossfade: float = 0.0):
    if not scene_times:
        logger.warning("No scenes to concatenate, skipping.")
        return
    has_audio = video_has_audio(video_path)
    if crossfade > 0 and len(scene_times) > 1:
        _concat_with_crossfade(video_path, scene_times, output_path, has_audio, crossfade)
        return

    filter_parts, concat_inputs = [], []
    for i, (s, e) in enumerate(scene_times):
        filter_parts.append(f"[0:v]trim=start={s}:end={e},setpts=PTS-STARTPTS[v{i}];")
        if has_audio:
            filter_parts.append(f"[0:a]atrim=start={s}:end={e},asetpts=PTS-STARTPTS[a{i}];")
            concat_inputs.append(f"[v{i}][a{i}]")
        else:
            concat_inputs.append(f"[v{i}]")

    n = len(scene_times)
    if has_audio:
        fc = "".join(filter_parts) + f"{''.join(concat_inputs)}concat=n={n}:v=1:a=1[outv][outa]"
        cmd = ["ffmpeg", "-y", "-i", video_path, "-filter_complex", fc,
               "-map", "[outv]", "-map", "[outa]", "-c:v", "libx264", "-c:a", "aac", output_path]
    else:
        fc = "".join(filter_parts) + f"{''.join(concat_inputs)}concat=n={n}:v=1:a=0[outv]"
        cmd = ["ffmpeg", "-y", "-i", video_path, "-filter_complex", fc,
               "-map", "[outv]", "-c:v", "libx264", "-an", output_path]
    logger.info("Running ffmpeg concat over %d scenes", n)
    subprocess.run(cmd, check=True)


def _concat_with_crossfade(video_path, scene_times, output_path, has_audio, xf):
    with tempfile.TemporaryDirectory(prefix="xfade_") as td:
        clips = []
        for i, (s, e) in enumerate(scene_times):
            clip = os.path.join(td, f"clip_{i}.mp4")
            cmd = ["ffmpeg", "-y", "-ss", str(s), "-i", video_path, "-t", str(e - s),
                   "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p"]
            cmd += (["-c:a", "aac"] if has_audio else ["-an"])
            cmd.append(clip)
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            clips.append((clip, e - s))

        inputs = []
        for clip, _ in clips:
            inputs += ["-i", clip]

        v_label, a_label = "[0:v]", "[0:a]"
        filt = []
        offset = clips[0][1] - xf
        for i in range(1, len(clips)):
            out_v = f"[vx{i}]"
            filt.append(f"{v_label}[{i}:v]xfade=transition=fade:duration={xf}:offset={offset:.3f}{out_v}")
            v_label = out_v
            if has_audio:
                out_a = f"[ax{i}]"
                filt.append(f"{a_label}[{i}:a]acrossfade=d={xf}{out_a}")
                a_label = out_a
            offset += clips[i][1] - xf

        fc = ";".join(filt)
        cmd = ["ffmpeg", "-y"] + inputs + ["-filter_complex", fc, "-map", v_label]
        if has_audio:
            cmd += ["-map", a_label, "-c:a", "aac"]
        cmd += ["-c:v", "libx264", output_path]
        logger.info("Running ffmpeg crossfade concat over %d clips", len(clips))
        subprocess.run(cmd, check=True)


def build_contact_sheet(video_path: str, segments: list, output_path: str, cols: int = 4):
    """One representative frame per segment, tiled into a single image."""
    if not segments:
        return None
    with tempfile.TemporaryDirectory(prefix="contact_") as td:
        thumbs = []
        for i, (s, e) in enumerate(segments):
            mid = (s + e) / 2
            tp = os.path.join(td, f"t_{i:03d}.jpg")
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(mid), "-i", video_path, "-frames:v", "1",
                 "-vf", "scale=320:-2", "-q:v", "3", tp],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if os.path.exists(tp):
                thumbs.append(tp)
        if not thumbs:
            return None
        rows = (len(thumbs) + cols - 1) // cols
        # The `tile` filter consumes a single image sequence, so renumber the
        # thumbnails into a contiguous %03d sequence and tile them in one pass.
        seq_dir = os.path.join(td, "seq")
        os.makedirs(seq_dir, exist_ok=True)
        for idx, t in enumerate(thumbs):
            os.rename(t, os.path.join(seq_dir, f"s_{idx:03d}.jpg"))
        tile = f"tile={cols}x{rows}:padding=6:margin=6:color=0x111418"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-framerate", "1", "-i", os.path.join(seq_dir, "s_%03d.jpg"),
                 "-vf", tile, "-frames:v", "1", output_path],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return output_path if os.path.exists(output_path) else None
        except subprocess.CalledProcessError as e:
            logger.warning("Contact sheet build failed: %s", e)
            return None


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _ts(x: float) -> str:
    h = int(x // 3600)
    m = int((x % 3600) // 60)
    s = x % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def build_vtt(segments: list) -> str:
    """segments: list of (start, end, score, reason)."""
    lines = ["WEBVTT", ""]
    for i, seg in enumerate(segments, 1):
        s, e = seg[0], seg[1]
        reason = seg[3] if len(seg) > 3 and seg[3] else ""
        lines.append(str(i))
        lines.append(f"{_ts(s)} --> {_ts(e)}")
        lines.append(f"Highlight {i}" + (f" — {reason}" if reason else ""))
        lines.append("")
    return "\n".join(lines)


def build_json(segments: list, meta: dict) -> str:
    payload = {
        "meta": meta,
        "highlights": [
            {"start": round(seg[0], 3), "end": round(seg[1], 3),
             "score": round(seg[2], 2) if len(seg) > 2 else None,
             "reason": seg[3] if len(seg) > 3 else None}
            for seg in segments
        ],
    }
    return json.dumps(payload, indent=2)


def write_exports(base_name: str, segments: list, meta: dict) -> dict:
    """Write JSON + VTT to OUTPUT_DIR. Returns {'json': path, 'vtt': path}."""
    paths = {}
    json_path = os.path.join(OUTPUT_DIR, f"{base_name}.json")
    vtt_path = os.path.join(OUTPUT_DIR, f"{base_name}.vtt")
    with open(json_path, "w") as f:
        f.write(build_json(segments, meta))
    with open(vtt_path, "w") as f:
        f.write(build_vtt(segments))
    paths["json"] = json_path
    paths["vtt"] = vtt_path
    return paths


# ---------------------------------------------------------------------------
# Whisper (optional speech signal)
# ---------------------------------------------------------------------------


def get_or_load_whisper():
    global _WHISPER_CACHE
    if _WHISPER_CACHE is None:
        with _WHISPER_CACHE_LOCK:
            if _WHISPER_CACHE is None:
                import whisper

                logger.info("Loading Whisper model: %s", WHISPER_MODEL)
                _WHISPER_CACHE = whisper.load_model(WHISPER_MODEL, device=DEVICE)
    return _WHISPER_CACHE


def transcribe_video(video_path: str) -> list:
    if not video_has_audio(video_path):
        return []
    try:
        model = get_or_load_whisper()
        result = model.transcribe(video_path, verbose=False)
        return [{"start": s["start"], "end": s["end"], "text": s["text"].strip()}
                for s in result.get("segments", [])]
    except Exception as e:
        logger.warning("Whisper transcription failed: %s", e)
        return []


def speech_score_for_window(transcript: list, keywords: list, start: float, end: float) -> tuple:
    if not transcript or not keywords:
        return 0.0, ""
    text = " ".join(
        seg["text"] for seg in transcript
        if seg["end"] > start and seg["start"] < end
    ).lower()
    if not text:
        return 0.0, ""
    hits = sum(1 for kw in keywords if kw in text)
    score = min(10.0, hits * 10.0 / max(1, len(keywords)) + (3.0 if hits else 0.0))
    return min(10.0, score), text[:200]


def extract_keywords(highlight_prompt: str) -> list:
    stop = {"the", "a", "an", "of", "in", "on", "and", "or", "to", "with", "that",
            "moments", "moment", "where", "find", "show", "me", "video", "highlights",
            "highlight", "when", "someone", "is", "are", "for", "any", "this", "they"}
    words = re.findall(r"[a-z']+", highlight_prompt.lower())
    return [w for w in words if len(w) > 2 and w not in stop]


def is_speech_focused_question(question: str) -> bool:
    """Heuristic: detect questions that should prefer transcript evidence."""
    q = (question or "").strip().lower()
    if not q:
        return False
    markers = [
        "what did",
        "what was said",
        "say",
        "said",
        "interview",
        "post fight",
        "post-fight",
        "quote",
        "mention",
        "talk about",
        "asked",
        "answer",
        "statement",
        "speech",
        "comment",
    ]
    return any(m in q for m in markers)


def answer_from_transcript(processor, model, question: str, transcript: list) -> str:
    """Answer a question primarily from transcript text with timestamps."""
    if not transcript:
        return "No speech transcript is available for this video."

    lines = []
    for seg in transcript:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        lines.append(f"[{start:.1f}s-{end:.1f}s] {text}")

    transcript_block = "\n".join(lines)
    if not transcript_block:
        return "Transcript was generated, but no readable speech text was found."

    transcript_block = transcript_block[:12000]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": (
            "You answer questions about spoken content in videos using transcript evidence. "
            "Quote key lines when relevant. If the transcript does not contain the answer, "
            "say that clearly and do not invent details."
        )}]},
        {"role": "user", "content": [{"type": "text", "text": (
            f"Question: {question}\n\n"
            f"Transcript:\n{transcript_block}\n\n"
            "Answer using only transcript evidence."
        )}]},
    ]
    return _generate(processor, model, messages, max_new_tokens=QA_TRANSCRIPT_MAX_TOKENS)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_and_processor(model_path: str, device: str = DEVICE, dtype=DTYPE):
    attn = pick_attention_backend(device)
    processor = AutoProcessor.from_pretrained(model_path)
    backend = configure_video_fetch_backend(processor)
    logger.info("Using video backend: %s | attention: %s", backend, attn)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=dtype, attn_implementation=attn, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return processor, model


def get_or_load_model_and_processor(model_path: str):
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        with _MODEL_CACHE_LOCK:
            if _MODEL_CACHE is None:
                _MODEL_CACHE = load_model_and_processor(model_path)
    return _MODEL_CACHE


# ---------------------------------------------------------------------------
# Model-driven analysis steps
# ---------------------------------------------------------------------------


def analyze_video_content(processor, model, video_path: str) -> str:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": (
            "You are a helpful assistant that can understand videos. "
            "Describe what type of video this is and what is happening in it."
        )}]},
        {"role": "user", "content": [
            {"type": "video", "path": video_path},
            {"type": "text", "text": (
                "What type of video is this and what is happening in it? "
                "Be specific about the content type and general activities you observe."
            )},
        ]},
    ]
    try:
        return _generate(processor, model, messages, max_new_tokens=512)
    except Exception as e:
        logger.error("Video analysis failed for %s: %s", video_path, e)
        raise


def determine_highlights(processor, model, video_description: str) -> str:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": (
            "You are a highlight editor. List archetypal dramatic moments that would make compelling "
            "highlights if they appear in the video. Each moment should be specific enough to be "
            "recognizable but generic enough to potentially exist in other videos of this type."
        )}]},
        {"role": "user", "content": [{"type": "text", "text": (
            f"Here is a description of a video:\n\n{video_description}\n\n"
            "List potential highlight moments to look for in this video:"
        )}]},
    ]
    return _generate(processor, model, messages, max_new_tokens=256)


def _score_messages(visual_content, highlight_types):
    return [
        {"role": "system", "content": [{"type": "text", "text": (
            "You are a video highlight analyzer. Rate how strongly the footage matches the "
            "described highlight criteria, focusing on skill, emotion, personality, or tension. "
            "Respond with a single integer score from 0 to 10 on the first line, then one short "
            "sentence of justification. 0 means nothing relevant; 10 means a perfect highlight."
        )}]},
        {"role": "user", "content": visual_content + [
            {"type": "text", "text": (
                f"Highlight criteria:\n{highlight_types}\n\n"
                "Score this footage from 0 to 10 and justify briefly."
            )},
        ]},
    ]


def score_segment_file(processor, model, video_path: str, highlight_types: str) -> tuple:
    """Score a segment from an MP4 file path.

    Falls back to frame-based scoring if the video decoder can't read the clip.
    """
    if not has_decodable_video(video_path):
        logger.warning("Segment is not decodable (%s); using frame fallback.", video_path)
        with tempfile.TemporaryDirectory(prefix="fallback_frames_") as fdir:
            try:
                dur = get_video_duration_seconds(video_path)
            except Exception:
                dur = 0.0
            frames = extract_frames(video_path, 0.0, dur or 1.0, fdir)
            if not frames:
                return 0.0, "(segment could not be decoded)"
            return score_segment_frames(processor, model, frames, highlight_types)

    messages = _score_messages([{"type": "video", "path": video_path}], highlight_types)
    try:
        return _parse_score(_generate(processor, model, messages, max_new_tokens=80))
    except Exception as e:
        logger.warning("Video scoring failed for %s (%s); falling back to frames.", video_path, e)
        with tempfile.TemporaryDirectory(prefix="fallback_frames_") as fdir:
            dur = end = 0.0
            try:
                dur = get_video_duration_seconds(video_path)
            except Exception:
                dur = 0.0
            frames = extract_frames(video_path, 0.0, dur or 1.0, fdir)
            if not frames:
                return 0.0, "(segment could not be decoded)"
            return score_segment_frames(processor, model, frames, highlight_types)


def score_segment_frames(processor, model, frame_paths: list, highlight_types: str) -> tuple:
    """Score a segment from sampled image frames (no intermediate MP4)."""
    visual = [{"type": "image", "path": p} for p in frame_paths]
    messages = _score_messages(visual, highlight_types)
    return _parse_score(_generate(processor, model, messages, max_new_tokens=80))


def answer_segment_question(processor, model, segment_path: str, question: str) -> str:
    sys_text = (
        "You are a grounded video analyst. A question has been asked about a longer video. "
        "You are seeing one segment of it. Extract any visible facts or observations "
        "that could help answer the question. Do NOT try to give a final answer — "
        "just list what you observe. If nothing relevant is visible, say so briefly. "
        "Do not hallucinate details."
    )
    user_text = (
        f"Question to help with: {question}\n\n"
        "Describe what is visible in this segment that is relevant to the question above."
    )

    def _msgs(visual):
        return [
            {"role": "system", "content": [{"type": "text", "text": sys_text}]},
            {"role": "user", "content": visual + [{"type": "text", "text": user_text}]},
        ]

    if not has_decodable_video(segment_path):
        logger.warning("Segment Q&A input is not decodable (%s); using frame fallback.", segment_path)
        with tempfile.TemporaryDirectory(prefix="qa_frames_") as fdir:
            try:
                dur = get_video_duration_seconds(segment_path)
            except Exception:
                dur = 1.0
            frames = extract_frames(segment_path, 0.0, dur, fdir)
            if not frames:
                return "(segment could not be decoded)"
            visual = [{"type": "image", "path": p} for p in frames]
            return _generate(processor, model, _msgs(visual), max_new_tokens=200)

    try:
        return _generate(processor, model, _msgs([{"type": "video", "path": segment_path}]),
                         max_new_tokens=QA_SEGMENT_MAX_TOKENS)
    except Exception as e:
        logger.warning("Video Q&A failed for %s (%s); falling back to frames.", segment_path, e)
        with tempfile.TemporaryDirectory(prefix="qa_frames_") as fdir:
            try:
                dur = get_video_duration_seconds(segment_path)
            except Exception:
                dur = 1.0
            frames = extract_frames(segment_path, 0.0, dur, fdir)
            if not frames:
                return "(segment could not be decoded)"
            visual = [{"type": "image", "path": p} for p in frames]
            return _generate(processor, model, _msgs(visual), max_new_tokens=QA_SEGMENT_MAX_TOKENS)


def summarize_answers(processor, model, question: str, segment_answers: dict, transcript: list = None) -> str:
    combined = "\n\n".join(
        f"[Segment {i + 1} ({start:.0f}s-{end:.0f}s)]: {ans}"
        for i, ((start, end), ans) in enumerate(segment_answers.items())
    )
    speech_block = ""
    if transcript:
        joined = " ".join(s["text"] for s in transcript)[:2000]
        speech_block = f"\n\nSpoken audio transcript (may help):\n{joined}"
    messages = [
        {"role": "system", "content": [{"type": "text", "text": (
            "You are a helpful assistant that synthesizes answers from multiple "
            "video segments into a single coherent response."
        )}]},
        {"role": "user", "content": [{"type": "text", "text": (
            f"Question: {question}\n\nSegment-by-segment answers:\n{combined}{speech_block}\n\n"
            "Summarize these into a single final answer."
        )}]},
    ]
    return _generate(processor, model, messages, max_new_tokens=QA_SUMMARY_MAX_TOKENS)


# ---------------------------------------------------------------------------
# Score timeline + results table (HTML)
# ---------------------------------------------------------------------------


def windows_to_payload(scored_windows: list, kept_ranges: list) -> list:
    """Serialize scored windows + kept flag for the frontend to render."""
    out = []
    for s, e, sc in scored_windows:
        in_kept = any(ks <= (s + e) / 2 <= ke for ks, ke in kept_ranges)
        out.append({"start": round(s, 3), "end": round(e, 3),
                    "score": round(sc, 2), "kept": in_kept})
    return out