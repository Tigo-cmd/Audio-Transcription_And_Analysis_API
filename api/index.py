#!/usr/bin/env python3
"""
DB-backed Flask backend for audio transcription + summary + QA using Groq.

Key points:
- Uses SQLAlchemy to persist Users, Jobs, and ConversationMessages.
- Jobs are owned by a user (user_id required via X-User-Id or Authorization Bearer token).
- Background worker processes transcription jobs from DB.
- Summary & QA calls use Groq chat and are persisted.
- Response shapes kept compatible with your frontend.

Environment:
- GROQ_API_KEY (required)
- DATABASE_URL (optional; default sqlite:///./jobs.db)
- JWT_SECRET (optional; if set, verifies Authorization Bearer tokens as JWTs and extracts 'sub' claim as user_id)
DB-backed Flask backend for audio transcription + summary + QA using Groq.
"""

from __future__ import annotations
import os
import io
import uuid
import time
import json
import traceback
import subprocess
from datetime import datetime, timedelta
from typing import Optional, Any, List, Dict

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.sqlite import JSON as SQLITE_JSON
from dotenv import load_dotenv
import threading
import base64

# Groq SDK
from groq import Groq

# Optional PDF parsing
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

# JWT decode if provided
try:
    import jwt
except Exception:
    jwt = None

# Load env
if "GROQ_API_KEY" not in os.environ:
    load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:////tmp/jobs.db")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", None)
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "whisper-large-v3")
JWT_SECRET = os.getenv("JWT_SECRET")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable required")

# Flask + DB
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
CORS(app, resources={r"/*": {"origins": "*"}})

db = SQLAlchemy(app)

# -------------------------
# DB models
# -------------------------
class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.String, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Job(db.Model):
    __tablename__ = "jobs"
    id = db.Column(db.String, primary_key=True)
    user_id = db.Column(db.String, db.ForeignKey("users.id"), nullable=False)
    type = db.Column(db.String, nullable=False)  # transcription|summary|qa
    status = db.Column(db.String, nullable=False, default="queued")
    meta = db.Column(SQLITE_JSON, default={})
    result = db.Column(SQLITE_JSON, default=None)
    error = db.Column(db.String, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

    messages = db.relationship("ConversationMessage", backref="job", cascade="all, delete-orphan")


class ConversationMessage(db.Model):
    __tablename__ = "conversation_messages"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    job_id = db.Column(db.String, db.ForeignKey("jobs.id"), nullable=False)
    role = db.Column(db.String, nullable=False)  # user | assistant
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


with app.app_context():
    db.create_all()

# -------------------------
# Groq client + safe wrapper
# -------------------------
def make_groq_client():
    if GROQ_BASE_URL:
        return Groq(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
    return Groq(api_key=GROQ_API_KEY)


groq_client = make_groq_client()


class APIConnectionError(Exception):
    pass


def safe_call(fn, *args, retries: int = 3, backoff: float = 1.2, **kwargs):
    last = None
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last = e
            time.sleep(backoff * (2 ** i))
    raise APIConnectionError(f"API call failed after {retries} attempts: {last}")

# -------------------------
# Universal Prompt
# -------------------------
CATEGORY_GUIDELINES = """
You are an expert call review analyst. You will receive transcribed phone calls and be asked questions about them based on one of the following categories:

1. Service Setter Booked — Determine if the caller booked an appointment with a service setter.
2. Live Conversation - Outbound — Determine if an outbound caller actually reached the intended contact (or their spokesperson/business) and classify outcome.
3. Dealership Discussion — Determine if the call included a discussion about a dealership, vehicle inventory, or related details.
4. Inbound — Determine whether the call was handled by a live, qualified employee or interactive system.
5. Reason for Outbound Call — Identify the reason a connected outbound call was placed by the caller (mainly to detect if it was sales-related or not).

Instructions:
- Use only the content from the transcript.
- Always clearly state your Yes/No or category answer first.
- If unsure or no info is available, say "Not enough information".
- Be concise and strictly follow the category definitions.
"""

DEFAULT_SYSTEM_PROMPT = CATEGORY_GUIDELINES.strip()

# -------------------------
# GroqChat helper (simple)
# -------------------------
class GroqChat:
    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT, model: str = LLM_MODEL, api_client: Optional[Groq] = None):
        self.system_prompt = system_prompt
        self.model = model
        self.client = api_client or groq_client

    def summarize(self, text: str, style: str = "short") -> str:
        prompt = f"Summarize in {style} style:\n\n{text[:20000]}"
        resp = safe_call(
            self.client.chat.completions.create,
            model=self.model,
            messages=[{"role": "system", "content": self.system_prompt},
                      {"role": "user", "content": prompt}],
            max_tokens=768,
            stream=False
        )
        return self._extract_text(resp)

    def answer_with_context(self, context: str, question: str, requirement_text: Optional[str] = None) -> str:
        parts = [
            "Answer based only on the provided call transcript and strictly follow the active category instructions.",
            "If the answer is not present in the transcript, say 'Not enough information'."
        ]
        if requirement_text:
            parts.append("Additional requirements:")
            parts.append(requirement_text)
        parts.append("Transcript context:")
        parts.append(context[:20000])
        parts.append("Question:")
        parts.append(question)
        prompt = "\n\n".join(parts)
        resp = safe_call(
            self.client.chat.completions.create,
            model=self.model,
            messages=[{"role": "system", "content": self.system_prompt},
                      {"role": "user", "content": prompt}],
            max_tokens=768,
            stream=False
        )
        return self._extract_text(resp)

    def _extract_text(self, resp: Any) -> str:
        try:
            if hasattr(resp, "choices"):
                parts = []
                for c in resp.choices:
                    if getattr(c, "message", None):
                        if isinstance(c.message, dict):
                            parts.append(c.message.get("content", "") or "")
                        else:
                            parts.append(getattr(c.message, "content", "") or "")
                    elif getattr(c, "delta", None):
                        parts.append(getattr(c.delta, "content", "") or "")
                    else:
                        parts.append(str(c))
                out = "".join(parts).strip()
                if out:
                    return out
            if isinstance(resp, dict):
                return resp.get("text") or resp.get("output_text") or str(resp)
            if hasattr(resp, "text"):
                return getattr(resp, "text")
        except Exception:
            pass
        return str(resp)


# -------------------------
# Utilities
# -------------------------
def iso_now():
    return datetime.utcnow().isoformat() + "Z"


def extract_user_id_from_request() -> Optional[str]:
    """
    Accepts either:
      - X-User-Id header (preferred for simple setups), or
      - Authorization: Bearer <token>
    If JWT_SECRET is set, the Bearer token will be decoded and 'sub' used as user id.
    Otherwise the Bearer token value is used as user id (dev mode).
    """
    uid = request.headers.get("X-User-Id")
    if uid:
        return uid
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
        if JWT_SECRET and jwt:
            try:
                decoded = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
                sub = decoded.get("sub") or decoded.get("user_id") or decoded.get("id")
                if sub:
                    return str(sub)
            except Exception:
                return None
        return token
    return None


def ensure_user(user_id: str):
    u = User.query.get(user_id)
    if not u:
        u = User(id=user_id)
        db.session.add(u)
        db.session.commit()
    return u


def format_job_for_frontend_db(job: Job) -> dict:
    meta = job.meta or {}
    audio_name = os.path.basename(meta.get("orig_path", "")) or meta.get("orig_name") or "unknown"
    return {
        "id": job.id,
        "audioFile": {
            "id": meta.get("audio_id") or job.id,
            "name": audio_name,
            "size": meta.get("orig_size") or 0,
            "type": meta.get("orig_type") or "audio/*",
            "file": None,
        },
        "status": job.status,
        "progress": meta.get("progress"),
        "error": job.error,
        "createdAt": job.created_at.isoformat() + "Z",
        "completedAt": job.updated_at.isoformat() + "Z" if job.status == "ready" else None
    }


def format_transcription_for_frontend_db(job: Job) -> dict:
    text = (job.result or {}).get("transcription", "")
    segments_raw = (job.result or {}).get("segments")
    duration = (job.meta or {}).get("duration", 0.0)
    segments = []
    if segments_raw:
        for i, s in enumerate(segments_raw):
            segments.append({
                "id": s.get("id") or f"seg_{i}",
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", s.get("start", 0.0))),
                "text": s.get("text", ""),
                "speaker": s.get("speaker"),
                "confidence": float(s.get("confidence", 1.0)) if s.get("confidence") is not None else 1.0
            })
    else:
        segments = [{
            "id": "seg_0",
            "start": 0.0,
            "end": float(duration or 0.0),
            "text": text or "",
            "speaker": None,
            "confidence": 1.0
        }]
    return {
        "id": f"trans_{job.id}",
        "jobId": job.id,
        "segments": segments,
        "fullText": text or "",
        "language": (job.meta or {}).get("language", "en")
    }


# -------------------------
# Audio file utilities
# -------------------------
# use /tmp when running in read-only environment (e.g. Vercel)
DEFAULT_TMP_DIR = "/tmp"
# Allow override via env if needed
UPLOAD_BASE = os.getenv("UPLOAD_DIR") or os.path.join(DEFAULT_TMP_DIR, "uploads")

# define subdirectories in tmp (or override)
RAW_UPLOAD_DIR = os.path.join(UPLOAD_BASE, "raw")
CONVERTED_DIR = os.path.join(UPLOAD_BASE, "converted")
EXPORT_DIR = os.path.join(UPLOAD_BASE, "exports")

# Create these directories safely, using only in tmp
for d in (RAW_UPLOAD_DIR, CONVERTED_DIR, EXPORT_DIR):
    try:
        os.makedirs(d, exist_ok=True)
    except OSError as e:
        # If we cannot create because read-only, log and proceed (uploads will fail)
        print(f"Warning: could not create directory {d}: {e}")

DIRECT_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".webm"}


def save_uploaded_file_obj(f) -> str:
    fn = f.filename
    ext = os.path.splitext(fn)[1].lower()
    unique = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex}{ext}"
    path = os.path.join(RAW_UPLOAD_DIR, unique)
    f.save(path)
    return path


def convert_to_mp3_path(src_path: str):
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    out = os.path.join(CONVERTED_DIR, f"{ts}_{uuid.uuid4().hex}.mp3")
    try:
        from moviepy.editor import AudioFileClip
        audio_clip = AudioFileClip(src_path)
        duration = float(audio_clip.duration)
        audio_clip.write_audiofile(out, logger=None)
        audio_clip.close()
        try:
            os.remove(src_path)
        except Exception:
            pass
        return out, duration
    except Exception:
        cmd = ["ffmpeg", "-y", "-i", src_path, "-vn", "-acodec", "libmp3lame", out]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="ignore")
            raise RuntimeError(f"ffmpeg conversion failed: {stderr}")
        try:
            os.remove(src_path)
        except Exception:
            pass
        return out, 0.0


def transcribe_with_groq_bytes(data: bytes, model: str = TRANSCRIBE_MODEL, timeout: int = 120):
    return safe_call(groq_client.audio.transcriptions.create, file=("audio.wav", data), model=model, timeout=timeout)


# -------------------------
# Background worker (DB-backed)
# -------------------------
WORKER_SLEEP = 1.0
WORKER_THREAD: Optional[threading.Thread] = None
WORKER_SHUTDOWN = False


def worker_loop_db():
    global WORKER_SHUTDOWN
    while not WORKER_SHUTDOWN:
        try:
            with app.app_context():
                job = Job.query.filter_by(type="transcription", status="queued").order_by(Job.created_at).first()
                if not job:
                    time.sleep(WORKER_SLEEP)
                    continue
                job.status = "processing"
                job.updated_at = datetime.utcnow()
                db.session.commit()

                audio_path = (job.meta or {}).get("audio_path")
                if not audio_path or not os.path.exists(audio_path):
                    job.status = "failed"
                    job.error = "Audio missing"
                    job.updated_at = datetime.utcnow()
                    db.session.commit()
                    continue

                # read audio bytes
                with open(audio_path, "rb") as fh:
                    data = fh.read()
                try:
                    transcript = transcribe_with_groq_bytes(data)
                    # robust extraction
                    text = None
                    segments = None
                    if isinstance(transcript, dict):
                        text = transcript.get("text") or transcript.get("transcription") or ""
                        segments = transcript.get("segments")
                    else:
                        text = getattr(transcript, "text", None) or str(transcript)
                        segments = getattr(transcript, "segments", None)

                    job.result = {"transcription": text, "segments": segments}
                    job.meta = job.meta or {}
                    job.meta["duration"] = job.meta.get("duration", 0.0)
                    job.status = "ready"
                    job.updated_at = datetime.utcnow()
                    db.session.commit()

                    # cleanup file
                    try:
                        if os.path.exists(audio_path):
                            os.remove(audio_path)
                    except Exception:
                        pass
                except Exception as e:
                    job.status = "failed"
                    job.error = str(e)
                    job.updated_at = datetime.utcnow()
                    db.session.commit()
        except Exception:
            print("Worker loop error:", traceback.format_exc())
            time.sleep(1.0)


def start_worker_db():
    global WORKER_THREAD
    if WORKER_THREAD is None:
        WORKER_THREAD = threading.Thread(target=worker_loop_db, daemon=True)
        WORKER_THREAD.start()


start_worker_db()

# -------------------------
# Routes (compatibility)
# -------------------------
@app.route("/api/v1/audio/upload", methods=["POST"])
def api_upload_audio():
    user_id = extract_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Missing user identity (X-User-Id or Authorization Bearer)"}), 401
    ensure_user(user_id)

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    try:
        raw_path = save_uploaded_file_obj(f)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    ext = os.path.splitext(raw_path)[1].lower()
    duration = 0.0
    if ext not in DIRECT_EXTS:
        try:
            send_path, duration = convert_to_mp3_path(raw_path)
        except Exception as e:
            if os.path.exists(raw_path):
                os.remove(raw_path)
            return jsonify({"error": f"Conversion failed: {str(e)}"}), 500
    else:
        send_path = raw_path
        # best-effort duration
        try:
            from moviepy.editor import AudioFileClip
            clip = AudioFileClip(send_path)
            duration = float(clip.duration)
            clip.close()
        except Exception:
            duration = 0.0

    meta = {
        "audio_path": send_path,
        "orig_path": raw_path,
        "orig_name": f.filename,
        "orig_size": os.path.getsize(raw_path) if os.path.exists(raw_path) else 0,
        "orig_type": getattr(f, "content_type", None) or "",
        "duration": duration,
    }

    job_id = uuid.uuid4().hex
    job = Job(
        id=job_id,
        user_id=user_id,
        type="transcription",
        status="queued",
        meta=meta,
        result=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.session.add(job)
    db.session.commit()
    return (
        jsonify({"job_id": job_id, "status": "queued", "job": format_job_for_frontend_db(job)}),
        201,
    )


@app.route("/api/v1/jobs/<job_id>/status", methods=["GET"])
def api_job_status_db(job_id):
    user_id = extract_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Missing user identity"}), 401
    job = Job.query.get(job_id)
    if not job or job.user_id != user_id:
        return jsonify({"error": "Not found"}), 404
    return jsonify(format_job_for_frontend_db(job)), 200


@app.route("/api/v1/jobs/<job_id>/transcription", methods=["GET"])
def api_get_transcription_db(job_id):
    user_id = extract_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Missing user identity"}), 401
    job = Job.query.get(job_id)
    if not job or job.user_id != user_id:
        return jsonify({"error": "Not found"}), 404
    if job.type != "transcription":
        return jsonify({"error": "Job is not a transcription job"}), 400
    if job.status != "ready":
        return jsonify({"status": job.status}), 202
    return jsonify(format_transcription_for_frontend_db(job)), 200


@app.route("/api/v1/jobs/<job_id>/summary", methods=["POST"])
def api_create_summary_db(job_id):
    user_id = extract_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Missing user identity"}), 401
    parent_job = Job.query.get(job_id)
    if not parent_job or parent_job.user_id != user_id:
        return jsonify({"error": "Not found"}), 404

    data = request.get_json(silent=True) or {}
    style = data.get("style", "short")
    text = data.get("text") or (parent_job.result or {}).get("transcription")
    if not text:
        return jsonify({"error": "No text available for summarization"}), 400

    # synchronous summarize & persist
    chat = GroqChat(system_prompt=DEFAULT_SYSTEM_PROMPT)
    try:
        summary_text = chat.summarize(text, style=style)
    except Exception as e:
        return jsonify({"error": f"Summary failed: {str(e)}"}), 500

    sid = uuid.uuid4().hex
    sjob = Job(
        id=sid,
        user_id=user_id,
        type="summary",
        status="ready",
        meta={"transcriptionId": job_id, "style": style},
        result={"summary": summary_text},
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.session.add(sjob)
    db.session.commit()
    return (
        jsonify(
            {
                "summary_job_id": sid,
                "status": "ready",
                "job": format_job_for_frontend_db(sjob),
            }
        ),
        201,
    )


@app.route("/api/v1/jobs/<job_id>/qa", methods=["POST"])
def api_create_qa_db(job_id):
    user_id = extract_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Missing user identity"}), 401
    parent_job = Job.query.get(job_id)
    if not parent_job or parent_job.user_id != user_id:
        return jsonify({"error": "Not found"}), 404

    question = None
    requirement_text = None
    context_text = None

    if request.content_type and "multipart/form-data" in request.content_type:
        question = request.form.get("question")
        context_text = request.form.get("context")
        req_file = request.files.get("requirement_file")
        if req_file:
            r_ext = os.path.splitext(req_file.filename)[1].lower()
            if r_ext in {".txt", ".md"}:
                requirement_text = req_file.read().decode("utf-8", errors="ignore")
            elif r_ext == ".pdf" and PyPDF2:
                buf = io.BytesIO(req_file.read())
                try:
                    reader = PyPDF2.PdfReader(buf)
                    pages = [p.extract_text() or "" for p in reader.pages]
                    requirement_text = "\n".join(pages)
                except Exception:
                    requirement_text = None
            else:
                # store file for later inspection
                save_uploaded_file_obj(req_file)
    else:
        data = request.get_json(silent=True) or {}
        question = data.get("question")
        requirement_text = data.get("requirement_text")
        context_text = data.get("context")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # prefer transcription context if not explicitly provided
    if not context_text:
        context_text = (parent_job.result or {}).get("transcription", "") or ""

    # persist user question message
    cm = ConversationMessage(job_id=parent_job.id, role="user", content=question)
    db.session.add(cm)
    db.session.commit()

    # build conversation context
    msgs = ConversationMessage.query.filter_by(job_id=parent_job.id).order_by(ConversationMessage.created_at).all()
    conv_text = "\n".join([f"{m.role}: {m.content}" for m in msgs])

    # synchronous LLM QA (persisted)
    chat = GroqChat(system_prompt=DEFAULT_SYSTEM_PROMPT)
    try:
        answer = chat.answer_with_context(f"{context_text}\n\nConversation:\n{conv_text}", question, requirement_text=requirement_text)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"QA failed: {str(e)}"}), 500

    # persist assistant reply
    cm2 = ConversationMessage(job_id=parent_job.id, role="assistant", content=answer)
    db.session.add(cm2)

    qid = uuid.uuid4().hex
    qjob = Job(
        id=qid,
        user_id=user_id,
        type="qa",
        status="ready",
        meta={"context_job_id": parent_job.id},
        result={"answer": answer},
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.session.add(qjob)
    db.session.commit()

    return (
        jsonify(
            {
                "qa_job_id": qid,
                "status": "ready",
                "job": format_job_for_frontend_db(qjob),
            }
        ),
        201,
    )


@app.route("/api/v1/jobs", methods=["GET"])
def api_list_jobs_db():
    user_id = extract_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Missing user identity"}), 401
    jobs_q = Job.query.filter_by(user_id=user_id).order_by(Job.created_at.desc()).all()
    out = [format_job_for_frontend_db(j) for j in jobs_q]
    return jsonify(out), 200


@app.route("/api/v1/jobs/<job_id>/download", methods=["GET"])
def api_download_db(job_id):
    user_id = extract_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Missing user identity"}), 401
    job = Job.query.get(job_id)
    if not job or job.user_id != user_id:
        return jsonify({"error": "Not found"}), 404
    fmt = request.args.get("format", "txt")
    if job.status != "ready":
        return jsonify({"status": job.status}), 202
    if job.type == "transcription":
        text = (job.result or {}).get("transcription", "")
    elif job.type == "summary":
        text = (job.result or {}).get("summary", "")
    elif job.type == "qa":
        text = (job.result or {}).get("answer", "")
    else:
        return jsonify({"error": "Unsupported job type for download"}), 400
    out = io.BytesIO(text.encode("utf-8"))
    out.seek(0)
    return send_file(out, as_attachment=True, download_name=f"{job_id}.txt")


@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "DB-backed transcription/QA API running"}), 200


# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
