import os
import sys
import uuid
import json
import shutil
import subprocess
import threading
import time
import urllib.request
import urllib.error
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from stt_service import transcribe_audio


UPLOAD_DIR = Path("./uploads/stt")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# AMD path globals
_amd_mode = False
_llama_model = None
_llama_model_ready = False

GGUF_FILENAME = "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
GGUF_URL = (
    "https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF"
    "/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
)
GGUF_PATH = Path("./models") / GGUF_FILENAME

ALLOWED_ORIGINS = [
    "https://sths-sms.vercel.app",
    "http://localhost:5173",
    "http://localhost:5273",
    "http://localhost:4173",
]


# ── AMD path ──────────────────────────────────────────────────────────────────

def _detect_amd() -> bool:
    if sys.platform != 'win32':
        return False

    def _check(output: str) -> bool:
        low = output.lower()
        return 'amd' in low or 'radeon' in low

    # Method 1: wmic
    try:
        r = subprocess.run(
            ['wmic', 'path', 'win32_VideoController', 'get', 'name'],
            capture_output=True, text=True, creationflags=0x08000000, timeout=5
        )
        print(f"[AI] GPU 감지 (wmic): {r.stdout.strip()}")
        if r.stdout.strip():
            if _check(r.stdout):
                return True
    except Exception as e:
        print(f"[AI] wmic 실패: {e}")

    # Method 2: PowerShell fallback
    try:
        r = subprocess.run(
            ['powershell', '-NoProfile', '-Command',
             'Get-WmiObject Win32_VideoController | Select-Object -ExpandProperty Name'],
            capture_output=True, text=True, creationflags=0x08000000, timeout=8
        )
        print(f"[AI] GPU 감지 (PS): {r.stdout.strip()}")
        if _check(r.stdout):
            return True
    except Exception as e:
        print(f"[AI] PowerShell 실패: {e}")

    # Method 3: AMD driver folder
    for folder in [Path("C:/Program Files/AMD"), Path("C:/Program Files (x86)/AMD")]:
        if folder.exists():
            print(f"[AI] AMD 드라이버 폴더 감지: {folder}")
            return True

    return False


def _init_llama_cpp():
    global _llama_model, _llama_model_ready

    # 1. Install llama-cpp-python (CPU-only wheel) if missing
    try:
        import llama_cpp  # noqa: F401
        print("[AMD-AI] llama-cpp-python 이미 설치됨.")
    except ImportError:
        print("[AMD-AI] llama-cpp-python 설치 중 (최초 1회)...")
        try:
            subprocess.run(
                [
                    sys.executable, '-m', 'pip', 'install',
                    'llama-cpp-python',
                    '--extra-index-url',
                    'https://abetlen.github.io/llama-cpp-python/whl/cpu',
                    '--quiet',
                ],
                check=True,
            )
            print("[AMD-AI] llama-cpp-python 설치 완료.")
        except Exception as e:
            print(f"[AMD-AI Error] 설치 실패: {e}")
            return

    # 2. Download GGUF model if not present
    GGUF_PATH.parent.mkdir(exist_ok=True)
    if not GGUF_PATH.exists():
        print(f"[AMD-AI] 모델 다운로드 중... (약 1.1GB, 시간이 걸립니다)")
        tmp_path = GGUF_PATH.with_suffix('.tmp')
        try:
            def _progress(block, block_size, total):
                if total > 0 and block % 200 == 0:
                    mb = block * block_size // 1024 // 1024
                    total_mb = total // 1024 // 1024
                    print(f"[AMD-AI] 다운로드 중... {mb}MB / {total_mb}MB")
            urllib.request.urlretrieve(GGUF_URL, str(tmp_path), reporthook=_progress)
            tmp_path.rename(GGUF_PATH)
            print("[AMD-AI] 모델 다운로드 완료.")
        except Exception as e:
            print(f"[AMD-AI Error] 모델 다운로드 실패: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
            return

    # 3. Load model into memory
    print("[AMD-AI] 모델 로딩 중...")
    try:
        from llama_cpp import Llama
        _llama_model = Llama(
            model_path=str(GGUF_PATH),
            n_ctx=4096,
            n_gpu_layers=0,   # CPU only — no GPU involvement
            verbose=False,
        )
        _llama_model_ready = True
        print("[AMD-AI] 모델 로딩 완료. AI 기능 준비됨.")
    except Exception as e:
        print(f"[AMD-AI Error] 모델 로딩 실패: {e}")


# ── Ollama path ───────────────────────────────────────────────────────────────

def _start_ollama():
    def pull_gemma_model():
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags")
            with urllib.request.urlopen(req, timeout=2) as response:
                data = json.loads(response.read().decode())
                models = data.get("models", [])
                has_gemma = any(
                    m.get("name") == "gemma4:e2b" or
                    m.get("name", "").startswith("gemma4:e2b") or
                    m.get("model") == "gemma4:e2b"
                    for m in models
                )
                if has_gemma:
                    print("\n[Ollama] gemma4:e2b model is already installed.")
                    return
        except Exception as e:
            print(f"\n[Ollama Error] Failed to check models: {e}")
            return

        bin_dir = Path("./bin")
        ollama_bin = "ollama"
        if sys.platform == "darwin" and (bin_dir / "ollama").exists():
            ollama_bin = str(bin_dir / "ollama")
        elif sys.platform == "win32" and (bin_dir / "ollama.exe").exists():
            ollama_bin = str(bin_dir / "ollama.exe")

        print(f"\n[Ollama] gemma4:e2b not found. Pulling via CLI ({ollama_bin})...")
        try:
            result = subprocess.run([ollama_bin, "pull", "gemma4:e2b"], check=True)
            if result.returncode == 0:
                print("[Ollama] gemma4:e2b pulled successfully.")
        except Exception as e:
            print(f"[Ollama Error] Failed to pull model: {e}")

    _ollama_env = {**os.environ, 'OLLAMA_NUM_GPU': '0'}

    # Kill any existing Ollama on Windows so our env vars take effect
    if sys.platform == 'win32':
        try:
            subprocess.run(
                ['taskkill', '/F', '/IM', 'ollama.exe'],
                capture_output=True, creationflags=0x08000000
            )
            time.sleep(1)
        except Exception:
            pass

    try:
        with urllib.request.urlopen("http://localhost:11434/", timeout=1):
            print("[Ollama] Service is already running.")
            pull_gemma_model()
            return
    except Exception:
        pass

    print("[Ollama] Starting local Ollama...")
    try:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, env=_ollama_env)
        for _ in range(5):
            time.sleep(1)
            try:
                with urllib.request.urlopen("http://localhost:11434/", timeout=1):
                    print("[Ollama] Started from system PATH.")
                    pull_gemma_model()
                    return
            except Exception:
                pass
    except FileNotFoundError:
        pass

    bin_dir = Path("./bin")
    bin_dir.mkdir(exist_ok=True)

    if sys.platform == "darwin":
        ollama_bin = bin_dir / "ollama"
        if not ollama_bin.exists():
            print("[Ollama] Downloading macOS binary...")
            try:
                urllib.request.urlretrieve(
                    "https://ollama.com/download/ollama-darwin", str(ollama_bin))
                ollama_bin.chmod(0o755)
            except Exception as e:
                print(f"[Ollama Error] Download failed: {e}")
                return
        subprocess.Popen([str(ollama_bin), "serve"], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, env=_ollama_env)

    elif sys.platform == "win32":
        ollama_bin = bin_dir / "ollama.exe"
        if not ollama_bin.exists():
            print("[Ollama] Downloading Windows standalone...")
            zip_path = bin_dir / "ollama-windows.zip"
            try:
                urllib.request.urlretrieve(
                    "https://github.com/ollama/ollama/releases/latest/download/ollama-windows-amd64.zip",
                    str(zip_path))
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(str(bin_dir))
                zip_path.unlink(missing_ok=True)
            except Exception as e:
                print(f"[Ollama Error] Setup failed: {e}")
                return
        subprocess.Popen([str(ollama_bin), "serve"], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, env=_ollama_env)

    for _ in range(15):
        time.sleep(1)
        try:
            with urllib.request.urlopen("http://localhost:11434/", timeout=1):
                print("[Ollama] Daemon ready.")
                pull_gemma_model()
                return
        except Exception:
            pass
    print("[Ollama Warning] Startup confirmation timed out.")


# ── Backend selector ──────────────────────────────────────────────────────────

def setup_ai_backend():
    global _amd_mode
    if _detect_amd():
        _amd_mode = True
        print("[AI] AMD/Radeon GPU 감지 — llama-cpp-python CPU 모드로 실행합니다.")
        _init_llama_cpp()
    else:
        _start_ollama()


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=setup_ai_backend, daemon=True).start()
    yield


app = FastAPI(title="STHS SMS STT Backend API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


class AnalyzeRequest(BaseModel):
    content: str


# ── STT endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/stt/ping")
async def ping():
    return {"status": "ok"}


@app.post("/api/stt/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    diarize: bool = Query(True)
):
    if not file.content_type.startswith("audio/") and not file.filename.endswith(
            (".webm", ".wav", ".mp3", ".m4a", ".ogg")):
        raise HTTPException(status_code=400, detail="Invalid file format.")

    temp_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1] or ".webm"
    temp_filepath = UPLOAD_DIR / f"{temp_id}{ext}"

    try:
        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write temp file: {e}")

    try:
        text = transcribe_audio(str(temp_filepath), diarize=diarize)
        return {"success": True, "text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT failed: {e}")
    finally:
        if temp_filepath.exists():
            try:
                temp_filepath.unlink()
            except Exception:
                pass


# ── AI endpoints ──────────────────────────────────────────────────────────────

@app.get("/api/ai/status")
def ai_status():
    if _amd_mode:
        return {
            "ollama_running": _llama_model_ready,
            "has_model": _llama_model_ready,
            "amd_mode": True,
        }
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=2) as response:
            data = json.loads(response.read().decode())
            models = data.get("models", [])
            has_gemma = any(
                m.get("name") == "gemma4:e2b" or
                m.get("name", "").startswith("gemma4:e2b") or
                m.get("model") == "gemma4:e2b"
                for m in models
            )
            return {"ollama_running": True, "has_model": has_gemma,
                    "models": [m.get("name") for m in models]}
    except Exception as e:
        return {"ollama_running": False, "has_model": False, "error": str(e)}


@app.post("/api/ai/pull")
def ai_pull():
    if _amd_mode:
        # Model is auto-downloaded by _init_llama_cpp; nothing to pull here
        def _noop():
            yield (json.dumps({"status": "AMD 모드: 모델이 자동으로 다운로드됩니다.",
                               "done": True}) + "\n").encode("utf-8")
        return StreamingResponse(_noop(), media_type="application/x-ndjson")

    def generate_pull_progress():
        try:
            req_data = json.dumps({"name": "gemma4:e2b"}).encode("utf-8")
            req = urllib.request.Request(
                "http://localhost:11434/api/pull",
                data=req_data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req) as response:
                while True:
                    line = response.readline()
                    if not line:
                        break
                    yield (line.decode("utf-8", errors="replace").strip() + "\n").encode("utf-8")
        except Exception as e:
            yield (json.dumps({"error": str(e)}) + "\n").encode("utf-8")

    return StreamingResponse(generate_pull_progress(), media_type="application/x-ndjson")


@app.post("/api/ai/analyze")
def ai_analyze(payload: AnalyzeRequest):
    system_msg = (
        "당신은 전문적인 학교 상담 교사입니다. "
        "반드시 아래 형식으로만 답변하고 다른 말은 생략하십시오.\n\n"
        "[상담 요약]\n- 요약 내용 1\n- 요약 내용 2\n- 요약 내용 3\n\n"
        "[추후 지도 계획]\n- 계획 내용 1\n- 계획 내용 2"
    )
    user_msg = (
        "다음 상담 기록을 바탕으로 상담 요약(3~4개 항목)과 "
        "추후 지도 계획(2~3개 항목)을 위 형식으로 작성해주세요.\n\n"
        f"상담 기록:\n{payload.content}"
    )

    # ── AMD path: llama-cpp-python ────────────────────────────────────────────
    if _amd_mode:
        def generate_llama_stream():
            if not _llama_model_ready:
                yield (json.dumps({
                    "error": "AI 모델 로딩 중입니다. 잠시 후 다시 시도하세요."
                }) + "\n").encode("utf-8")
                return
            print("\n[AMD-AI] 분석 요청 수신. 생성 중...\n")
            try:
                stream = _llama_model.create_chat_completion(
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user",   "content": user_msg},
                    ],
                    max_tokens=1024,
                    temperature=0.7,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk["choices"][0]["delta"]
                    token = delta.get("content", "")
                    finish = chunk["choices"][0]["finish_reason"]
                    done = finish is not None
                    if token:
                        sys.stdout.write(token)
                        sys.stdout.flush()
                    yield (json.dumps({"response": token, "done": done}) + "\n").encode("utf-8")
                    if done:
                        break
                print("\n\n[AMD-AI] 생성 완료.")
            except Exception as e:
                print(f"\n[AMD-AI Error] {e}")
                yield (json.dumps({"error": str(e)}) + "\n").encode("utf-8")

        return StreamingResponse(generate_llama_stream(), media_type="application/x-ndjson")

    # ── Ollama path ───────────────────────────────────────────────────────────
    prompt = (
        f"당신은 전문적인 학교 상담 교사입니다. 다음 상담 기록 대화 요지 및 관찰 내용을 바탕으로 두 가지를 작성해 주세요.\n"
        "1. 상담 요약 (핵심적인 내용을 3~4개의 글머리 기호 문장으로 요약)\n"
        "2. 추후 지도 계획 (Action Plan) (구체적이고 실천 가능한 학생 관리 및 지도 방안을 2~3개 제시)\n\n"
        "반드시 아래와 같은 형식(양식)으로 답변해 주십시오. 다른 안내 문구나 서론, 결론은 생략하십시오.\n\n"
        "[상담 요약]\n- 요약 내용 1\n- 요약 내용 2\n- 요약 내용 3\n\n"
        "[추후 지도 계획]\n- 계획 내용 1\n- 계획 내용 2\n\n"
        f"상담 기록 내용:\n{payload.content}"
    )

    print("\n[Ollama] Streaming AI Analysis Request Received.")
    print("[Ollama] Generating AI Response: \n")

    def generate_analysis_stream():
        try:
            req_data = json.dumps({
                "model": "gemma4:e2b",
                "prompt": prompt,
                "stream": True
            }).encode("utf-8")
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=req_data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=90) as response:
                while True:
                    line = response.readline()
                    if not line:
                        break
                    try:
                        decoded = line.decode("utf-8", errors="replace").strip()
                        data = json.loads(decoded)
                        token = data.get("response", "")
                        sys.stdout.write(token)
                        sys.stdout.flush()
                        yield (decoded + "\n").encode("utf-8")
                        if data.get("done", False):
                            break
                    except Exception as inner_e:
                        print(f"\n[Ollama Warning] JSON parse error: {inner_e}")
            print("\n\n[Ollama] AI Generation completed successfully.")
        except Exception as e:
            print(f"\n[Ollama Error] Failed to generate AI analysis: {e}")
            yield (json.dumps({"error": str(e)}) + "\n").encode("utf-8")

    return StreamingResponse(generate_analysis_stream(), media_type="application/x-ndjson")


if __name__ == "__main__":
    # Kill any stale process on port 8000 before uvicorn binds (prevents Errno 10048)
    if sys.platform == 'win32':
        try:
            _r = subprocess.run(
                ['netstat', '-ano'], capture_output=True, text=True,
                creationflags=0x08000000
            )
            _my_pid = str(os.getpid())
            for _line in _r.stdout.splitlines():
                if ':8000 ' in _line and 'LISTENING' in _line:
                    _pid = _line.split()[-1]
                    if _pid.isdigit() and _pid != _my_pid:
                        subprocess.run(
                            ['taskkill', '/F', '/PID', _pid],
                            capture_output=True, creationflags=0x08000000
                        )
                        print(f"[Startup] Killed stale process on port 8000 (PID {_pid})")
                        time.sleep(0.5)
        except Exception:
            pass
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="warning", reload=False)
