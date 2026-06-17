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

ALLOWED_ORIGINS = [
    "https://sths-sms.vercel.app",
    "http://localhost:5173",
    "http://localhost:5273",
    "http://localhost:4173",
]


def start_local_ollama():
    """
    Check if Ollama is running on port 11434.
    If not, check if 'ollama' command exists in the system.
    If yes, run 'ollama serve' in a background process.
    If no, download the official standalone Ollama binary/installer and run it!
    Then, verify if model 'gemma4:e2b' is installed. If not, pull it and print progress in console.
    """

    def pull_gemma_model():
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags")
            with urllib.request.urlopen(req, timeout=2) as response:
                data = json.loads(response.read().decode())
                models = data.get("models", [])
                has_gemma = any(
                    m.get("name") == "gemma4:e2b" or
                    m.get("name").startswith("gemma4:e2b") or
                    m.get("model") == "gemma4:e2b"
                    for m in models
                )
                if has_gemma:
                    print("\n[Ollama] gemma4:e2b model is already installed.")
                    return
        except Exception as e:
            print(f"\n[Ollama Error] Failed to check models: {e}")
            return

        platform = sys.platform
        bin_dir = Path("./bin")
        ollama_bin = "ollama"
        if platform == "darwin" and (bin_dir / "ollama").exists():
            ollama_bin = str(bin_dir / "ollama")
        elif platform == "win32" and (bin_dir / "ollama.exe").exists():
            ollama_bin = str(bin_dir / "ollama.exe")

        print(f"\n[Ollama] gemma4:e2b model not found. Pulling via CLI ({ollama_bin})...")
        try:
            result = subprocess.run([ollama_bin, "pull", "gemma4:e2b"], check=True)
            if result.returncode == 0:
                print("[Ollama] gemma4:e2b model pulled successfully via CLI.")
            else:
                print(f"[Ollama Warning] pull command exited with code {result.returncode}")
        except Exception as e:
            print(f"[Ollama Error] Failed to pull model via subprocess CLI: {e}")

    # AMD GPU safety: set env vars that completely hide the GPU from ROCm/HIP stack.
    # OLLAMA_NUM_GPU=0 alone is insufficient — GGML scheduler still detects the GPU
    # device and attempts CPU+GPU splits mid-inference, triggering GGML_SCHED_MAX_SPLIT_INP.
    # ROCR/HIP vars make the GPU invisible at the driver level before Ollama initializes.
    _ollama_env = {
        **os.environ,
        'OLLAMA_NUM_GPU': '0',
        'ROCR_VISIBLE_DEVICES': '-1',   # hide GPU from AMD ROCm runtime
        'HIP_VISIBLE_DEVICES': '-1',    # hide GPU from AMD HIP runtime
    }

    # 1. On Windows, kill any existing Ollama and restart with safe env vars.
    #    "Already running" Ollama may have been started without the AMD-safe vars.
    if sys.platform == 'win32':
        try:
            subprocess.run(
                ['taskkill', '/F', '/IM', 'ollama.exe'],
                capture_output=True, creationflags=0x08000000
            )
            print("[Ollama] Stopped existing Ollama process (restarting with AMD-safe env).")
            time.sleep(1)
        except Exception:
            pass

    # Check port after potential kill
    try:
        with urllib.request.urlopen("http://localhost:11434/", timeout=1) as response:
            print("[Ollama] Service is already running.")
            pull_gemma_model()
            return
    except Exception:
        pass

    # 2. Try starting it if ollama command exists
    print("[Ollama] Port 11434 is closed. Attempting to start local Ollama...")
    try:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         env=_ollama_env)
        for _ in range(5):
            time.sleep(1)
            try:
                with urllib.request.urlopen("http://localhost:11434/", timeout=1) as response:
                    print("[Ollama] Started successfully from system PATH.")
                    pull_gemma_model()
                    return
            except Exception:
                pass
    except FileNotFoundError:
        pass

    # 3. If ollama command does not exist, download standalone binary/installer
    platform = sys.platform
    bin_dir = Path("./bin")
    bin_dir.mkdir(exist_ok=True)

    if platform == "darwin":
        ollama_bin = bin_dir / "ollama"
        if not ollama_bin.exists():
            print("[Ollama] Downloading Ollama CLI binary for macOS...")
            url = "https://ollama.com/download/ollama-darwin"
            try:
                urllib.request.urlretrieve(url, str(ollama_bin))
                ollama_bin.chmod(0o755)
            except Exception as e:
                print(f"[Ollama Error] Failed to download macOS binary: {e}")
                return
        print("[Ollama] Starting local Ollama daemon from local bin...")
        subprocess.Popen([str(ollama_bin), "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         env=_ollama_env)

    elif platform == "win32":
        ollama_bin = bin_dir / "ollama.exe"
        if not ollama_bin.exists():
            print("[Ollama] Downloading Ollama standalone CLI for Windows...")
            url = "https://github.com/ollama/ollama/releases/latest/download/ollama-windows-amd64.zip"
            zip_path = bin_dir / "ollama-windows.zip"
            try:
                urllib.request.urlretrieve(url, str(zip_path))
                print("[Ollama] Extracting Ollama files...")
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(str(bin_dir))
                zip_path.unlink(missing_ok=True)
            except Exception as e:
                print(f"[Ollama Error] Failed to set up Windows standalone Ollama: {e}")
                return
        print("[Ollama] Starting local Ollama daemon from local bin...")
        subprocess.Popen([str(ollama_bin), "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         env=_ollama_env)

    # Wait for Ollama startup confirmation
    for _ in range(15):
        time.sleep(1)
        try:
            with urllib.request.urlopen("http://localhost:11434/", timeout=1) as response:
                print("[Ollama] Daemon initialized successfully.")
                pull_gemma_model()
                return
        except Exception:
            pass
    print("[Ollama Warning] Failed to confirm local Ollama startup.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=start_local_ollama, daemon=True).start()
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


@app.get("/api/stt/ping")
async def ping():
    return {"status": "ok"}


@app.post("/api/stt/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    diarize: bool = Query(True)
):
    if not file.content_type.startswith("audio/") and not file.filename.endswith((".webm", ".wav", ".mp3", ".m4a", ".ogg")):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload an audio file.")

    temp_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1] or ".webm"
    temp_filepath = UPLOAD_DIR / f"{temp_id}{ext}"

    try:
        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write temporary file: {str(e)}")

    try:
        text = transcribe_audio(str(temp_filepath), diarize=diarize)
        return {"success": True, "text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT Transcription failed: {str(e)}")
    finally:
        if temp_filepath.exists():
            try:
                temp_filepath.unlink()
            except Exception as cleanup_err:
                print(f"Warning: Failed to delete temp file {temp_filepath}: {cleanup_err}")


@app.get("/api/ai/status")
def ai_status():
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=2) as response:
            data = json.loads(response.read().decode())
            models = data.get("models", [])
            has_gemma = any(
                m.get("name") == "gemma4:e2b" or
                m.get("name").startswith("gemma4:e2b") or
                m.get("model") == "gemma4:e2b"
                for m in models
            )
            return {
                "ollama_running": True,
                "has_model": has_gemma,
                "models": [m.get("name") for m in models]
            }
    except Exception as e:
        return {"ollama_running": False, "has_model": False, "error": str(e)}


@app.post("/api/ai/pull")
def ai_pull():
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
    prompt = f"""당신은 전문적인 학교 상담 교사입니다. 다음 상담 기록 대화 요지 및 관찰 내용을 바탕으로 두 가지를 작성해 주세요.
1. 상담 요약 (핵심적인 내용을 3~4개의 글머리 기호 문장으로 요약)
2. 추후 지도 계획 (Action Plan) (구체적이고 실천 가능한 학생 관리 및 지도 방안을 2~3개 제시)

반드시 아래와 같은 형식(양식)으로 답변해 주십시오. 다른 안내 문구나 서론, 결론은 생략하십시오.

[상담 요약]
- 요약 내용 1
- 요약 내용 2
- 요약 내용 3

[추후 지도 계획]
- 계획 내용 1
- 계획 내용 2

상담 기록 내용:
{payload.content}"""

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
    # Kill any stale process holding port 8000 before uvicorn binds (prevents Errno 10048 after auto-patch restart)
    if sys.platform == 'win32':
        try:
            _r = subprocess.run(
                ['netstat', '-ano'], capture_output=True, text=True,
                creationflags=0x08000000  # CREATE_NO_WINDOW
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
