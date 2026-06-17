import os
import wave
import struct
import math
import subprocess
from pathlib import Path
from faster_whisper import WhisperModel

# Detect environment: Hugging Face Space vs Local PC
IS_HF_SPACE = "SPACE_ID" in os.environ

if IS_HF_SPACE:
    # HF Space Cloud: base model + int8 for low memory and CPU footprint
    MODEL_SIZE = "base"
    COMPUTE_TYPE = "int8"
else:
    # Local PC: Upgrade to small model for vastly superior Korean recognition
    MODEL_SIZE = "small"
    COMPUTE_TYPE = "int8"

model = None

def get_model():
    global model
    if model is None:
        model = WhisperModel(
            MODEL_SIZE,
            device="cpu",
            compute_type=COMPUTE_TYPE
        )
    return model

# 3.1 전문 전공 용어 및 고유 어휘 한글/영문 자동 교정 사전
CORRECTION_DICTIONARY = {
    # 1. 정보컴퓨터 교과/전공 용어 (사용자 전공 반영)
    "수파베이스": "Supabase",
    "수파 베이스": "Supabase",
    "버셀": "Vercel",
    "버 셀": "Vercel",
    "리액트": "React",
    "타입스크립트": "TypeScript",
    "타입 스크립트": "TypeScript",
    "자바스크립트": "JavaScript",
    "자바 스크립트": "JavaScript",
    "깃허브": "GitHub",
    "깃 허브": "GitHub",
    "바이트": "Vite",
    "플러터": "Flutter",
    "파이썬": "Python",
    "데이터베이스": "Database",
    "포스트그레스": "PostgreSQL",
    "포스트그레": "PostgreSQL",
    "에이피아이": "API",
    "오오쓰": "OAuth",
    "오오 스": "OAuth",
    
    # 2. 학교 행정/입시/상담 고유 용어 및 축약어 교정
    "생기부": "생활기록부",
    "자소서": "자기소개서",
    "포폴": "포트폴리오",
    "수행": "수행평가",
    "세특": "세부능력 및 특기사항",
    "창체": "창의적 체험활동",
    "행특": "행동특성 및 종합의견",
    "플젝": "프로젝트"
}

def post_process_text(text: str) -> str:
    """
    Applies regex/replace corrections on the transcribed text for technical terms.
    """
    for key, value in CORRECTION_DICTIONARY.items():
        text = text.replace(key, value)
        # Handle lowercase/uppercase variant combinations if any
    return text

def convert_to_wav(input_path: str, output_path: str):
    """
    Converts any input audio to 16kHz, mono, 16-bit PCM WAV using ffmpeg.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_path
    ]
    # Run silently
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def calculate_segment_rms(wav_path: str, start_sec: float, end_sec: float) -> float:
    """
    Calculates the Root Mean Square (RMS) volume of the audio from start_sec to end_sec.
    Used for lightweight speaker diarization.
    """
    try:
        with wave.open(wav_path, 'rb') as w:
            framerate = w.getframerate()
            sampwidth = w.getsampwidth()
            
            start_frame = int(start_sec * framerate)
            end_frame = int(end_sec * framerate)
            num_frames = end_frame - start_frame
            
            if num_frames <= 0:
                return 0.0
                
            w.setpos(start_frame)
            frames = w.readframes(num_frames)
            
            count = len(frames) // 2
            if count == 0:
                return 0.0
                
            shorts = struct.unpack(f"<{count}h", frames)
            sum_squares = sum(s ** 2 for s in shorts)
            return math.sqrt(sum_squares / count)
    except Exception as e:
        print(f"Warning: Failed to calculate RMS for segment {start_sec}-{end_sec}: {e}")
        return 0.0

def transcribe_audio(filepath: str, diarize: bool = True) -> str:
    """
    Transcribes audio file into structured text with lightweight speaker diarization (Teacher vs Student)
    and custom terminology correction.
    """
    whisper_model = get_model()
    
    # 1. Transcribe segments with timestamps
    segments, info = whisper_model.transcribe(
        filepath,
        language="ko",
        vad_filter=True
    )
    
    segments_list = list(segments)
    if not segments_list:
        return ""

    # 2. Extract PCM WAV for RMS volume profiling
    temp_wav = Path(filepath).with_suffix(".temp.wav")
    has_wav = False
    try:
        convert_to_wav(filepath, str(temp_wav))
        has_wav = True
    except Exception as e:
        print(f"Warning: ffmpeg conversion failed, defaulting to flat text without diarization: {e}")

    # 3. Profiling RMS values for each segment
    rms_values = []
    if has_wav:
        for seg in segments_list:
            val = calculate_segment_rms(str(temp_wav), seg.start, seg.end)
            rms_values.append(val)
        
        # Clean up temp WAV
        if temp_wav.exists():
            temp_wav.unlink()

    # 4. Determine Speaker labels based on RMS profile
    # Teacher (typically closer to mic) -> Higher volume RMS
    # Student (typically further away) -> Lower volume RMS
    result = []
    if has_wav and len(rms_values) > 0:
        is_single_speaker = not diarize
        
        if diarize:
            n = len(rms_values)
            if n <= 1:
                is_single_speaker = True
            else:
                sorted_rms = sorted(rms_values)
                
                # 1D K-means (K=2) partition to find optimal threshold dividing speakers
                best_i = 0
                min_variance_sum = float('inf')
                for i in range(1, n):
                    low_part = sorted_rms[:i]
                    high_part = sorted_rms[i:]
                    
                    low_mean = sum(low_part) / len(low_part)
                    high_mean = sum(high_part) / len(high_part)
                    
                    low_var = sum((x - low_mean) ** 2 for x in low_part)
                    high_var = sum((x - high_mean) ** 2 for x in high_part)
                    
                    if (low_var + high_var) < min_variance_sum:
                        min_variance_sum = low_var + high_var
                        best_i = i
                
                low_group = sorted_rms[:best_i]
                high_group = sorted_rms[best_i:]
                
                low_mean = sum(low_group) / len(low_group)
                high_mean = sum(high_group) / len(high_group)
                
                # Handle microphone static / breathing noise floor skewing low_mean
                floor_val = high_mean * 0.15
                low_mean_corrected = max(low_mean, floor_val)
                
                ratio = high_mean / (low_mean_corrected + 1e-6)
                
                avg_rms = sum(rms_values) / n
                variance = sum((x - avg_rms) ** 2 for x in rms_values) / n
                std_dev = math.sqrt(variance)
                cv = std_dev / (avg_rms + 1e-6)
                
                print(f"[STT Speaker Profiling] Segments: {n}, Avg RMS: {avg_rms:.1f}, CV: {cv:.2f}, Low Mean: {low_mean:.1f}, High Mean: {high_mean:.1f}, Corrected Ratio: {ratio:.2f}")
                
                # Single speaker conditions:
                # 1. Ratio of means is small (little differences in speech levels, likely 1 speaker)
                # 2. One of the clusters has too few segments (likely outlier noise or occasional cough/mutter)
                # 3. Overall CV is low (stable speech level)
                if ratio < 2.2 or len(low_group) < n * 0.20 or len(high_group) < n * 0.20 or len(low_group) <= 1 or len(high_group) <= 1 or cv < 0.35:
                    is_single_speaker = True

        if is_single_speaker:
            # 1-Speaker Mode: Output continuous text without speaker tags
            for seg in segments_list:
                corrected_text = post_process_text(seg.text.strip())
                if corrected_text:
                    result.append(corrected_text)
        else:
            # 2-Speakers Mode: Apply median-threshold based speaker labeling
            sorted_rms = sorted(rms_values)
            median_rms = sorted_rms[len(sorted_rms) // 2]
            
            last_speaker = None
            for seg, rms in zip(segments_list, rms_values):
                speaker = "교사" if rms >= median_rms else "학생"
                corrected_text = post_process_text(seg.text.strip())
                if not corrected_text:
                    continue

                if speaker == last_speaker and len(result) > 0:
                    result[-1] = f"{result[-1]} {corrected_text}"
                else:
                    result.append(f"\n[{speaker}] {corrected_text}")
                    last_speaker = speaker
    else:
        # Fallback to simple concatenation if WAV analysis fails
        for seg in segments_list:
            corrected_text = post_process_text(seg.text.strip())
            if corrected_text:
                result.append(corrected_text)

    return "\n".join(result).strip()
