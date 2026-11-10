#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPT-SoVITS v2 Ultra-Fast Streaming Server (Mode 3 & Sentence Stream)
Features:
  - V2 Model Architecture with Native HiFi-GAN (Clean acoustic output)
  - WebUI Quality Match: Uses 'cut1' (凑四句一切) & top_k=15, top_p=1.0, temperature=1.0
  - Advanced Medical & Colloquial Text Normalization:
      * Converts numbers to Chinese characters (128 -> 一百二十八, 82 -> 八十二)
      * Replaces abrupt dunhao '、' with '，' to eliminate harsh pitch breaks and pauses
  - Zero-Click Boundary Smoothing: Raw continuous int16 streaming (No per-chunk peak distortion)
  - Token-Level Chunk Streaming (Mode 3, min_chunk_length=24, overlap_length=2) for sub-second TTFA
  - Optional WebUI Sentence Streaming (streaming_mode=0 / 'sentence')
  - Strict OpenAI-compatible API & Go backend headers (Content-Type: audio/pcm, X-Stream-Mode: streaming)
  - Dedicated Nurse voice pre-warmed & cached in GPU VRAM
"""

import os
import sys
import re
import time
from datetime import datetime
import struct
import argparse
import asyncio
import threading
from typing import Optional, List, Dict, Any, Union
import numpy as np
from scipy import signal
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from functools import partialmethod

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "GPT_SoVITS"))


class CleanStdoutFilter:
    """Filter out internal noisy debug prints and eliminate blank lines."""
    def __init__(self, original_stdout):
        self.orig = original_stdout
        self.buffer = ""

    def write(self, s):
        # Discard in-place carriage returns (\r) completely for clean API logs
        if "\r" in s and "\n" not in s:
            return

        self.buffer += s
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            st = line.strip()
            if not st:
                continue

            # Allow our formatted server logs, uvicorn access logs, banners, and real errors
            allow = (
                st.startswith("[") or
                st.startswith("INFO:") or
                st.startswith("WARNING:") or
                st.startswith("ERROR:") or
                st.startswith("=") or
                st.startswith("Starting") or
                st.startswith("Port:") or
                st.startswith("Endpoints:") or
                "Traceback" in st or
                st.startswith("File ") or
                "Error" in st
            )
            if not allow:
                continue

            self.orig.write(line + "\n")

    def flush(self):
        if self.buffer.strip():
            self.orig.write(self.buffer + "\n")
        self.buffer = ""
        self.orig.flush()

    def isatty(self):
        return getattr(self.orig, "isatty", lambda: False)()

    def fileno(self):
        return self.orig.fileno()

    def __getattr__(self, name):
        return getattr(self.orig, name)


# Attach clean logger unless VERBOSE is explicitly requested
if not os.environ.get("TTS_VERBOSE"):
    sys.stdout = CleanStdoutFilter(sys.stdout)

# Completely disable tqdm progress bars for clean, deterministic API logging
os.environ["TQDM_DISABLE"] = "1"
try:
    import tqdm
    import tqdm.std
    tqdm.tqdm.__init__ = partialmethod(tqdm.tqdm.__init__, disable=True)
    tqdm.std.tqdm.__init__ = partialmethod(tqdm.std.tqdm.__init__, disable=True)
except Exception:
    pass

try:
    from tools.portable_runtime import activate as activate_portable_runtime
    activate_portable_runtime(change_cwd=True)
except Exception:
    pass

try:
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
except ImportError as e:
    TTS = None
    TTS_Config = None

device = "cuda" if torch.cuda.is_available() else "cpu"
current_pipeline = None

gpu_lock = threading.Lock()
model_lock = asyncio.Lock()

# 4th-order Butterworth low-pass filter at 7500 Hz (fs=32000 Hz)
# Eliminates 14kHz-16kHz Nyquist ringing from vocoder, preventing DAC "pop/click" and Go linear downsampler aliasing
ANTI_ALIASING_SOS = signal.butter(4, 7500, 'lowpass', fs=32000, output='sos')

VOICES_DIR = os.path.join(BASE_DIR, "voices")
os.makedirs(VOICES_DIR, exist_ok=True)

# Default reference audio (Female Nurse)
NURSE_AUDIO = os.path.join(VOICES_DIR, "female_nurse.wav")
if os.path.exists(NURSE_AUDIO):
    DEFAULT_SAMPLE_PROMPT = NURSE_AUDIO
else:
    DEFAULT_SAMPLE_PROMPT = os.path.join(BASE_DIR, "sample_prompt.wav")
    if not os.path.exists(DEFAULT_SAMPLE_PROMPT):
        DEFAULT_SAMPLE_PROMPT = os.path.join(BASE_DIR, "test_end2end.wav")

VOICE_CONFIGS: Dict[str, Dict[str, str]] = {}
PRESET_VOICES: List[Dict[str, str]] = []


def scan_voices():
    """Scan voices directory and register Nurse voice configs."""
    global PRESET_VOICES, VOICE_CONFIGS
    VOICE_CONFIGS = {}
    new_voices = []

    nurse_audio = os.path.join(VOICES_DIR, "female_nurse.wav")
    nurse_txt = os.path.join(VOICES_DIR, "female_nurse.txt")
    if not os.path.exists(nurse_audio) and os.path.exists(os.path.join(VOICES_DIR, "nurse.wav")):
        nurse_audio = os.path.join(VOICES_DIR, "nurse.wav")
        nurse_txt = os.path.join(VOICES_DIR, "nurse.txt")

    p_text = "请您慢点坐下，深呼吸放松一下。有什么不舒服的地方，随时都可以告诉我。"
    if os.path.exists(nurse_txt):
        try:
            with open(nurse_txt, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    p_text = content
        except Exception:
            pass

    nurse_cfg = {
        "ref_audio_path": nurse_audio,
        "prompt_text": p_text,
        "prompt_lang": "zh",
    }
    VOICE_CONFIGS["female_nurse"] = nurse_cfg
    VOICE_CONFIGS["nurse"] = nurse_cfg
    VOICE_CONFIGS["default"] = nurse_cfg

    # Scan any extra audio if placed in voices dir
    if os.path.exists(VOICES_DIR):
        for fname in os.listdir(VOICES_DIR):
            if fname.lower().endswith((".wav", ".mp3", ".flac")):
                v_id = os.path.splitext(fname)[0].lower()
                audio_path = os.path.join(VOICES_DIR, fname)
                txt_path = os.path.join(VOICES_DIR, f"{v_id}.txt")
                txt = ""
                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, "r", encoding="utf-8") as f:
                            txt = f.read().strip()
                    except Exception:
                        pass
                VOICE_CONFIGS[v_id] = {
                    "ref_audio_path": audio_path,
                    "prompt_text": txt or p_text,
                    "prompt_lang": "zh",
                }

    # Standard OpenAI Voice Metadata for Go backend
    new_voices.append({
        "id": "female_nurse",
        "name": "Nurse (贴心护士 - 轻柔关怀用药提醒)",
        "locale": "zh-CN",
        "gender": "female",
        "description": "温柔耐心的护士音色，贴心关怀与健康用药提醒",
    })

    PRESET_VOICES = new_voices


def num_to_zh(match_or_str) -> str:
    """Convert Arabic numeral to natural spoken Chinese words."""
    s = match_or_str.group(0) if hasattr(match_or_str, "group") else str(match_or_str)
    digits = "零一二三四五六七八九"
    if "." in s:
        parts = s.split(".")
        try:
            int_part = int(parts[0])
            dec_str = "".join(digits[int(d)] for d in parts[1] if d.isdigit())
            if int_part < 10:
                zh_int = digits[int_part]
            elif int_part < 20:
                zh_int = "十" + (digits[int_part % 10] if int_part % 10 != 0 else "")
            elif int_part < 100:
                zh_int = digits[int_part // 10] + "十" + (digits[int_part % 10] if int_part % 10 != 0 else "")
            else:
                zh_int = str(int_part)
            return zh_int + "点" + dec_str
        except Exception:
            return s
    try:
        val = int(s)
        if val == 0:
            return "零"
        if val < 10:
            return digits[val]
        if val < 20:
            return "十" + (digits[val % 10] if val % 10 != 0 else "")
        if val < 100:
            return digits[val // 10] + "十" + (digits[val % 10] if val % 10 != 0 else "")
        if val < 1000:
            res = digits[val // 100] + "百"
            rem = val % 100
            if rem == 0:
                return res
            if rem < 10:
                return res + "零" + digits[rem]
            if rem < 20:
                return res + "一十" + (digits[rem % 10] if rem % 10 != 0 else "")
            return res + digits[rem // 10] + "十" + (digits[rem % 10] if rem % 10 != 0 else "")
        if val < 10000:
            res = digits[val // 1000] + "千"
            rem = val % 1000
            if rem == 0:
                return res
            if rem < 100:
                return res + "零" + num_to_zh(str(rem))
            return res + num_to_zh(str(rem))
        return s
    except Exception:
        return s


def normalize_speech_text(text: str) -> str:
    """Normalize medical indicators, units, and punctuation for natural speech."""
    if not text:
        return ""

    # Strip leading ellipses or filler
    text = re.sub(r"^[…\.。\s]+", "", text)
    # Convert exclamation marks to periods to avoid emotional runaway
    text = text.replace("！", "。").replace("!", "。")
    text = text.replace("~", "，").replace("～", "，")

    # 1. Blood pressure: 128/82 mmHg -> 高压 128、低压 82
    def bp_repl(match):
        sys_p, dia_p = match.group(1), match.group(2)
        return f"高压 {sys_p}，低压 {dia_p}"
    text = re.sub(r"(\d{2,3})\s*/\s*(\d{2,3})", bp_repl, text)

    # 2. mmHg -> 毫米汞柱
    text = re.sub(r"\bmmHg\b", "毫米汞柱", text, flags=re.I)

    # 3. Blood sugar: 5.6 mmol/L -> 5.6 毫摩尔每升
    text = re.sub(r"(\d+(?:\.\d+)?)\s*mmol/L", r"\1 毫摩尔每升", text, flags=re.I)

    # 4. Heart rate: 72 bpm -> 72 次每分钟
    text = re.sub(r"(\d+)\s*bpm", r"\1 次每分钟", text, flags=re.I)

    # 5. SpO2 Blood oxygen: 98% -> 百分之 98
    text = re.sub(r"(?:SpO2|血氧)\s*[:：]?\s*(\d+)%", r"血氧百分之 \1", text, flags=re.I)

    # 6. Temperature: 36.5 ℃ -> 36.5 摄氏度
    text = re.sub(r"(\d+(?:\.\d+)?)\s*[℃]", r"\1 摄氏度", text)

    # 7. Replace dunhao '、' with gentle comma '，' to eliminate sudden pause & pitch drop
    text = text.replace("、", "，")

    # 8. Erhua smoothing: eliminate harsh tone collisions (e.g. shù'ér -> shù, yīhuì'ér -> yīhuì)
    erhua_map = {
        "具体数儿": "具体数值",
        "数儿": "数",
        "一会儿": "一会",
        "点儿": "点",
        "空儿": "空",
        "事儿": "事",
        "话儿": "话",
        "活儿": "活",
        "伴儿": "伴",
        "声儿": "声",
        "个儿": "个",
        "样儿": "样",
    }
    for k, v in erhua_map.items():
        text = text.replace(k, v)

    # 9. Convert numerals to Chinese words for natural human tone (prevents digit reading & pitch drops)
    text = re.sub(r"\d+(?:\.\d+)?", num_to_zh, text)

    # 10. Clean up redundant spaces between Chinese characters
    text = re.sub(r"\s+", "", text)

    return text.strip()


def create_streaming_wav_header(sr=32000, num_channels=1, bits_per_sample=16) -> bytes:
    byte_rate = sr * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 0x7fffffff, b"WAVE",
        b"fmt ", 16, 1, num_channels, sr, byte_rate, block_align, bits_per_sample,
        b"data", 0x7fffffff
    )


def load_model():
    """Load GPT-SoVITS V2 Model with CUDA FP16 for native Token-Chunk Streaming."""
    global current_pipeline
    with gpu_lock:
        if current_pipeline is not None:
            return current_pipeline

        if TTS is None:
            raise RuntimeError("GPT-SoVITS 依赖未就绪，请检查环境")

        # Explicitly configure V2 pretrained model paths on CUDA FP16
        v2_config_dict = {
            "custom": {
                "version": "v2",
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "is_half": torch.cuda.is_available(),
                "t2s_weights_path": os.path.join(BASE_DIR, "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"),
                "vits_weights_path": os.path.join(BASE_DIR, "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"),
                "cnhuhbert_base_path": os.path.join(BASE_DIR, "GPT_SoVITS/pretrained_models/chinese-hubert-base"),
                "bert_base_path": os.path.join(BASE_DIR, "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"),
            }
        }

        print("=" * 60)
        print("  [GPT-SoVITS Fast Server] Loading V2 Pretrained Weights...")
        print(f"  Device:    {device} (FP16: {torch.cuda.is_available()})")
        print("  Streaming: Token-Level Chunk Streaming (Mode 3 Enabled)")
        print("=" * 60)

        tts_config = TTS_Config(v2_config_dict)
        current_pipeline = TTS(tts_config)

        scan_voices()

        # Warmup pipeline with Nurse reference audio to prime GPU CUDA kernels & prompt cache
        print(f"[GPT-SoVITS Fast Server] Pre-warming pipeline with Nurse voice...")
        try:
            nurse_cfg = VOICE_CONFIGS.get("female_nurse") or VOICE_CONFIGS.get("default")
            warm_req = {
                "text": "系统启动预热。",
                "text_lang": "zh",
                "ref_audio_path": nurse_cfg["ref_audio_path"],
                "prompt_text": nurse_cfg.get("prompt_text", ""),
                "prompt_lang": "zh",
                "top_k": 15,
                "top_p": 1.0,
                "temperature": 1.0,
                "text_split_method": "cut_smart",
                "streaming_mode": False,
                "return_fragment": True,
                "fragment_interval": 0.08,
            }
            res = current_pipeline.run(warm_req)
            if hasattr(res, "__iter__") and not isinstance(res, tuple):
                for _ in res:
                    pass
            print(f"[GPT-SoVITS Fast Server] Pre-warming completed! Ready for sub-second TTFA.")
        except Exception as we:
            print(f"[GPT-SoVITS Fast Server] Warmup warning: {we}")

        return current_pipeline


app = FastAPI(
    title="GPT-SoVITS v2 Ultra-Fast Streaming Server",
    description="Sub-second low-latency streaming TTS server for Go backend integration.",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SpeechRequest(BaseModel):
    model: Optional[str] = Field("gpt-sovits", description="Model name")
    input: Optional[str] = Field(None, description="The text to generate audio for")
    text: Optional[str] = Field(None, description="Alias for input")
    voice: Optional[str] = Field("default", description="Voice/speaker ID")
    speaker: Optional[str] = Field(None, description="Alias for voice")
    response_format: Optional[str] = Field("pcm", description="Audio format: pcm, wav")
    language: Optional[str] = Field("zh", description="Target language: zh, en, ja, ko, yue")
    speed: Optional[float] = Field(1.0, description="Playback speed multiplier")
    temperature: Optional[float] = Field(1.0, description="Sampling temperature (WebUI natural default 1.0)")
    top_p: Optional[float] = Field(1.0, description="Top-p sampling")
    top_k: Optional[int] = Field(15, description="Top-k sampling")
    stream: Optional[bool] = Field(True, description="Whether to stream response")
    text_split_method: Optional[str] = Field("cut_smart", description="Text split method: cut_smart")
    streaming_mode: Optional[Any] = Field(False, description="Streaming mode: False")
    min_chunk_length: Optional[int] = Field(24, description="Min chunk token length for Mode 3")
    overlap_length: Optional[int] = Field(2, description="Token overlap length")
    batch_size: Optional[int] = Field(1, description="Batch size")
    fragment_interval: Optional[float] = Field(0.08, description="Pause between fragments in seconds (80ms natural breath)")


@app.get("/health")
@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "engine": "GPT-SoVITS v2 (Ultra-Fast Streaming Mode 3 / Sentence Stream)",
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "default_sample_rate": 32000,
        "expected_ttfa_ms": "350-600",
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "gpt-sovits", "name": "GPT-SoVITS v2 Ultra-Fast (Token-Chunk & Sentence Streaming)", "description": "毫秒级首包极速流式语音合成"},
        ]
    }


@app.get("/v1/audio/voices")
@app.get("/v1/voices")
async def list_voices():
    scan_voices()
    return {
        "voices": PRESET_VOICES,
        "data": PRESET_VOICES,
    }


@app.post("/v1/audio/speech")
async def generate_speech(req: SpeechRequest, request: Request):
    content = req.input or req.text
    if not content or not content.strip():
        raise HTTPException(status_code=400, detail="Field 'input' or 'text' must not be empty.")

    # Medical & colloquial text normalization for natural speech
    content = normalize_speech_text(content)

    req_id = request.headers.get("x-request-id") or request.headers.get("x-tts-request-id") or f"tts-{int(time.time()*1000)}"
    target_voice = (req.voice or req.speaker or "default").strip().lower()
    fmt = req.response_format.strip().lower() if req.response_format else "pcm"

    if current_pipeline is None:
        load_model()

    # Determine reference voice (defaults and fallbacks to Nurse)
    fallback_cfg = VOICE_CONFIGS.get("female_nurse") or VOICE_CONFIGS.get("nurse") or VOICE_CONFIGS.get("default")
    v_info = VOICE_CONFIGS.get(target_voice, fallback_cfg)
    ref_audio_path = v_info["ref_audio_path"]
    prompt_text = v_info.get("prompt_text", "")
    prompt_lang = v_info.get("prompt_lang", "zh")

    text_preview = content.strip().replace("\n", " ")
    if len(text_preview) > 35:
        text_preview = text_preview[:35] + "..."

    client_host = request.client.host if request.client else "unknown"
    now_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    temp_val = float(req.temperature) if req.temperature is not None else 1.0
    split_method = req.text_split_method or "cut_smart"
    if split_method in ("cut5", "cut_smart"):
        split_method = "cut_smart"
    frag_interval = float(req.fragment_interval if req.fragment_interval is not None else 0.08)
    print(f"[{now_str}] [Fast-Req] [{req_id}] Client: {client_host} | Voice: {target_voice} | Split: {split_method} | Temp: {temp_val} | FragGap: {frag_interval*1000:.0f}ms | Fmt: {fmt} | Text: \"{text_preview}\"")

    async def audio_stream_generator():
        t0 = time.time()
        first_chunk_logged = False
        total_chunks = 0
        total_bytes = 0

        # GPT-SoVITS v2 default output sample rate is 32000 Hz
        sample_rate = 32000
        chunk_size_bytes = 3840  # 3840 bytes = 60ms Opus frame equivalent at 32kHz 16-bit mono

        if fmt == "wav":
            yield create_streaming_wav_header(sr=sample_rate, num_channels=1, bits_per_sample=16)

        q = asyncio.Queue()
        loop = asyncio.get_running_loop()
        stop_event = threading.Event()

        def _inference_thread():
            try:
                with gpu_lock:
                    if stop_event.is_set():
                        return

                    # Only use Token Chunk Streaming if explicitly requested ("3" or "chunk")
                    # Default uses WebUI Native Sentence Streaming with cut5 for zero-sola pristine quality & 400ms TTFA
                    is_token_streaming = str(req.streaming_mode).lower() in ("3", "chunk")

                    if is_token_streaming:
                        # Fast Mode 3: Token Chunk Streaming (sub-second TTFA, 24 tokens)
                        inputs = {
                            "text": content,
                            "text_lang": req.language or "zh",
                            "ref_audio_path": ref_audio_path,
                            "prompt_text": prompt_text,
                            "prompt_lang": prompt_lang,
                            "top_k": int(req.top_k or 15),
                            "top_p": float(req.top_p or 1.0),
                            "temperature": temp_val,
                            "speed_factor": float(req.speed or 1.0),
                            "text_split_method": split_method,
                            "batch_size": 1,
                            "streaming_mode": 3,
                            "fixed_length_chunk": True,
                            "min_chunk_length": int(req.min_chunk_length or 24),
                            "overlap_length": int(req.overlap_length or 2),
                            "parallel_infer": False,
                        }
                    else:
                        # WebUI Mode: Sentence Streaming (zero SOLA, maximum audio quality)
                        inputs = {
                            "text": content,
                            "text_lang": req.language or "zh",
                            "ref_audio_path": ref_audio_path,
                            "prompt_text": prompt_text,
                            "prompt_lang": prompt_lang,
                            "top_k": int(req.top_k or 15),
                            "top_p": float(req.top_p or 1.0),
                            "temperature": temp_val,
                            "speed_factor": float(req.speed or 1.0),
                            "text_split_method": split_method,
                            "batch_size": 1,
                            "streaming_mode": False,
                            "return_fragment": True,
                            "parallel_infer": True,
                            "fragment_interval": frag_interval,
                        }

                    res = current_pipeline.run(inputs)
                    for sr, audio_chunk in res:
                        if stop_event.is_set():
                            break
                        if audio_chunk is None:
                            continue
                        if isinstance(audio_chunk, torch.Tensor):
                            audio_chunk = audio_chunk.detach().cpu().numpy()
                        audio_chunk = np.squeeze(audio_chunk)
                        if isinstance(audio_chunk, np.ndarray) and audio_chunk.size == 0:
                            continue

                        # WebUI Native Clean int16 Audio (No IIR phase distortion, zero artificial noise)
                        if audio_chunk.dtype == np.int16:
                            chunk_int16 = audio_chunk
                        else:
                            chunk_int16 = (np.clip(audio_chunk, -1.0, 1.0) * 32767).astype(np.int16)

                        loop.call_soon_threadsafe(q.put_nowait, chunk_int16.tobytes())
            except Exception as exc:
                print(f"[{req_id}] Inference error: {exc}")
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        infer_thread = threading.Thread(target=_inference_thread, daemon=True)

        try:
            async with model_lock:
                infer_thread.start()
                while True:
                    if await request.is_disconnected():
                        print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [Fast Cancel] [{req_id}] Client disconnected, aborting inference")
                        stop_event.set()
                        if current_pipeline:
                            current_pipeline.stop()
                        break

                    try:
                        raw_pcm = await asyncio.wait_for(q.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue

                    if raw_pcm is None:
                        break

                    if not first_chunk_logged:
                        ttfa = (time.time() - t0) * 1000
                        print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [⚡ Fast First-Chunk] [{req_id}] TTFA: {ttfa:.1f} ms!")
                        first_chunk_logged = True

                    # Stream out PCM bytes in chunk_size_bytes
                    for offset in range(0, len(raw_pcm), chunk_size_bytes):
                        sub_chunk = raw_pcm[offset:offset + chunk_size_bytes]
                        total_chunks += 1
                        total_bytes += len(sub_chunk)
                        yield sub_chunk

            total_time = time.time() - t0
            bytes_per_sec = sample_rate * 2
            audio_dur = total_bytes / float(bytes_per_sec)
            rtf = total_time / max(audio_dur, 0.001)
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [Fast Done] [{req_id}] Total: {total_time:.2f}s | Audio: {audio_dur:.2f}s | Chunks: {total_chunks} | RTF: {rtf:.2f}")

        except BaseException as e:
            if not isinstance(e, (asyncio.CancelledError, GeneratorExit)):
                print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [Fast Stream Abort] [{req_id}]: {e}")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [Fast Stream Cancelled] [{req_id}]")
            raise
        finally:
            stop_event.set()

    resp_headers = {
        "X-Request-ID": req_id,
        "X-Model": "gpt-sovits-v2-fast",
        "X-Speaker": target_voice,
    }

    if fmt == "pcm":
        media_type = "audio/pcm"
        resp_headers.update({
            "Content-Type": "audio/pcm",
            "X-Stream-Mode": "streaming",
            "X-Audio-Format": "pcm_s16le",
            "X-Audio-Channels": "1",
            "X-Audio-Bit-Depth": "16",
            "X-Audio-Byte-Order": "little-endian",
            "X-Audio-Signedness": "signed",
            "X-Audio-Encoding": "raw_pcm_s16le",
            "X-Audio-Sample-Rate": "32000",
        })
    elif fmt == "wav":
        media_type = "audio/wav"
        resp_headers.update({
            "Content-Type": "audio/wav",
            "X-Stream-Mode": "streaming",
            "X-Audio-Sample-Rate": "32000",
        })
    else:
        media_type = "audio/octet-stream"

    return StreamingResponse(
        audio_stream_generator(),
        media_type=media_type,
        headers=resp_headers,
    )


def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS v2 Ultra-Fast Streaming Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=50045, help="Port to listen on")
    parser.add_argument("--verbose", action="store_true", help="Enable raw debug logs and progress bars")
    args = parser.parse_args()

    if args.verbose:
        sys.stdout = sys.__stdout__
        os.environ.pop("TQDM_DISABLE", None)

    print("=" * 60)
    print("  Starting GPT-SoVITS v2 Ultra-Fast Streaming Server (Mode 3 & Sentence)")
    print(f"  Port:      http://localhost:{args.port}")
    print(f"  Endpoints: POST http://localhost:{args.port}/v1/audio/speech")
    print(f"             GET  http://localhost:{args.port}/v1/audio/voices")
    print(f"             GET  http://localhost:{args.port}/v1/models")
    print("=" * 60)

    load_model()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
