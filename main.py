import os
import io
import time
import tempfile
import requests
import streamlit as st
from typing import Optional, Tuple, List

# Audio/Video
from moviepy.editor import VideoFileClip  # requires ffmpeg installed
from faster_whisper import WhisperModel   # local transcription

# ---------- Config ----------
LMSTUDIO_API_URL = "http://localhost:1234/v1/chat/completions"
DEFAULT_MODEL = "gemma-3-12b-it"  # LM Studio model id
DEFAULT_LANGUAGE = "ru"           # force language for whisper
WHISPER_MODEL_SIZE = "large-v3"   # faster-whisper local model

st.set_page_config(page_title="🎙️Транскрибировать аудио", page_icon="🗒️", layout="wide")

# ---------- Utils ----------

def save_uploaded_file(uploaded_file) -> str:
    """Persist uploaded file to a temp path and return its path."""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name

def extract_audio_from_video(video_path: str) -> str:
    """Extract audio (wav) from a video using moviepy/ffmpeg."""
    out_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    clip = VideoFileClip(video_path)
    try:
        clip.audio.write_audiofile(out_wav, verbose=False, logger=None)
    finally:
        clip.close()
    return out_wav

def transcribe_audio_whisper(audio_path: str,
                             language: str = "ru",
                             beam_size: int = 5,
                             vad_filter: bool = True,
                             device: Optional[str] = None) -> Tuple[str, List[Tuple[float, float, str]]]:
    """
    Transcribe using faster-whisper.
    Returns full text and segments list of (start, end, text).
    """
    # device: None -> auto; options: "cuda", "cpu"
    model = WhisperModel(WHISPER_MODEL_SIZE, device=device or "auto", compute_type="auto")
    segments_iter, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        vad_parameters=dict(min_silence_duration_ms=300)
    )
    segments = []
    full_text_chunks = []
    for seg in segments_iter:
        segments.append((seg.start, seg.end, seg.text))
        full_text_chunks.append(seg.text.strip())
    full_text = "\n".join([t for t in full_text_chunks if t])
    return full_text.strip(), segments

def call_lmstudio_chat(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1200,
    timeout: float = 60.0
) -> str:
    """
    Call LM Studio’s OpenAI-compatible chat endpoint.
    """
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    }
    resp = requests.post(LMSTUDIO_API_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return str(data)

def make_protocol_prompt() -> str:
    return (
        "Создай текст в стиле протокольных решений, разделив его по ролям говорящих. "
        "Структурируй как официальный протокол собрания: укажи дату (если в тексте есть), "
        "список участников (по репликам, если явно не указаны — предложи роли), повестку (если выводима), "
        "и по каждому вопросу — краткие записи, решения, поручения и сроки. "
        "Оформи разделы: «Участники», «Повестка», «Ход заседания», «Принятые решения», «Поручения». "
        "Не добавляй комментарии от себя, не пересказ, а чёткий протокол."
    )

def format_segments_table(segments: List[Tuple[float, float, str]]) -> str:
    lines = ["Начало\tКонец\tТекст"]
    for s, e, t in segments:
        lines.append(f"{s:.2f}\t{e:.2f}\t{t.strip()}")
    return "\n".join(lines)

def download_button_bytes(text: str, filename: str, label: str):
    return st.download_button(
        label=label,
        data=text.encode("utf-8"),
        file_name=filename,
        mime="text/plain"
    )

# ---------- UI ----------

st.title("🎙️Транскрибировать аудио")

with st.sidebar:
    st.header("⚙️ Настройки")
    language = st.selectbox("Язык распознавания (Whisper)", ["ru", "en", "auto"], index=0)
    temperature = st.slider("Temperature (LM)", 0.0, 1.5, 0.7, 0.1)
    max_tokens = st.slider("Max tokens (LM)", 128, 16000, 12000, 64)
    beam_size = st.slider("Beam size (Whisper)", 1, 10, 5, 1)
    vad_filter = st.checkbox("VAD фильтр (тишина)", value=True)
    device_choice = st.selectbox("Устройство", ["auto", "cuda", "cpu"], index=0)
    model_name = st.text_input("LM Studio model", value=DEFAULT_MODEL)
    st.markdown("---")
    st.markdown("**Требования:** ffmpeg, `faster-whisper`, `moviepy`, `streamlit`. LM Studio — запущен с REST API.")

# 👇 добавил поддержку .m4a
uploaded = st.file_uploader(
    "Загрузите аудио/видео (.mp3, .mp4, .m4a)",
    type=["mp3", "mp4", "m4a"]
)

go = st.button("Запустить обработку 🚀", type="primary", disabled=uploaded is None)

if uploaded and go:
    try:
        with st.status("Подготовка файлов…", expanded=False) as status:
            src_path = save_uploaded_file(uploaded)
            ext = os.path.splitext(src_path)[1].lower()
            status.update(label="Определение типа…")

            # видео — только .mp4 (m4a считаем аудио)
            is_video = ext == ".mp4"

            if is_video:
                status.update(label="Извлекаю аудио из видео…")
                audio_path = extract_audio_from_video(src_path)
            else:
                # mp3 и m4a идут напрямую в whisper
                audio_path = src_path

            status.update(label="Транскрибирую...")
            with st.spinner("Whisper работает… это может занять время ⏳"):
                transcript, segments = transcribe_audio_whisper(
                    audio_path=audio_path,
                    language=None if language == "auto" else language,
                    beam_size=beam_size,
                    vad_filter=vad_filter,
                    device=None if device_choice == "auto" else device_choice
                )

            if not transcript.strip():
                st.error("Не получилось распознать речь. Проверьте качество аудио/язык.")
                status.update(label="Ошибка транскрибации", state="error")
            else:
                status.update(label="Форматирую в протокол…")
                system_msg = "Ты профессиональный секретарь заседаний. Пиши кратко, официально, структурировано."
                user_msg = f"{make_protocol_prompt()}\n\nНиже — расшифровка стенограммы:\n\n{transcript}"

                with st.spinner("Форматирую текст…"):
                    protocol_text = call_lmstudio_chat(
                        system_prompt=system_msg,
                        user_prompt=user_msg,
                        model=model_name,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )

                status.update(label="Готово ✅", state="complete")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📝 Расшифровка")
            st.text_area("Transcript", transcript, height=360)
            download_button_bytes(transcript, "transcript.txt", "Скачать transcript.txt")

            with st.expander("Сегменты (время начала/конца)"):
                st.text(format_segments_table(segments))

        with col2:
            st.subheader("📑 Протокол")
            st.text_area("Protocol", protocol_text, height=360)
            download_button_bytes(protocol_text, "protocol.txt", "Скачать protocol.txt")

        st.success("Пакет доставлен. Если нужно — добавим выгрузку в DOCX/Markdown/JSON.")
    except requests.exceptions.ConnectionError:
        st.error("Не удалось подключиться к LM Studio API. Убедись, что LM Studio запущен и REST API включён (http://localhost:1234).")
    except Exception as e:
        st.exception(e)
