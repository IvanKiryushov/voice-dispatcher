"""
Voice Sync Distributor — локальный пайплайн обработки аудио- и видеозаписей.
Сканирует входящую директорию (например, синхронизированную папку Google Drive),
расшифровывает файлы через faster-whisper, опционально раскладывает Markdown-заметки
по категориям (MEETINGS, CONTENT, WORK, INBOX) и переносит обработанные исходники.
"""

import datetime
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Поддержка запуска и как `python src/voice_sync_distributor.py`, и как `python -m src.voice_sync_distributor`
try:
    from transcriber_service import TranscriberService
except ImportError:
    from src.transcriber_service import TranscriberService

# Поддерживаемые форматы аудио и видео (faster-whisper читает аудиодорожки через PyAV/ffmpeg)
SUPPORTED_EXTENSIONS = {
    ".m4a",
    ".mp4",
    ".mp3",
    ".wav",
    ".ogg",
    ".mkv",
    ".mov",
    ".webm",
}

# Конфигурация из переменных окружения (.env)
INPUT_AUDIO_DIR = os.getenv("INPUT_AUDIO_DIR", "./data/input")
PROCESSED_AUDIO_DIR = os.getenv("PROCESSED_AUDIO_DIR", "./data/processed")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./data/output")
ARCHIVE_AUDIO_DIR = os.getenv("ARCHIVE_AUDIO_DIR", "").strip()

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "ru")
WHISPER_INITIAL_PROMPT = os.getenv("WHISPER_INITIAL_PROMPT") or None
ENABLE_CLASSIFICATION = os.getenv("ENABLE_CLASSIFICATION", "true").lower() in ("1", "true", "yes")

# Соответствие категорий и подпапок в OUTPUT_DIR
CATEGORY_FOLDERS = {
    "MEETINGS": "Meetings",
    "CONTENT": "Content",
    "WORK": "Work",
    "INBOX": "Inbox",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VoiceSyncDistributor")

_transcriber_service: Optional[TranscriberService] = None


def get_transcriber() -> Optional[TranscriberService]:
    """Ленивая инициализация сервиса транскрибации с параметрами из .env."""
    global _transcriber_service
    if _transcriber_service is None:
        try:
            _transcriber_service = TranscriberService(
                model_size=WHISPER_MODEL,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
                language=WHISPER_LANGUAGE,
                initial_prompt=WHISPER_INITIAL_PROMPT,
            )
        except Exception as e:
            logger.error(f"Не удалось инициализировать TranscriberService: {e}")
    return _transcriber_service


def classify_content(filename: str, text: str) -> dict:
    """
    Пример простой классификации по ключевым словам в имени файла и тексте.
    Замените списки слов под свои задачи или отключите через ENABLE_CLASSIFICATION=false.

    Возвращает: {"domain": "MEETINGS"|"CONTENT"|"WORK"|"INBOX", "title": str, "sub_type": str}
    """
    if not ENABLE_CLASSIFICATION:
        return {
            "domain": "INBOX",
            "title": "Голосовая заметка",
            "sub_type": "Raw Note",
        }

    fn_lower = filename.lower()
    text_lower = text.lower()

    # 1. Встречи и созвоны
    meeting_keywords = [
        "встреч", "созвон", "планерк", "митинг", "интервью",
        "договорились", "обсудили", "клиент", "заказчик",
    ]
    if any(k in fn_lower for k in ["meet", "call", "zoom", "interview"]) or any(
        k in text_lower for k in meeting_keywords
    ):
        return {
            "domain": "MEETINGS",
            "title": "Запись встречи / созвона",
            "sub_type": "Meeting",
        }

    # 2. Идеи для контента
    content_keywords = [
        "контент", "пост", "стать", "сценарий", "выпуск",
        "подкаст", "рилс", "видео", "блог", "канал",
    ]
    if any(k in fn_lower for k in ["content", "post", "video"]) or any(
        k in text_lower for k in content_keywords
    ):
        return {
            "domain": "CONTENT",
            "title": "Идея для контента",
            "sub_type": "Content Draft",
        }

    # 3. Рабочие задачи и проекты
    work_keywords = [
        "задач", "проект", "дедлайн", "релиз", "спринт",
        "баг", "фича", "документац", "отчет", "техническое задание",
    ]
    if any(k in fn_lower for k in ["work", "task", "proj"]) or any(
        k in text_lower for k in work_keywords
    ):
        return {
            "domain": "WORK",
            "title": "Рабочая заметка",
            "sub_type": "Project Task",
        }

    # 4. Входящие по умолчанию
    return {
        "domain": "INBOX",
        "title": "Голосовая заметка",
        "sub_type": "Raw Note",
    }


def parse_date_from_filename(filename: str):
    """
    Извлекает дату и время из имени файла (форматы YYYYMMDD_HHMMSS или YYYY-MM-DD_HH-MM-SS).
    Если в имени даты нет, использует текущее локальное время.
    """
    match = re.search(
        r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})[ _T-](\d{2})[-_:]?(\d{2})[-_:]?(\d{2})",
        filename,
    )
    if match:
        year, month, day, h, m, s = match.groups()
        date_iso = f"{year}-{month}-{day}"
        time_iso = f"{h}-{m}-{s}"
        return date_iso, time_iso, f"{date_iso}_{time_iso}"

    now = datetime.datetime.now()
    date_iso = now.strftime("%Y-%m-%d")
    time_iso = now.strftime("%H-%M-%S")
    return date_iso, time_iso, f"{date_iso}_{time_iso}"


def find_existing_transcript(base_name: str) -> str:
    """Проверяет, есть ли уже сохранённый .txt транскрипт в подпапках OUTPUT_DIR."""
    for folder in CATEGORY_FOLDERS.values():
        candidate = os.path.join(OUTPUT_DIR, folder, f"{base_name}.txt")
        if os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    text = f.read()
                if text.strip():
                    logger.info(f"Найден существующий транскрипт: {candidate}")
                    return text
            except Exception as e:
                logger.warning(f"Не удалось прочитать {candidate}: {e}")
    return ""


def process_audio_file(audio_path: str) -> bool:
    """Обрабатывает один файл: расшифровка -> классификация -> сохранение Markdown -> перенос в processed."""
    filename = os.path.basename(audio_path)
    base_name = os.path.splitext(filename)[0]
    logger.info(f"==> Обработка файла: {filename}")

    date_iso, _time_iso, timestamp = parse_date_from_filename(filename)

    # 1. Поиск готового транскрипта или запуск faster-whisper
    text = find_existing_transcript(base_name)
    if not text.strip():
        logger.info("Запуск локальной транскрибации через faster-whisper...")
        transcriber = get_transcriber()
        if transcriber is None:
            logger.error(f"Пропуск {filename}: TranscriberService недоступен.")
            return False
        try:
            text = transcriber.transcribe(audio_path)
            logger.info(f"Транскрибация завершена ({len(text)} символов).")
        except Exception as e:
            logger.error(f"Ошибка транскрибации {filename}: {e}")
            return False

    if not text.strip():
        logger.warning(f"Файл {filename} пустой или речь не обнаружена.")
        return False

    # 2. Классификация
    cls_res = classify_content(filename, text)
    domain = cls_res["domain"]
    title = cls_res["title"]
    sub_type = cls_res["sub_type"]
    category_folder = CATEGORY_FOLDERS.get(domain, "Inbox")
    logger.info(f"Категория: {domain} ({sub_type})")

    # 3. Сохранение Markdown-заметки
    target_dir = os.path.join(OUTPUT_DIR, category_folder)
    os.makedirs(target_dir, exist_ok=True)

    note_path = os.path.join(target_dir, f"{timestamp}_{base_name}.md")
    if not os.path.exists(note_path):
        note_content = (
            f"---\n"
            f'title: "{title}: {date_iso}"\n'
            f"created: {date_iso}\n"
            f"updated: {date_iso}\n"
            f'source: "Voice Note"\n'
            f'original_file: "{filename}"\n'
            f'category: "{domain}"\n'
            f'sub_type: "{sub_type}"\n'
            f"status: inbox\n"
            f"tags:\n"
            f"  - voice-note\n"
            f"  - {domain.lower()}\n"
            f"---\n\n"
            f"# {title} ({timestamp})\n\n"
            f"- **Файл:** `{filename}`\n"
            f"- **Категория:** {domain} ({sub_type})\n\n"
            f"---\n\n"
            f"## Расшифровка\n\n"
            f"{text}\n"
        )
        with open(note_path, "w", encoding="utf-8") as f:
            f.write(note_content)
        logger.info(f"Сохранена заметка: {note_path}")

    # 4. Опциональное копирование исходника в архив по категориям
    if ARCHIVE_AUDIO_DIR:
        archive_target_dir = os.path.join(ARCHIVE_AUDIO_DIR, category_folder)
        os.makedirs(archive_target_dir, exist_ok=True)
        try:
            shutil.copy2(audio_path, os.path.join(archive_target_dir, filename))
        except Exception as e:
            logger.warning(f"Не удалось скопировать файл в архив {archive_target_dir}: {e}")

    # 5. Перемещение обработанного файла в PROCESSED_AUDIO_DIR
    os.makedirs(PROCESSED_AUDIO_DIR, exist_ok=True)
    target_processed_path = os.path.join(PROCESSED_AUDIO_DIR, filename)
    try:
        shutil.move(audio_path, target_processed_path)
        logger.info(f"Исходный файл перемещён в: {target_processed_path}")
    except Exception as e:
        logger.warning(f"Не удалось переместить файл в {PROCESSED_AUDIO_DIR}: {e}")

    return True


def run_sync() -> None:
    """Сканирует INPUT_AUDIO_DIR и последовательно обрабатывает все поддерживаемые медиафайлы."""
    logger.info("=== Запуск локальной обработки голосовых заметок ===")
    input_path = Path(INPUT_AUDIO_DIR)
    if not input_path.exists():
        logger.error(f"Входящая директория не найдена: {input_path.resolve()}")
        logger.info("Создайте папку или укажите корректный INPUT_AUDIO_DIR в .env")
        return

    os.makedirs(PROCESSED_AUDIO_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    media_files = sorted(
        str(p)
        for p in input_path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not media_files:
        logger.info("Новых аудио/видео файлов для обработки не обнаружено.")
        return

    logger.info(f"Найдено файлов для обработки: {len(media_files)}")
    processed_count = 0
    for media_file in media_files:
        try:
            if process_audio_file(media_file):
                processed_count += 1
        except Exception as e:
            logger.error(f"Ошибка при обработке {media_file}: {e}", exc_info=True)

    logger.info(f"=== Готово. Обработано файлов: {processed_count}/{len(media_files)} ===")


if __name__ == "__main__":
    run_sync()
