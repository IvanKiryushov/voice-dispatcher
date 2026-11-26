"""
Сервис локальной транскрибации аудио и видео на базе faster-whisper.
Изолирует инициализацию модели, подключение CUDA DLL на Windows
и автоматический откат на CPU при недоступности GPU.
"""

import logging
import os
import site
import sys
from typing import Optional

from faster_whisper import WhisperModel

logger = logging.getLogger("TranscriberService")

# Базовый промпт для русского языка: задаёт грамотную пунктуацию без лишних повторов
DEFAULT_RU_PROMPT = (
    "Это расшифровка речи на русском языке. "
    "Используй правильную пунктуацию: точки, запятые, знаки вопроса, тире и кавычки. "
    "Пиши грамотно, без лишних повторов."
)


class TranscriberService:
    """
    Обёртка над faster-whisper с ленивой загрузкой модели,
    фильтром тишины (VAD) и разбиением текста на абзацы по паузам.
    """

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "int8",
        language: Optional[str] = "ru",
        initial_prompt: Optional[str] = None,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        # Пустая строка или "auto" переводятся в None для автоопределения языка в faster-whisper
        self.language = None if not language or language.lower() == "auto" else language
        if initial_prompt is not None:
            self.initial_prompt = initial_prompt if initial_prompt.strip() else None
        else:
            self.initial_prompt = DEFAULT_RU_PROMPT if self.language == "ru" else None
        self.model = None

    def _setup_cuda_dlls(self) -> None:
        """
        Регистрирует директории с CUDA DLL из pip-пакетов (nvidia-cublas-cu12, nvidia-cudnn-cu12 и др.)
        в текущем виртуальном окружении Python на Windows.
        """
        if sys.platform != "win32":
            return

        candidate_site_packages = [
            os.path.join(sys.prefix, "Lib", "site-packages"),
        ]
        try:
            for sp in site.getsitepackages():
                if sp not in candidate_site_packages:
                    candidate_site_packages.append(sp)
        except Exception:
            pass

        nvidia_subdirs = [
            ("nvidia", "cublas", "bin"),
            ("nvidia", "cudnn", "bin"),
            ("nvidia", "cuda_nvrtc", "bin"),
        ]

        added_dirs = set()
        for sp in candidate_site_packages:
            for sub in nvidia_subdirs:
                dll_dir = os.path.join(sp, *sub)
                if dll_dir not in added_dirs and os.path.exists(dll_dir):
                    try:
                        # Добавляем путь в поиск DLL для текущего процесса (Python 3.8+)
                        os.add_dll_directory(dll_dir)
                        os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
                        added_dirs.add(dll_dir)
                        logger.info(f"Подключена директория CUDA DLL: {dll_dir}")
                    except Exception as e:
                        logger.warning(f"Не удалось подключить {dll_dir}: {e}")

    def load_model(self) -> None:
        """
        Лениво загружает модель Whisper.
        При сбое инициализации на GPU автоматически откатывается на CPU (int8).
        """
        if self.model is not None:
            return

        if self.device == "cuda":
            self._setup_cuda_dlls()

        logger.info(
            f"Загрузка модели '{self.model_size}' (device={self.device}, compute_type={self.compute_type})..."
        )
        try:
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info(f"Модель успешно загружена на {self.device} ({self.compute_type}).")
        except Exception as e:
            if self.device == "cpu":
                logger.error(f"Не удалось загрузить модель на CPU: {e}", exc_info=True)
                raise

            logger.error(f"Сбой инициализации на {self.device} ({e}). Переход на CPU (int8)...")
            try:
                self.model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                )
                logger.info("Модель загружена на CPU (int8). Обработка займёт больше времени.")
            except Exception as e_cpu:
                logger.error(f"Критическая ошибка загрузки модели на CPU: {e_cpu}", exc_info=True)
                raise

    def transcribe(self, audio_data, stream: bool = False):
        """
        Выполняет транскрибацию аудио/видео файла (путь str/Path) или numpy-массива.
        При stream=True возвращает генератор сегментов, иначе — текст, разбитый на абзацы.
        """
        if self.model is None:
            self.load_model()

        if audio_data is None or (hasattr(audio_data, "__len__") and len(audio_data) == 0):
            return [] if stream else ""

        try:
            beam_size = 2 if stream else 5

            segments, _info = self.model.transcribe(
                audio_data,
                beam_size=beam_size,
                language=self.language,
                initial_prompt=self.initial_prompt,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            if stream:
                return segments

            # Разбиваем расшифровку на абзацы по длительности пауз между сегментами
            paragraphs = []
            current_paragraph = []
            last_end = None

            for segment in segments:
                text = segment.text.strip()
                if not text:
                    continue

                if last_end is not None and current_paragraph:
                    pause = segment.start - last_end
                    last_text = current_paragraph[-1]
                    ends_with_punc = bool(last_text) and last_text[-1] in ".!?"

                    # Новый абзац при паузе > 2.0 с или при паузе > 1.2 с после завершённого предложения
                    if pause > 2.0 or (ends_with_punc and pause > 1.2):
                        paragraphs.append(" ".join(current_paragraph))
                        current_paragraph = [text]
                    else:
                        current_paragraph.append(text)
                else:
                    current_paragraph.append(text)

                last_end = segment.end

            if current_paragraph:
                paragraphs.append(" ".join(current_paragraph))

            return "\n\n".join(paragraphs)
        except Exception as e:
            logger.error(f"Ошибка транскрибации: {e}", exc_info=True)
            return [] if stream else ""
