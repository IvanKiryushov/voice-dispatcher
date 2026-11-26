# Voice Dispatcher

Локальный пайплайн для автоматической расшифровки голосовых заметок и видеозаписей созвонов в Markdown-файлы, плюс опциональный облачный шаблон на `n8n + Gemini Flash`.

## О проекте

Инструмент забирает аудио- и видеофайлы из входной папки (например, синхронизированной с Google Drive), расшифровывает речь, раскладывает готовые `.md`-заметки с YAML frontmatter по категориям (`Meetings`, `Content`, `Work`, `Inbox`) и переносит обработанные исходники в `processed`.

Репозиторий состоит из двух независимых веток:

1. **Локальная ветка (`src/`)** — основная. Python-скрипт на базе `faster-whisper` с фильтром тишины (VAD), разбиением на абзацы по паузам и откатом `cuda → cpu`. Работает оффлайн, бесплатно и без лимитов на длину файла.
2. **Облачная ветка (`n8n/`)** — опциональная. Готовый JSON-шаблон воркфлоу для `n8n` (Google Drive Trigger → Gemini Flash STT + классификация → сохранение Markdown → перенос файла). У автора эта ветка отключена в пользу локальной, но оставлена как рабочий референс.

```text
voice-dispatcher-public/
├── src/
│   ├── transcriber_service.py      # Обёртка faster-whisper (CUDA DLL loader, VAD, абзацы, fallback на CPU)
│   └── voice_sync_distributor.py   # Сканер папки, сортировка по категориям, генерация .md
├── n8n/
│   └── voice_dispatcher_workflow.json  # Обезличенный шаблон воркфлоу для n8n
├── .env.example
├── requirements.txt
└── README.md
```

---

## Исходная конфигурация автора

Пайплайн собирался и обкатывался под конкретное железо и сценарий:

- **ОС и GPU:** Windows 11, видеокарта NVIDIA GTX 1050 Ti (4 GB VRAM, архитектура Pascal). Из-за ограничений архитектуры Pascal и 4 GB видеопамяти используется квантизация `compute_type="int8"` и модель `large-v3-turbo` — это даёт хорошее качество русского языка без вылета по памяти.
- **Цепочка доставки файлов:** диктофон на Android → приложение фоновой автосинхронизации (DriveSync / FolderSync) → папка в Google Drive (подмонтированная локально через Google Drive for Desktop) → запуск `voice_sync_distributor.py`.

---

## Быстрый старт (локальная ветка)

Требуется Python 3.10+.

```bash
# 1. Создать и активировать виртуальное окружение
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Если на Windows есть GPU NVIDIA, но не установлен системный CUDA Toolkit 12:
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12

# 4. Создать .env из шаблона и прописать свои пути
cp .env.example .env

# 5. Запустить обработку
python src/voice_sync_distributor.py
```

При первом запуске `faster-whisper` автоматически скачает веса выбранной модели из Hugging Face в локальный кэш.

### Адаптация под своё железо

Параметры задаются в `.env` через `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE` и `WHISPER_MODEL`:

| Железо | `WHISPER_DEVICE` | `WHISPER_COMPUTE_TYPE` | `WHISPER_MODEL` | Примечания |
| --- | --- | --- | --- | --- |
| NVIDIA GPU, 4–6 GB VRAM (GTX 1050 Ti / 1650 / RTX 3050) | `cuda` | `int8` | `large-v3-turbo` | Оптимальный баланс качества и расхода VRAM |
| NVIDIA GPU, 8+ GB VRAM (RTX 3060 / 4060 и выше) | `cuda` | `float16` | `large-v3-turbo` или `large-v3` | Максимальная точность и скорость |
| ПК без видеокарты / слабый CPU | `cpu` | `int8` | `small` или `medium` | Работает заметно медленнее реального времени (для длинных записей лучше облачный API) |
| macOS Apple Silicon (M1–M4) | — | — | — | См. нюанс ниже |

**Нюанс для macOS (Apple Silicon M1–M4):** движок `CTranslate2`, на котором работает `faster-whisper`, не поддерживает ускорение Apple Metal / MPS и на Mac запускается только на CPU. Для Mac разумнее использовать `mlx-whisper` (CLI/Python на Apple MLX) или готовые GUI-приложения на `whisper.cpp` (Vibe, MacWhisper).

---

## Практические нюансы

### iPhone (iOS)

На iOS нет полноценной фоновой автосинхронизации папки диктофона с Google Drive, как через DriveSync/FolderSync на Android. Кроме того, встроенная транскрибация в Apple Voice Memos не поддерживает русский язык.

Рабочие обходные пути для iPhone:
- После записи нажимать **«Поделиться» → сохранить в папку Google Drive / iCloud Drive**, откуда файл заберёт скрипт.
- Настроить **iOS Shortcut (Команду)**, которая берёт последний файл из «Диктофона» и отправляет его в облачную папку или напрямую в API.
- Отправлять голосовые/аудиофайлы в собственного **Telegram-бота**, который сохраняет их в облако или сразу расшифровывает через API.

### Онлайн-созвоны (видеозаписи)

Скрипт поддерживает форматы `.m4a`, `.mp4`, `.mp3`, `.wav`, `.ogg`, `.mkv`, `.mov`, `.webm`. Библиотека `faster-whisper` (через PyAV/ffmpeg) умеет читать аудиодорожку напрямую из видеофайлов — достаточно положить запись Zoom/Meet/OBS в `INPUT_AUDIO_DIR`.

Если видеофайл очень тяжёлый (несколько гигабайт) и нужно сэкономить место при синхронизации, аудиодорожку можно предварительно выдернуть через `ffmpeg`:

```bash
ffmpeg -i meeting.mp4 -vn -acodec libopus -b:a 32k meeting.ogg
```

### Диаризация спикеров (разделение по ролям)

В текущей версии **диаризации нет** — весь текст выводится сплошным потоком, разбитым на абзацы по паузам. Если нужно разделение на «Спикер 1 / Спикер 2», варианты такие:
- **Локально (open-source):** связка `whisperX` (`Whisper` + `pyannote.audio`) или свежая сверхбыстрая end-to-end модель **[NVIDIA Nemotron Diarization (Sortformer)](https://huggingface.co/blog/nvidia/nemotron-diarization)** (через NVIDIA NeMo / `nemo-speech.cpp` — работает без тяжёлой кластеризации).
- **Облачные API со встроенной диаризацией:** **ElevenLabs Scribe** (одна из самых точных и быстрых моделей с нативной диаризацией и хорошим русским, ~\$0.20–\$0.22/час), **AssemblyAI** (~\$0.15/час) и **Deepgram Nova-3** (~\$0.26/час).

### Категоризация заметок

Функция `classify_content()` в [src/voice_sync_distributor.py](src/voice_sync_distributor.py) содержит базовый пример сортировки по ключевым словам (`MEETINGS`, `CONTENT`, `WORK`, `INBOX`). Перепишите списки слов под свои домены или полностью отключите сортировку через `ENABLE_CLASSIFICATION=false` в `.env` (тогда все файлы пойдут в `Inbox/`).

---

## Облачная ветка (`n8n/`)

Файл [n8n/voice_dispatcher_workflow.json](n8n/voice_dispatcher_workflow.json) — обезличенный шаблон для self-hosted или cloud `n8n`.

Поток работы:
1. `Google Drive Trigger (Audio)` отслеживает появление новых файлов во входной папке.
2. `Download Audio Binary` скачивает аудио.
3. `Gemini Flash (STT + Classifier)` отправляет аудио в Gemini API и получает JSON с транскриптом и категорией (с fallback на `gemini-2.5-flash-lite` при ошибке и 5 ретраями).
4. `Parse Gemini JSON` нормализует ответ и формирует timestamp.
5. `Save Raw Note to Inbox` сохраняет сырую заметку, а `Switch Domain Router` создаёт отформатированную заметку по категории.
6. `Move Audio to Processed Folder` переносит исходник в папку обработанных файлов.

**Настройка после импорта в n8n (`Workflows → Import from File`):**
- Подключите свои Google Drive OAuth2 credentials во всех узлах Google Drive (`YOUR_GOOGLE_DRIVE_CREDENTIAL_ID`).
- Замените ID папок Google Drive: `YOUR_INPUT_AUDIO_FOLDER_ID`, `YOUR_OUTPUT_TRANSCRIPTS_FOLDER_ID`, `YOUR_PROCESSED_AUDIO_FOLDER_ID`.
- Укажите свой ключ Gemini API вместо `YOUR_GEMINI_API_KEY` (в узлах `Gemini Flash` и `Gemini Fallback`).
- Активируйте воркфлоу.

---

## Альтернативы на рынке (если не хочется возиться с кодом)

Если задача — просто быстро расшифровывать встречи и голосовые без настройки Python и окружений:

### Локальные GUI-приложения (бесплатно / оффлайн)
- **[Vibe](https://github.com/thewh1teagle/vibe)** (MIT, Windows / macOS / Linux) — минималистичный десктопный клиент на `whisper.cpp`, умеет перетаскивать аудио/видео и поддерживает экспорт.
- **[MacWhisper](https://goodsnooze.gumroad.com/l/macwhisper)** (macOS) — быстрая локальная расшифровка на Apple Silicon с удобным интерфейсом.
- **[Meetily](https://github.com/Zackriya-Solutions/meeting-minutes)** (MIT) — открытый инструмент для локальной записи и расшифровки онлайн-созвонов без подключения ботов в комнату.

### Дешёвые облачные API (слабое железо или быстрый сценарий с iPhone)
- **Groq API (`whisper-large-v3-turbo`)** — порядка \$0.04 за час аудио, экстремально высокая скорость ответа.
- **Google Gemini 2.5 Flash API** — принимает до ~9.5 часов аудио в одном запросе; за один проход можно получить и транскрипт, и выжимку тезисов / задач / контент-плана.
- **ElevenLabs Scribe** — ~\$0.20–\$0.22/час, очень быстрая транскрибация с качественной встроенной диаризацией и поддержкой русского языка.
- **AssemblyAI** — ~\$0.15/час (есть \$50 стартового кредита), качественная диаризация спикеров из коробки.
- **Deepgram Nova-3** — ~\$0.26/час (есть \$200 стартового кредита), высокая точность и потоковый режим.

### Аппаратное решение для оффлайн-встреч с iPhone
- **Plaud Note / Plaud NotePin** — компактные диктофоны с хорошим качеством русского языка и диаризацией спикеров. Минусы: стоимость самого устройства и привязка к проприетарному облаку с подпиской.
