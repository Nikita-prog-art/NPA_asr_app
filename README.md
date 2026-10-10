# NPA ASR — GigaAM-v3 e2e_rnnt на ggml

Кроссплатформенное (Windows + Linux) приложение для распознавания русской речи
моделью [`ai-sage/GigaAM-v3`](https://huggingface.co/ai-sage/GigaAM-v3)
(revision `e2e_rnnt`) на экосистеме **ggml** (llama.cpp / ggml / whisper.cpp):
свой Conformer-энкодер и RNN-T-декодер на графах ggml, без Python и без
libtorch в рантайме.

* **Push-to-talk**: зажали сочетание клавиш — пошла запись и **потоковая**
  транскрибация (текст появляется уже через ~0.7–1 с, а не после отпускания).
* **Настройка числа потоков**, тёмная/светлая темы.
* **Скачивание модели** нужного уровня квантизации прямо из приложения (или
  указание пути к своему `.gguf`).
* Финальный текст **вставляется в активное окно** и дублируется в буфер обмена.
* Таргет под архитектуру процессора (например `alderlake`: AVX2 + AVX-VNNI) с
  рантайм-фолбеком на `haswell`/`sandybridge`/`sse42`/`x64` — через
  `GGML_CPU_ALL_VARIANTS` + динамические бэкенды ggml.
* Сборка с Intel oneAPI (`~/dists/intel/oneapi`,
  `C:\Program Files (x86)\Intel\oneAPI`): компилятор `icx/icpx` и oneMKL DFTI для
  FFT мел-спектрограммы (с автоматическим фолбеком на собственный FFT).

---

## 1. Что уже проверено

Численная эквивалентность эталонной PyTorch-реализации (тот самый
`modeling_gigaam.py` с Hugging Face) проверена по шагам:

| Проверка | Результат |
|---|---|
| log-mel фронтенд против `torchaudio` | `max_abs = 4.1e-05` |
| энкодер целиком (`enc`, 16 слоёв) | `max_abs = 1.2e-03` при диапазоне ±2.4 |
| промежуточные слои 0…15 | совпадают (ошибка ≤ 1.5e-02 на значениях до ±200) |
| предиктор (embedding + LSTM) | `max_abs = 6e-08` |
| joint-сеть (сырые логиты) | `max_abs = 1.5e-02` |
| **итоговые токены** (жадное декодирование) | **те же**, что у PyTorch |
| текст на 3 тестовых файлах | совпадает с эталоном, включая пунктуацию и нормализацию чисел |

Примеры (TTS-файлы 16 кГц, синтез русским голосом Windows):

```
«Привет, это тест распознавания речи.»
  → Привет. Это тест распознавания речи.

«Сегодня хорошая погода, давайте проверим, как работает потоковое распознавание речи на русском языке.»
  → Сегодня хорошая погода. Давайте проверим, как работает потоковое распознавание речи на русском языке.

«Запиши встречу на пятнадцатое марта в четырнадцать тридцать.»
  → Запиши встречу на 15 марта в 14:30.
```

Скорость на i9-13900HK (8 потоков, `f16`): **~10–12× реального времени**
(≈ 380–780 мс на 4–9 с аудио), модель 222.5M параметров, 445 МБ в `f16`.

---

## 2. Как это устроено

```
микрофон ──► кольцевой буфер ──► log-mel (320-точечный FFT: oneMKL DFTI или свой Bluestein)
                                      │
                                      ▼
                     Conformer-энкодер на ggml (16 слоёв)
                     subsampling conv1d ×2, rotary self-attention,
                     conv-модуль (depthwise + LayerNorm), FFN ×2
                                      │
                                      ▼
                     RNN-T: предиктор (embedding + LSTM) + joint
                     покадровое жадное декодирование (max 10 символов/кадр)
                                      │
                                      ▼
                     SentencePiece (словарь зашит в GGUF) ──► текст
```

**Про потоковость.** RNN-T-декодер потоковый по своей природе (он покадровый),
но Conformer-энкодер этой модели обучен с полным (не причинным) вниманием,
поэтому «настоящего» причинного стриминга у модели нет. Приложение использует
скользящее окно с локальным согласованием (LocalAgreement-2, как в
whisper-stream):

* пока запись идёт, каждые `step_sec` (по умолчанию 1 с) окно последних
  `max_window_sec` (по умолчанию 20 с) перекодируется целиком — то есть
  внимание остаётся ровно таким, как при обучении, качество не теряется;
* стабильный префикс токенов (совпавший с предыдущим окном) «коммитится» и
  показывается как готовый текст, хвост — как предварительный;
* по отпусканию клавиши делается финальный проход и текст точный.

Короткие реплики (до 20 с — обычный push-to-talk) всегда кодируются целиком,
так что финальный результат совпадает с офлайн-распознаванием.

---

## 3. Сборка

### Linux

```bash
scripts/build_linux.sh              # gcc по умолчанию; ASR_COMPILER=icx — с icx/icpx
ASR_MKL_FFT=ON scripts/build_linux.sh  # + oneMKL DFTI для мел-FFT (см. ниже)
ASR_NATIVE=ON scripts/build_linux.sh   # один бинарник под этот CPU (быстрее, без фолбека)
ASR_JOBS=6 scripts/build_linux.sh      # ограничить параллелизм (см. про память ниже)
```

oneAPI подхватывается (`source ~/dists/intel/oneapi/setvars.sh --force`)
**только** если запрошен `ASR_COMPILER=icx` или `ASR_MKL_FFT=ON`;
по умолчанию сборка от oneAPI не зависит. Результат: `build/bin/npa_asr`.

Зависимости: `cmake`, `ninja`, `libx11-dev`, `libxtst-dev`,
`libgl1-mesa-dev`, `libcurl4-openssl-dev` (последние два — для GUI и
загрузчика; ggml/miniaudio/imgui/glfw скачиваются CMake-ом автоматически,
`-DASR_FETCH_DEPS=OFF` — если хотите положить их руками).

### Windows

```cmd
scripts\build_windows.cmd            :: MSVC + oneAPI (oneMKL, Intel OpenMP)
set ASR_COMPILER=icx && scripts\build_windows.cmd
set ASR_MKL_FFT=ON && scripts\build_windows.cmd
set ASR_NATIVE=ON && scripts\build_windows.cmd
```

Скрипт находит Visual Studio через `vswhere`, подхватывает
`C:\Program Files (x86)\Intel\oneAPI\setvars.bat` (только если запрошен icx или
oneMKL) и собирает `build-windows\bin\npa_asr.exe`. Рядом с exe должны лежать
`ggml-cpu-*.dll` — именно они дают рантайм-выбор варианта CPU; GLFW
линкуется статически, так что больше никаких DLL не требуется. Готовые сборки
(Windows/Linux) собираются в CI: `.github/workflows/build.yml`.

Если разработка идёт в WSL (как у автора), удобнее собрать Windows-версию
оттуда — MSVC не умеет работать по UNC-пути, поэтому скрипт упаковывает исходники
в один tar, распаковывает на хост и запускает обычную сборку:

```bash
scripts/build_windows_from_wsl.sh          # WIN_BUILD_DIR=C:\dists\builds\npa_asr_build
```

### oneAPI: что он реально даёт

Измерено на i9-13900HK (8 потоков, `asr_test bench`, файл 8.98 с; gcc-бинарь
запускался без `LD_LIBRARY_PATH` oneAPI, icx — со своими runtime-библиотеками):

| Сборка | мел-фронтенд | энкодер (10 итераций × 3 прогона) | всё распознавание |
|---|---|---|---|
| gcc 14 + oneMKL DFTI | 33.8 мс (3.8 %) | 857 мс (856/856/857) | 896 мс (10.0× RT) |
| gcc 14 + свой FFT (Bluestein) | 35.4 мс (3.8 %) | 857 мс | 923 мс (9.7× RT) |
| icx/icpx 2025.3 + Intel OpenMP | 28–31 мс | 876 мс (840–910) | 1015 мс (8.8× RT) |

Выводы:

* **oneMKL ускоряет только FFT, а FFT — это ~4 % времени**, то есть итоговый
  выигрыш ≈ 0.2 % (разница в энкодере — шум). При этом линковка oneMKL тянет
  зависимость от окружения oneAPI в рантайме: на Windows exe без `setvars.bat`
  падал с «не обнаружена mkl_rt.2.dll», на Linux в процесс отображается ~100 МБ
  библиотек MKL. Поэтому `ASR_MKL_FFT=OFF` по умолчанию: свой FFT (Bluestein
  поверх radix-2) даёт ту же точность (проверено: `max_abs = 4.1e-05` против
  `torchaudio`) и не тянет зависимостей.
* **icx/icpx + Intel OpenMP не быстрее gcc 14** (скорее на ~2 % медленнее и
  заметно менее стабилен по времени). Это ожидаемо: ядра ggml написаны
  интринсиками вручную, компилятору почти нечего улучшать. Поэтому по умолчанию
  собираем gcc/MSVC, а `ASR_COMPILER=icx` остаётся опцией.
* `--fft internal|mkl|auto` переключает FFT-бэкенд в рантайме,
  `npa_asr --info` показывает текущий и список доступных вариантов CPU.

Реальные рычаги производительности — число потоков (`--threads`, в GUI ползунок)
и `ASR_NATIVE=ON` (сборка под конкретный CPU без динамических бэкендов, но и без
фолбека).

### Про память при сборке

`GGML_CPU_ALL_VARIANTS` собирает `ggml-cpu` по разу на каждый вариант ISA, и
большие TU (`ggml-cpu/ops.cpp` с AVX512) требуют ~2 ГБ ОЗУ каждый. Поэтому
скрипты считают `-j` как `min(ядра, ОЗУ_ГБ/2)`; переопределяется через
`ASR_JOBS`. Без этого `-j20` на 16-гигабайтной WSL легко съедает всю память и
вешает дистрибутив (проверено на себе). Если памяти всё равно мало, добавьте в
`%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=24GB
swap=8GB
```

---

## 4. Модель

Три способа получить веса:

1. **Скачать из приложения** (вкладка «Модели» → «Каталог моделей»): приложение
   читает `models/manifest.json` из репозитория/релиза, показывает доступные
   квантизации и качает выбранную с проверкой SHA-256.
   Чтобы файлы появились, один раз запустите workflow
   `.github/workflows/convert-model.yml` (Actions → Run workflow): он скачает
   `pytorch_model.bin` (449 МБ) с Hugging Face, сконвертирует в GGUF
   (`f16`, `q8_0`, `q5_0`, `q4_0`), опубликует их как release assets и обновит
   `models/manifest.json` в репозитории.
2. **Сконвертировать самому** (Python, ~1 минута):
   ```bash
   python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
   python -m pip install gguf numpy sentencepiece
   # config.json, tokenizer.model, pytorch_model.bin из revision e2e_rnnt
   python tools/convert_gigaam_to_gguf.py \
       --model-dir models/hf/gigaam-v3-e2e_rnnt \
       --out models/gguf/gigaam-v3-e2e_rnnt-f16.gguf --quant f16
   ```
3. **Указать путь** к любому готовому `.gguf` (вкладка «Модели» → поле пути или
   перетаскивание файла в окно).

Уровни квантизации: `f16` — точная копия оригинальных весов (энкодер в
чекпоинте и так fp16), `q8_0`/`q5_0`/`q4_0` — меньше размер, чуть быстрее;
декодер, joint-сеть и мел-фильтрбанк всегда остаются fp32.

---

## 5. Использование

### GUI

```bash
./build/bin/npa_asr            # или npa_asr.exe
```

Зажмите **Ctrl+Alt+Space** (настраивается, в том числе «записать сочетание» в
настройках) или кнопку «Удерживайте для записи» мышью. Текст появляется на
экране по мере распознавания, по отпусканию вставляется в активное окно и
кладётся в буфер обмена (режим вывода настраивается: в окно + буфер / только
буфер / только в окне).

### CLI

```bash
npa_asr -m model.gguf -i audio.wav          # офлайн-распознавание файла
npa_asr -m model.gguf -i audio.wav --stream # эмуляция потока кусками по 100 мс
npa_asr -m model.gguf --mic                 # push-to-talk без GUI
npa_asr --list-devices                      # устройства захвата
npa_asr --list-models                       # локальные .gguf
npa_asr --download --quant q8_0             # скачать модель из релиза
npa_asr --info                              # какой ggml-бэкенд/CPU выбран
```

### Горячие клавиши на Linux

* **X11 (в т.ч. WSLg)**: по умолчанию используется XRecord — сочетание
  *наблюдается*, но не перехватывается у других приложений. Если libXtst
  недоступна, приложение падает на `XGrabKey` (тогда сочетание забирается себе).
* **Wayland**: глобальных хоткеев без порталов нет, поэтому используется
  чтение `/dev/input/event*` (evdev). Нужно добавить пользователя в группу
  `input`: `sudo usermod -aG input $USER` и перелогиниться.
* Вставка текста на Linux делается через XTest (libXtst) в активное окно; если
  библиотеки нет — текст остаётся в буфере обмена, вставляется вручную.

---

## 6. Структура репозитория

```
CMakeLists.txt              сборка (опции ASR_GUI/ASR_NATIVE/ASR_MKL_FFT/ASR_FETCH_DEPS)
cmake/oneapi.cmake          поиск oneAPI/oneMKL, выбор FFT
cmake/deps.cmake            пины и загрузка ggml/imgui/glfw/miniaudio/json
src/core/                   мел-фронтенд, FFT, энкодер, RNN-T, токенизатор, распознаватель
src/audio/                  захват с микрофона (miniaudio) и чтение файлов
src/platform/               глобальный хоткей, буфер обмена, вставка текста (Win32 / X11 / evdev)
src/net/                    загрузка моделей (WinHTTP / libcurl), SHA-256, манифест
src/app/                    настройки, менеджер моделей, контроллер, GUI, CLI
tools/                      конвертер в GGUF, генераторы эталонов, тесты
scripts/                    сборка под Linux/Windows
.github/workflows/          сборка приложения и конвертация модели в GGUF
```

Тесты и эталоны (нужен Python-venv с torch). `ref_gigaam.py` прогоняет
официальный `modeling_gigaam.py` с Hugging Face на любом 16-кГц wav и
выгружает эталонные тензоры, `asr_test` сверяет с ними C++/ggml-реализацию:

```bash
python -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install numpy sentencepiece omegaconf hydra-core transformers torchaudio
.venv/bin/python tools/ref_gigaam.py my_audio.wav --dump goldens.npz --dump-layers
.venv/bin/python tools/dump_goldens.py goldens.npz goldens.bin
./build/bin/asr_test mel models/gguf/gigaam-v3-e2e_rnnt-f16.gguf goldens.bin   # мел против torchaudio
./build/bin/asr_test enc models/gguf/gigaam-v3-e2e_rnnt-f16.gguf goldens.bin   # энкодер по слоям
./build/bin/asr_test run models/gguf/gigaam-v3-e2e_rnnt-f16.gguf goldens.bin   # токены + текст, офлайн и стрим
./build/bin/asr_test bench models/gguf/gigaam-v3-e2e_rnnt-f16.gguf my_audio.wav 8 5
```

Эталонные wav для проверок в репозитории не лежат: те, что использовались при
разработке, были синтезированы русским голосом Windows
(`System.Speech.Synthesis`, `Microsoft Irina Desktop`) — скрипт есть в
истории разработки, но подойдёт любой 16-кГц моно wav с известным текстом.

---

## 7. Лицензии и благодарности

* модель — [ai-sage/GigaAM-v3](https://huggingface.co/ai-sage/GigaAM-v3), MIT;
* [ggml](https://github.com/ggml-org/ggml) (MIT) — ядро инференса;
* [miniaudio](https://github.com/mackron/miniaudio) (public domain / MIT-0) — звук;
* [Dear ImGui](https://github.com/ocornut/imgui) (MIT) + [GLFW](https://www.glfw.org/) (zlib) — интерфейс;
* [nlohmann/json](https://github.com/nlohmann/json) (MIT) — настройки и манифест;
* [oneMKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) — FFT (опционально).
