# Video Orchestrator — Project Specification

> Локален агентен оркестратор за AI видео продукция.
> Дирижира Gemma 4 (LLM + vision) и ComfyUI (Klein/Wan/ReActor) в самооценяващ pipeline.
> Хардуер: RTX 5070 Ti 16GB VRAM, 64GB RAM, Xubuntu.

---

## 1. Какво прави системата

Превръща текстова идея за история в серия консистентни видео клипове, автономно:

```
Идея → storyboard → [per scene: keyframe промпт → изображение → ОЦЕНКА →
        анимационен промпт → видео → ОЦЕНКА → face swap → ОЦЕНКА] → клипове за монтаж
```

Ключовото: след всяка генерация Gemma (vision) **оценява** резултата спрямо задачата
и решава accept / retry. При retry — коригира промпта и пробва пак (макс N опита).

---

## 2. Компоненти (endpoint-и и команди)

### Gemma 4 (LLM дирижор + vision оценител)
- Модел: `~/models/gemma4/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` (Q4; Q6 не влиза в VRAM)
- Vision projector: `~/models/gemma4/mmproj-BF16.gguf` (ЗАДЪЛЖИТЕЛЕН за vision)
- Архитектура: MoE, 26B общо / ~4B активни. Multimodal (text + image; БЕЗ audio на 26B).
- API: OpenAI-съвместим, `http://localhost:8080/v1/chat/completions`
- Старт команда:
  ```bash
  ./build/bin/llama-server \
    -m ~/models/gemma4/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \
    --mmproj ~/models/gemma4/mmproj-BF16.gguf \
    --n-gpu-layers 24 --ctx-size 32768 --no-kv-offload \
    --host 0.0.0.0 --port 8080
  ```
- Vision вход: image като base64 в `image_url` content блок (data:image/...;base64,...)
- ПОТВЪРДЕНО: дава детайлна, точна оценка (действие, идентичност, стил, емоция, артефакти).

### ComfyUI (генерация)
- API: `http://localhost:8188`
- Пускане: `POST /prompt` с workflow JSON (API формат)
- Резултат: poll `GET /history/{prompt_id}` докато готово, после изходните файлове
- Порт за реалното изпълнение; workflow JSON-ите са в `workflows/`
- Output: `~/comfy/ComfyUI/output/` (изображения и video/ подпапка)
- Input: `~/comfy/ComfyUI/input/` (тук отиват изображения за LoadImage node-ове)

---

## 3. Workflow-и (в workflows/, API формат) + кои параметри се заменят

Всеки workflow е "шаблон". Оркестраторът зарежда JSON-а, заменя само изброените полета, праща.

### image_flux2_klein_text_to_image.json — генерира keyframe от текст
- Модел: flux-2-klein-4b-fp8 + qwen_3_4b (type flux2) + flux2-vae
- ПОДМЕНИ:
  - node `76` .inputs.value → **позитивен промпт** (PrimitiveStringMultiline)
  - node `75:73` .inputs.noise_seed → **seed** (0 = random)
  - node `75:68` / `75:69` .inputs.value → **width / height**
  - node `75:62` .inputs.steps → steps (base=20+, distilled=4)
  - node `75:63` .inputs.cfg → cfg (base=5, distilled=1)
  - node `75:70` .inputs.unet_name → base vs distilled модел
- ЗАБЕЛЕЖКА: за маслена естетика → BASE модел (flux-2-klein-base-4b-fp8), 24 steps, CFG 5.

### klein_identity_edit.json — keyframe СЪС запазено лице (2 reference)
- Като горния + 2 LoadImage reference (Joro.jpg, Joro _prezarejda.png) — в ComfyUI input/
- ВНИМАНИЕ: reference-ите ЗАКЛЮЧВАТ композицията. Използвай само когато искаш
  СЪЩАТА композиция с леки промени. За НОВИ сцени → text_to_image, не това.
- ПОДМЕНИ:
  - node `4` / `5` .inputs.image → reference изображения
  - node `10` .inputs.text → промпт (с identity-lock фрази)
  - node `18` .inputs.noise_seed → seed
  - node `17` / `19` → width/height (1024x576 за видео-готово)

### wan22_i2v_gguf.json — анимира keyframe → видео (dual-model каскада)
- HighNoise + LowNoise GGUF + umt5 encoder + wan_2.1_vae
- ПОДМЕНИ:
  - node `5` .inputs.image → **start_image** (keyframe файл в input/)
  - node `6` .inputs.text → **анимационен позитивен промпт** (описва ДВИЖЕНИЕ)
  - node `7` .inputs.text → негативен промпт
  - node `8` .inputs.width/height/length → 832x480, length=49 (4n+1!)
  - node `9` / `10` .inputs.noise_seed → seed (двата sampler-а)
  - node `12` .inputs.fps → 16
- ДВА sampler-а: node 9 (HighNoise, steps 0-10) → node 10 (LowNoise, 10-end). Каскада.
- Изход: mp4/h264 в output/video/

### ReActor_Face_Swap.json — налага консистентно лице
- ПОДМЕНИ:
  - node `1` .inputs.image → **input_image** (сцената/кадърът за swap)
  - node `2` .inputs.image → **source_image** (master лицето)
  - node `3` .inputs.face_restore_model → GFPGANv1.4.pth или codeformer.pth
- swap_model: inswapper_128.onnx (в models/insightface/)
- ЗА ВИДЕО: ReActor може да swap-ва и видео кадри (консистентност след Wan дрейф).

### image_flux2_klein_9b_kv_image_edit.json — (резерв, по-сложен edit вариант, FluxKVCache)
- Не за първа фаза. Налично за по-късно ако трябва KV-cache edit.

---

## 4. КРИТИЧНО: VRAM стратегия

16GB е тясно. Gemma Q4 (~частичен offload) + ComfyUI модели НЕ се събират едновременно.
Klein ~8GB, Wan dual 22GB (с offload към RAM), Gemma Q4 ~частично GPU.

**РЕШЕНИЕ (за финализиране от Георги — виж раздел 7):**
Опция A: суичване (разтоварвай един, зареждай друг всеки оценъчен цикъл)
Опция B: Gemma постоянно на CPU (бавна, но нула конфликт; ComfyUI владее GPU)

→ Препоръка: тествай дали Gemma на CPU е поносимо бавна. Ако да — Опция B, по-проста.
→ ComfyUI има model unload API: `POST /free` (изпразва VRAM между workflow-и).

---

## 5. Фазиране на pipeline-а (за минимум суичване)

ПО ФАЗА, не по сцена:
1. Gemma фаза: генерира storyboard + ВСИЧКИ промпти (keyframe + animation за всички сцени) → JSON
2. Klein фаза: генерира ВСИЧКИ keyframes (Klein зареден веднъж)
3. Оценка фаза: Gemma оценява всички keyframes; маркира retry
4. (retry keyframes ако трябва)
5. Wan фаза: анимира всички одобрени keyframes
6. ReActor фаза: swap на всички клипове
7. Финална оценка фаза: Gemma оценява клиповете

Така зареждаш всеки модел минимален брой пъти.

> **Изключение — последователна свързаност (continuity).** При `run-sequence` сцена, маркирана
> `continuous: true`, тръгва от ПОСЛЕДНИЯ кадър на предходния клип. Това е по същество
> последователно (сцена N зависи от видеото на N-1), затова Wan фазата НЕ може да се батчва както
> горе — Klein се зарежда веднъж за cut-сцените, после Wan+ReActor върви сцена по сцена. Виж
> `docs/sequential-continuity-plan.md` и DESIGN.md §13.

---

## 6. Критерии за оценка (какво Gemma проверява)

Gemma връща СТРУКТУРИРАН JSON (не проза). Промпт я кара да оцени:
```json
{
  "action_correct": true,      // прави ли описаното действие (меси, презарежда...)
  "identity_match": true,      // правилният герой ли е (за swap-нати кадри)
  "style_match": true,         // маслена живопис, видими мазки (ако е арт стил)
  "artifacts": false,          // деформации, морфинг, extra пръсти
  "composition_ok": true,      // кадриране, баланс
  "score": 8,                  // 0-10
  "verdict": "accept",         // "accept" | "retry"
  "fix_suggestion": "..."      // ако retry: какво да се промени в промпта
}
```

ОБЕКТИВНИ проверки (action/identity/artifacts) → надеждни, автоматизирай.
СУБЕКТИВНА естетика (mood/качество) → по-слаба; маркирай за човешки преглед, не блокирай.

---

## 7. РЕШЕНИЯ (финализирани — виж DESIGN.md)

- [x] VRAM: **Опция B (Gemma на CPU)** — ComfyUI владее GPU, нула суичване.
- [x] Фазиране: **по фаза** (раздел 5).
- [x] Автономия: **полуавтономен** — човешки gate след storyboard и след оценката на keyframes (преди скъпата Wan фаза).
- [x] Език/оркестрация: **Python, детерминистично ядро; Gemma като извикван инструмент; без agent framework.**
- [x] Тригер: **YAML job файл + CLI** (`python -m vorch run job.yaml`). Активите са динамични, подават се през job spec-а.
- [x] Отхвърлени: Hermes (text-only, без vision), OpenClaw като executor (LLM planner — крехко, VRAM цена; евентуално само външен trigger слой по-късно).
- [x] Structured output: llama.cpp grammar / `response_format` json_schema ПРИ ИЗТОЧНИКА + **Instructor/Pydantic** валидация отвън (grammar може да "fail open" — issue #19051). Tool calling иска `--jinja`.
- [x] Backend: **FastAPI + SSE** (live progress) + **asyncio** опашка (без Celery/Redis — over-engineering за single-user). State: първоначално SQLite → **2026-09: PostgreSQL 16** (локален, peer auth, JSONB, SQL миграции в `vorch/migrations/`; `vorch db init`).
- [x] Frontend: **NiceGUI** (върху FastAPI → един процес backend+frontend; io_bound/cpu_bound; HITL gates). Gradio е по-бърз за медийно UI, но по-слаб за устойчива опашка — отхвърлен за executor UI.
- [x] Цялата валидирана архитектура: виж DESIGN.md раздели 11-12 (deep-research, 25 източника).

---

## 8. Активи в проекта

```
video-orchestrator/
├── NOTES.md                  # този файл
├── SKILL.md (+ skills/)      # инструкции за повтарящи се операции
├── workflows/                # 5-те API JSON шаблона
├── assets/
│   └── jena-cropped.jpeg     # master лице за консистентност
└── reference/
    ├── good/                 # примери "така трябва"
    └── bad/                  # примери "така НЕ трябва"
```

---

## 9. Фазиране на РАЗРАБОТКАТА (не презареждай първата версия)

Минимален проходящ скелет ПЪРВО:
1. ComfyUI API клиент (пусни 1 workflow програмно, вземи резултата) ← фундамент
2. Gemma клиент (текст + vision заявка)
3. Прост pipeline за ЕДНА сцена (без оценка): промпт → Klein → Wan → ReActor
4. → ПОТВЪРДИ че веригата работи end-to-end
Чак после:
5. Оценител (vision feedback loop + retry)
6. VRAM мениджър (suичване/unload)
7. State/logване + цяла история (много сцени)

Проходящ скелет рано > перфектна архитектура която не тръгва.
