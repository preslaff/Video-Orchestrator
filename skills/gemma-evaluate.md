# Gemma Evaluate — vision оценка на генериран резултат

Как оркестраторът праща изображение/видео кадри на Gemma 4 и получава structured JSON verdict.

## Endpoint

OpenAI-съвместим: `POST http://localhost:8080/v1/chat/completions`
Gemma ТРЯБВА да е стартирана с `--mmproj` (иначе сляпа). Виж NOTES.md раздел 2.

## Оценка на ИЗОБРАЖЕНИЕ (keyframe)

```python
import base64, json, requests

GEMMA = "http://localhost:8080/v1/chat/completions"

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def evaluate_image(image_path, task_description):
    b64 = encode_image(image_path)
    system = (
        "Ти си строг арт директор. Оцени изображението спрямо целта. "
        "Върни САМО валиден JSON, без друг текст, без markdown."
    )
    user_text = f"""Цел на кадъра: "{task_description}"

Оцени и върни JSON точно с тези полета:
{{
  "action_correct": bool,    // прави ли описаното действие
  "identity_match": bool,    // правилният герой ли е (ако приложимо, иначе true)
  "style_match": bool,       // съответства ли стилът (маслена живопис и т.н.)
  "artifacts": bool,         // ИМА ли деформации/морфинг/extra пръсти
  "composition_ok": bool,    // добро ли е кадрирането
  "score": int,              // 0-10
  "verdict": "accept" | "retry",
  "fix_suggestion": "ако retry — конкретно какво да се смени в промпта, иначе празно"
}}"""

    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": user_text}
            ]}
        ],
        "max_tokens": 400,
        "temperature": 0.2   # ниска за консистентна оценка
    }
    r = requests.post(GEMMA, json=payload, timeout=300)
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"]
    return parse_json_lenient(raw)

def parse_json_lenient(raw):
    # Gemma понякога обвива JSON в ```json ... ``` — изчисти
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())
```

## Оценка на ВИДЕО

Gemma 26B вижда изображения, не видео поток. Изваждай кадри РАВНОМЕРНО по цялото
видео и подавай като последователност.

**Колко кадъра:** за 5-сек клип (~80 кадъра @16fps) → **5 кадъра** е балансът
(добро покритие на движението, без да задавиш CPU Gemma / контекста).
3 = минимум за дрейф, 8+ = по-точно но бавно.

**ВАЖНО — равномерно по ЦЯЛОТО видео**, не струпани в началото. Затова смятаме
реалния step от общия брой кадри (наивното mod(n,30) пропуска края на клипа):

```python
import subprocess, tempfile, os

def extract_frames(video_path, n=5):
    """n равномерно разпределени кадъра по цялото времетраене."""
    tmpdir = tempfile.mkdtemp(prefix="eval_")
    total = int(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0", video_path
    ]).decode().strip())
    step = max(1, total // n)
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vf", f"select='not(mod(n\\,{step}))'",
        "-vsync", "0", "-frames:v", str(n),
        f"{tmpdir}/f%02d.png"
    ], check=True, capture_output=True)
    return sorted(os.path.join(tmpdir, f) for f in os.listdir(tmpdir))
```

**Какво Gemma МОЖЕ да прецени от кадри** (фокусирай промпта тук):
- случва ли се действието (различни пози между кадри = движение се е случило)
- консистентност лице/обекти (морфират ли между кадрите) ← силно
- артефакти/деформации (видими на статичен кадър)

**Какво НЕ може надеждно** (не питай за това):
- плавност/накъсаност на движението (губи се между семплите)
- скорост/темпо
- subtle motion (леко дишане може да не личи между 5 кадъра)

```python
def evaluate_video(video_path, task_description, n_frames=5):
    frames = extract_frames(video_path, n_frames)
    content = []
    for fp in frames:
        b64 = encode_image(fp)
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}})
    content.append({"type": "text", "text": f"""Това са {len(frames)} кадъра от видео клип (по ред).
Цел: "{task_description}"
Оцени КОНСИСТЕНТНОСТТА между кадрите (лице/обекти не трябва да морфират) и дали движението е правилно.
Върни същия JSON формат както за изображение, плюс:
  "temporal_consistent": bool   // запазват ли се лице/обекти между кадрите
"""})
    # ... същият payload с този content, същото парсване
```

## Капани

- **mmproj задължителен.** Ако verdict е безсмислен или "не виждам" → server без --mmproj.
- **Температура ниска (0.2).** Оценката трябва да е консистентна, не творческа.
- **JSON парсване tolerant.** Gemma понякога обвива в ```json или добавя коментар. Чисти преди json.loads.
- **Видео = кадри, не поток.** Темпоралните артефакти (трептене) се хващат частично; обективните (морфинг между кадри) се хващат добре чрез последователност от кадри.
- **VRAM:** оценката иска Gemma заредена → ComfyUI трябва да е разтоварен преди това (виж vram-manager). Vision заявка яде допълнителен VRAM за визуалните токени — при OOM намали n-gpu-layers.
- **Субективното е по-слабо.** action_correct/artifacts/identity → надеждни. score/mood → ориентир, не абсолют. При съмнение маркирай за човек.
