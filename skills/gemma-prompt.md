# Gemma Prompt — storyboard и генериране на промпти

Как оркестраторът ползва Gemma 4 (текстов режим) за наративната предпродукция.

## Три задачи

1. **Storyboard** — идея → списък сцени
2. **Keyframe промпти** — сцена → FLUX.2 промпт за изображение
3. **Animation промпти** — сцена → Wan промпт за движение

Всичко връща structured JSON, за да го верижи оркестраторът.

## Storyboard

```python
def generate_storyboard(idea, n_scenes=None):
    n = f"около {n_scenes} сцени" if n_scenes else "толкова сцени колкото историята изисква"
    system = "Ти си режисьор и сценарист. Връщаш само валиден JSON."
    user = f"""Идея за видео: "{idea}"

Разбий на {n}. За всяка сцена върни JSON масив:
[
  {{
    "scene": 1,
    "description": "какво се случва в сцената (кратко, наративно)",
    "character": "кой герой е в кадъра (за консистентност)",
    "setting": "обстановка/локация",
    "mood": "емоция/тон"
  }}
]
Само JSON, без друг текст."""
    return _gemma_text(system, user, max_tokens=1500)
```

## Keyframe промпт (за FLUX.2 Klein)

ВАЖНО: FLUX.2 е буквален, без upsampling. Промптът трябва конкретика, не абстракции.
За маслена естетика — техниката ОТПРЕД (виж научените правила долу).

```python
def keyframe_prompt(scene, style="oil painting"):
    system = """Ти пишеш промпти за FLUX.2 Klein image модел. Правила:
- Техниката/стилът ОТПРЕД (impasto, palette knife, visible brushstrokes за маслено)
- Конкретни физически детайли, НЕ абстракции (не "symbolic atmosphere", "timeless mood")
- Структура: техника → субект → действие (механично описано) → среда → светлина → камера
- За действие: опиши МЕХАНИКАТА (hands pressing, arms bent), не само етикет
- Течлива проза, на английски
Връщаш само JSON."""
    user = f"""Сцена: {json.dumps(scene, ensure_ascii=False)}
Стил: {style}
Върни JSON: {{"positive": "...", "negative": "smooth, digital render, 3d, airbrushed, ..."}}"""
    return _gemma_text(system, user, max_tokens=600)
```

## Animation промпт (за Wan I2V)

ВАЖНО: Wan описва ДВИЖЕНИЕ, не статика. Едно доминиращо движение на клип.

```python
def animation_prompt(scene):
    system = """Ти пишеш промпти за Wan 2.2 image-to-video. Правила:
- Описвай ДВИЖЕНИЕ и КАМЕРА, не статичната сцена (тя вече е в keyframe-а)
- ЕДНО доминиращо движение на клип (не много едновременно — моделът се обърква)
- Завършвай с camera move + cinematic
Връщаш само JSON."""
    user = f"""Сцена: {json.dumps(scene, ensure_ascii=False)}
Какво движение трябва (леко, локализирано работи най-добре на 16GB).
Върни JSON: {{"positive": "...", "negative": "distorted, morphing, jitter, ..."}}"""
    return _gemma_text(system, user, max_tokens=500)
```

## Helper

```python
def _gemma_text(system, user, max_tokens=800):
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7   # по-висока за креативност (за разлика от оценката)
    }
    r = requests.post(GEMMA, json=payload, timeout=300)
    raw = r.json()["choices"][0]["message"]["content"]
    return parse_json_lenient(raw)  # същият от gemma-evaluate.md
```

## Научени правила за промпти (от практиката)

**FLUX.2 маслена живопис** — какво работи:
- impasto, palette knife, visible chunky brushwork, raised ridges, alla prima, canvas grain
- НЕ: glazing, refined, soft (това = гладко, обратното на груба четка)
- BASE модел (не distilled) за реална текстура; distilled винаги е по-плоско
- негатив активно: smooth, blended, digital render, airbrushed, waxy

**Действие в кадър** — FLUX е слаб за сложни действия:
- опиши механиката пространствено (hands pressing into dough, arms bent)
- ако пак не става → генерирай проста поза, остави движението на Wan
- CFG нагоре (5→6) кара модела да следва действието по-стриктно

**Идентичност** — НЕ чрез reference при всяка сцена (заключва композицията):
- генерирай сцените свободно (text-to-image)
- наложи лицето в post с ReActor swap към master
- reference editing само за "същата композиция, лек edit"

## Капани

- **Температура:** генерация 0.7 (креативно), оценка 0.2 (консистентно). Не ги бъркай.
- **JSON tolerant парсване** — както в gemma-evaluate.md.
- Запиши генерираните промпти в state-а — за retry трябва оригинала + fix_suggestion.
