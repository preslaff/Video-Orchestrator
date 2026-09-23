# Поправка на дефекти — кой инструмент за кой дефект

Как се поправя сбъркан кадър (деформирани ръце, разпаднали се дръжки, разкривен надпис) с
инструментите, които vorch вече има. Правилата за промпти са в `skills/gemma-prompt.md`,
текстът в клипове — в `docs/label-fidelity.md`. Този файл казва **кога кое**.

## Първо: носи ли reference-ът дефекта?

Преди да пипнеш промпта, погледни reference-а със същото внимание, с което гледаш кадъра. Ако
дефектът е и в него, моделът възпроизвежда вярно това, което си му показал — никакъв промпт,
workflow, seed или CFG няма да го оправи. Трябва **нов reference**.

Реален случай (job `644f92a3`, сцена 2): дръжките на чантата не се захващат за тялото ѝ.
Reference-ът `e38fbb55` е същият кадър, със същата двусмислена дръжка. Плюс промпт
`fix the bag handels`, `edit_workflow=''` (пада на `identity_edit`) и `edit_base=''`. Четири
независими причини, всяка от които сама по себе си стига да провали поправката.

```bash
JOB=644f92a3...; python3 - <<'PY'
import json, urllib.request
j = json.load(urllib.request.urlopen(f"http://localhost:8000/api/jobs/{JOB}"))
s = j["scenes"][1]
print(s["keyframe_url"], s["reference_asset_ids"], s["mode"], s["edit_workflow"], s["edit_base"])
PY
```

## Класификация на дефекта

- **структурен/геометричен** — дръжки, каишки, ръце, пръсти, топология на предмета
- **печатен текст / бранд** — етикет, лого, надпис
- **идентичност** — грешно лице
- **стил / светлина / композиция**
- **логика на сцената** — предмети се сливат, висят във въздуха, сянка без източник

Класът избира инструмента. Не започвай от промпта.

## Кой инструмент за кой дефект

| Дефект | Инструмент | Настройки |
|---|---|---|
| Детайл сбъркан, композицията е добра | `kv_edit` върху собствения кадър | `mode=edit`, `edit_workflow=kv_edit`, `edit_base=keyframe` |
| Геометрията на предмета е грешна навсякъде | коригиращ reference + `identity_edit` | `mode=edit`, `reference_asset_ids=[чиста снимка]` |
| Самата композиция е грешна | пълна регенерация | `mode=t2i`, преработен промпт |
| Само движението е сбъркано | реанимация | `what=video`, нов `animation_prompt` |
| Бранд текст, статичен кадър | композит отвън, после `keyframe_asset_id` | kv_edit доказано разваля шрифт |
| Бранд текст, клип | locked regions | `PUT /scenes/{i}/locks` |
| Грешно лице | ReActor | `identity_asset_id` на сцената |
| По-стар дубъл е бил по-добър | избор на версия | `chosen_variant_id` / `chosen_clip_id` — безплатно |
| Нищо не помага | прекадрирай | дефектът излиза извън кадър |

## Промпт за поправка

**Никога "fix", "correct", "repair".** Моделът не вижда какво е сбъркано в момента — той рисува
това, което описваш. Описвай **крайното състояние**.

```text
лошо:  fix the bag handels
добро: two slim rolled leather handles rise from the tote and are stitched flat onto the front
       panel, each ending in a rectangular anchor tab with two rivets; her fingers close around
       both handles together
```

Ред на думите (от `_SYSTEM_KEYFRAME`): техника → субект → действие (**механика**) → среда →
светлина → камера. Ескалация от `skills/gemma-prompt.md`: опиши механиката пространствено → ако
пак не става, опрости позата и остави движението на Wan → CFG 5→6 за по-стриктно следване.

Когато прилагаш съвета на оценителя, форматът е един и същ навсякъде (`pipeline.py:202`,
`orchestrator.py:299`, `GatePanel.tsx:353`) — или просто подай `apply_suggestion: true`:

```text
"{keyframe_prompt}. Correction: {fix_suggestion}"
```

## Избор на reference

- Reference-ът трябва да **показва правилното нещо** — не същия дефект, не "нещо подобно".
- `fit: same_shot` → edit mode е безопасен. `motif_only` (същият субект, друго кадриране) →
  едитът ще заключи чужда композиция; по-добре `t2i`.
- Резолюция ≥ размера на рендера. `media.fit_image` прави cover-fit + централен кроп, така че
  различен aspect мълчаливо отрязва страните.
- И двата edit workflow-а имат **точно 2 слота**. При `edit_base=keyframe` слот 0 е самият кадър,
  слот 1 е твоят reference. Трети се игнорира мълчаливо (с `warning` решение в лога).
- `kv_edit` взема изходния размер от **първия** reference — няма `width`/`height`.

Ако нищо в проекта не показва предмета правилно, спри и поискай снимка. Кажи точно каква:

> три-четвърти изглед на чантата, дръжките на фокус, ясно се вижда мястото където дръжката се
> захваща за тялото, ≥1024 px

```bash
python -m vorch project add-asset <slug> <path> --kind product --label "tote, handle anchor"
```

## Негативи

`DEFAULT_KEYFRAME_NEGATIVE` е `smooth, digital render, 3d, airbrushed`; social профилите добавят
само `text, watermark`. **Никой профил не споменава ръце, пръсти или деформации** — потискането
на артефакти днес стои изцяло на retry-with-correction. Добавяй ръчно според класа:

- ръце/пръсти: `deformed hands, extra fingers, fused fingers, missing fingers, twisted wrist`
- предмети/дръжки: `disconnected strap, floating handle, melted leather, merged geometry`
- общо: `warped, malformed, duplicated limbs`

**`kv_edit` не приема `negative`** (`workflows/kv_edit.yaml`). Негативи работят само при `t2i` и
`identity_edit` — при kv_edit цялата тежест е върху позитивния промпт.

## Проверка

Съди по обективната порта, не по `score`:

```text
action_correct ∧ identity_match ∧ composition_ok ∧ ¬artifacts ∧ framing_ok is not False
```

`artifacts: true` значи **лошо**. `score` и `style_match` нарочно не влизат в портата — те само
маркират за човешки преглед (`NOTES.md` раздел 6).

```bash
python -m vorch evaluate <image> "<какво трябва да показва кадърът>"   # exit 2 = retry
```

И сравни кропове в натурален размер преди/след. Окото хваща неща, които оценителят пропуска —
особено при малки детайли, където Gemma е слаба.

## Капани

- **Репетирай, преди да похарчиш.** `python -m vorch edit "<prompt>" --ref <кадър> --ref
  <reference> --workflow kv_edit` прави еднократен рендер **без** да пипа job-а и без да отваря
  gate. Чак когато резултатът е добър, прилагай към сцената.
- **`identity_edit` заключва композицията** (инвариант 5) и е грешният инструмент за "запази
  този кадър". `identity_edit` кондиционира нов рендер върху reference-ите; `kv_edit` запазва
  базовото изображение и променя само това, което промптът иска (`state.py:149`). За поправка
  на съществуващ кадър → `kv_edit` + `edit_base=keyframe`.
- **kv_edit не връща бранд текст.** Проверено: дава `BEOWER BHORMNCE` (`docs/label-fidelity.md`).
  За текст → locked regions при клип, композит отвън при статичен кадър.
- **Регенерирането на keyframe трие `locked_regions`**, ако маркировките вече не пасват на новия
  кадър (`orchestrator.py:320`). Маркирай наново след поправка.
- **409 докато тече сегмент.** `POST /api/jobs/{id}/stop`, изчакай `active:false`, чак тогава
  редактирай.
- **Най-евтината поправка често не е рендер, а избор.** `chosen_variant_id` / `chosen_clip_id`
  връщат по-стар, по-добър дубъл безплатно — историята се пази цялата.
- Температура 0.7 генерация / 0.2 оценка — не ги бъркай. Оценката иска Gemma заредена; освободи
  ComfyUI VRAM първо (`skills/vram-manager.md`).
