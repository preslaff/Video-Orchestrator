# VRAM Manager — управление на паметта между моделите

16GB е тясно. Този skill определя как се избягва OOM при суичване между Gemma и ComfyUI.

## Проблемът

- Klein 4B fp8 ~8GB
- Wan dual-model (High+Low) Q5 ~22GB — влиза само с offload към 64GB RAM
- Gemma 4 Q4 + mmproj — **измерено 1838 MiB дори с `--n-gpu-layers 0`**
- ReActor — лек (~2GB)
- **OmniVoice-BG (narration) — измерено 11268 MiB.** Не се събира с Wan. Виж `NOTES.md` §4
  за пълната измерена таблица и `docs/services.md` за последствията.

Два тежки модела НЕ се събират едновременно. Gemma (за оценка) и ComfyUI модел
не могат да са заредени заедно.

## Две стратегии (избор в NOTES.md раздел 7)

### Опция B — Gemma постоянно на CPU (ПРЕПОРЪЧАНА за простота)

Gemma тече на CPU (A4B активира само 4B → поносимо), ComfyUI владее GPU цял.
Нула суичване, нула OOM конфликт.

Старт на Gemma БЕЗ GPU слоеве: **командата е в `README.md` → "Running the services"** и
това е единственото ѝ копие. (Тук имаше вариант с `--ctx-size 16384` — остарял.)

⚠️ `--n-gpu-layers 0` НЕ значи нула VRAM: **измерено 1838 MiB** остават заети от CUDA контекста
и projector-а. "Gemma на CPU" важи за изчислението, не за паметта.
Цена: оценката е по-бавна (CPU). Но не е time-critical, и логиката е тривиална —
двата свята не се бият. ComfyUI работи на пълна VRAM скорост.

### Опция A — суичване (по-бързо когато работи, но крехко)

Gemma и ComfyUI се редуват на GPU. Изисква дисциплина:

```python
def free_comfyui():
    requests.post("http://localhost:8188/free",
                  json={"unload_models": True, "free_memory": True})

# Gemma на GPU режим — но трябва да я рестартираш или да ползваш
# llama-server endpoint за unload (по-сложно). Затова Опция B е по-чиста.
```

Проблем: llama-server не разтоварва лесно "на живо". Реално значи или рестарт на
Gemma server между фазите, или Gemma на CPU. → затова Опция B печели за простота.

## Правила за оркестратора

1. **Преди тежка ComfyUI генерация** → увери се че предишният модел е разтоварен:
   `POST /free` с unload_models преди да смениш Klein→Wan.

2. **Wan е най-тежък** — преди Wan фазата, изпразни всичко друго от VRAM.

3. **Vision оценка на Gemma** (ако Опция A) → ComfyUI трябва вече да е свободен.
   При Опция B няма проблем — Gemma е на CPU отделно.

4. **При OOM** — диагностика по ред:
   - Wan: намали length (49→25) или резолюция
   - Klein: вече лек, рядко OOM
   - Gemma vision: намали n-gpu-layers или ctx-size; vision токените ядат extra

5. **Фазирай за минимум зареждания** (NOTES.md раздел 5) — всички Klein наведнъж,
   после `/free`, после всички Wan, после `/free`, после ReActor.

## Препоръка за финализиране

Започни с **Опция B (Gemma на CPU)**. Тествай дали скоростта на оценка е поносима.
Ако оценката е твърде бавна и пречи на workflow-а → тогава инвестирай в Опция A
с рестарт-базирано суичване. Но почти сигурно B е достатъчно — оценката е
секунди-до-минута на CPU за A4B, а генерацията (минути) доминира времето така или иначе.

## Мониторинг

```python
def gpu_free_mb():
    import subprocess
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free",
         "--format=csv,noheader,nounits"]).decode().strip()
    return int(out.split("\n")[0])

# преди тежка операция:
if gpu_free_mb() < 10000:
    free_comfyui()
    time.sleep(3)
```
