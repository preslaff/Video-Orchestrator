# ComfyUI API — пускане на workflow програмно

Как оркестраторът зарежда workflow шаблон, заменя параметри, изпълнява и взима резултата.

## Основен поток

1. Зареди workflow JSON от `workflows/` (API формат — node-овете са плосък dict по ID)
2. Замени само нужните полета (по node ID — виж NOTES.md раздел 3)
3. `POST http://localhost:8188/prompt` с `{"prompt": <workflow_dict>}`
4. Вземи `prompt_id` от отговора
5. Poll `GET /history/{prompt_id}` докато се появи (значи готово)
6. Извлечи изходните файлове от history-то (path-ове в output/)

## Псевдокод

```python
import json, requests, time, uuid

COMFY = "http://localhost:8188"

def load_workflow(path):
    with open(path) as f:
        return json.load(f)

def run_workflow(wf: dict, timeout=600):
    client_id = str(uuid.uuid4())
    r = requests.post(f"{COMFY}/prompt",
                      json={"prompt": wf, "client_id": client_id})
    r.raise_for_status()
    prompt_id = r.json()["prompt_id"]

    # poll за готовност
    start = time.time()
    while time.time() - start < timeout:
        h = requests.get(f"{COMFY}/history/{prompt_id}").json()
        if prompt_id in h:
            return extract_outputs(h[prompt_id])
        time.sleep(2)
    raise TimeoutError(f"Workflow {prompt_id} не завърши за {timeout}s")

def extract_outputs(history_entry):
    # изходите са в history_entry["outputs"][node_id]
    # изображения: ["images"], видео: ["gifs"] или ["videos"]
    files = []
    for node_id, out in history_entry["outputs"].items():
        for key in ("images", "gifs", "videos"):
            for item in out.get(key, []):
                files.append(item)  # {filename, subfolder, type}
        return files
```

## Замяна на параметри (примери по NOTES.md раздел 3)

```python
# Klein text-to-image: смени промпт + seed + размери
wf = load_workflow("workflows/image_flux2_klein_text_to_image.json")
wf["76"]["inputs"]["value"] = positive_prompt
wf["75:73"]["inputs"]["noise_seed"] = seed
wf["75:68"]["inputs"]["value"] = width
wf["75:69"]["inputs"]["value"] = height

# Wan I2V: смени start_image + анимационен промпт + дължина
wf = load_workflow("workflows/wan22_i2v_gguf.json")
wf["5"]["inputs"]["image"] = keyframe_filename   # трябва да е в input/
wf["6"]["inputs"]["text"] = animation_prompt
wf["8"]["inputs"]["length"] = 49   # 4n+1!

# ReActor: смени source (master) + input (сцена)
wf = load_workflow("workflows/ReActor_Face_Swap.json")
wf["2"]["inputs"]["image"] = master_face        # source
wf["1"]["inputs"]["image"] = scene_image         # input
```

## Подаване на изображение към ComfyUI

LoadImage node-овете четат от `~/comfy/ComfyUI/input/`. За да подадеш генериран файл към следваща стъпка:
- или копирай файла в input/ и подай само името
- или ползвай `POST /upload/image` (multipart) — качва в input/, връща името

```python
def upload_image(filepath):
    with open(filepath, "rb") as f:
        r = requests.post(f"{COMFY}/upload/image",
                          files={"image": f})
    return r.json()["name"]  # това име подаваш на LoadImage
```

## VRAM освобождаване

Между тежки workflow-и (Klein → Wan), изпразни VRAM:
```python
requests.post(f"{COMFY}/free", json={"unload_models": True, "free_memory": True})
```
Това е ключово за 16GB — виж VRAM мениджъра.

## Капани

- **API формат ≠ UI формат.** Workflow-ите в `workflows/` са API export (плосък dict). UI export-ът (с "nodes" масив) НЕ работи с /prompt. Ако зареждаш нов workflow, увери се че е API формат.
- **Wan dual-model** е два UnetLoaderGGUF + два KSamplerAdvanced в каскада. И двата трябва да са в graph-а; не пропускай LowNoise.
- **Изходните файлове са с автоинкремент** (FaceSwap_00001, _00002...). Не приемай фиксирано име — чети реалното от history-то.
- **Видеото е в output/video/** подпапка (заради filename_prefix "video/...").
