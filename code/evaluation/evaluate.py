# ============================================================
# 📊 GenArtist / Diffusion Image Evaluation Script for Colab
# Metrics: AttributeBinding, Color, Shape, Texture, Spatial, NonSpatial
# ============================================================

# === 1️⃣ Setup ===
#!pip install openai==1.37.1 ftfy regex tqdm pillow pandas --quiet
#!pip install git+https://github.com/openai/CLIP.git --quiet

# === Define image evaluation paths ===
# Put your generated images and prompts here
import os
IMAGES_DIR = os.environ.get("RAGARTIST_IMAGES_DIR", "Images")
output_path = os.environ.get("RAGARTIST_EVALUATION_CSV", "evaluation_results.csv")


import os, base64, torch, clip, json, time
import pandas as pd
from tqdm import tqdm
from io import BytesIO
from PIL import Image
import openai

from transformers import AutoProcessor, AutoModel
import torch
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

# OpenAI Connection
openai.api_key = os.environ.get("OPENAI_API_KEY")
gpt_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Load CLIP model for low-level visual metrics ===
model, preprocess = clip.load("ViT-L/14", device=device)

folders = os.listdir(IMAGES_DIR)
print(f'Processing {len(folders)} set of images...')
evaluations = []
outCsv = pd.read_csv(output_path) if os.path.exists(output_path) else pd.DataFrame()
if len(outCsv)>0: outCsv = outCsv.drop_duplicates(subset=['file'])
evaluated_files = set(outCsv['file']) if 'file' in outCsv.columns else set()

for folder in folders:
    if '.' in folder[:-4]: continue

    folderPath = IMAGES_DIR+'/'+folder
    images = os.listdir(folderPath)
    print(f'Processing {len(images)} with prompt: {folder}')
    if images:
        with open(folderPath + '/0.txt') as txtPrompt:
            txtPrompt = list(txtPrompt)
        caption = txtPrompt[0].strip().split('promt:')[-1]

        if folder+ "/" + "0.png" not in evaluated_files:
            evaluations.append({"file": folder+ "/" + "0.png", "prompt": caption})
            images = [x for x in images if '_' not in x and x.endswith('.png')]
            if images[0] != '0.png':
                evaluations.append({"file": folder+ "/" + images[0], "prompt": caption})

# === CLIP-based scoring (Color, Shape, Texture) ===
def clip_similarity(img_path, text_list):
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    scores = {}
    for t in text_list:
        text_features = model.encode_text(clip.tokenize([t]).to(device))
        text_features /= text_features.norm(dim=-1, keepdim=True)
        sim = (image_features @ text_features.T).item()
        scores[t] = sim
    return scores

# === SigLIP-based Image Evaluation for Caption Similarity ===
def siglip_similarity(caption, image_paths, model_name="google/siglip-base-patch16-384", device=None):
    """
    Compute SigLIP similarity scores between a text caption and one or more images.

    Args:
        caption (str): Text caption (can be Persian, English, etc.)
        image_paths (list[str] or str): Path or list of image paths.
        model_name (str): Hugging Face SigLIP model checkpoint.
        device (str): "cuda" or "cpu" (auto-detect if None)

    Returns:
        dict: {image_path: similarity_score (float, cosine 0–1)}
    """
    if isinstance(image_paths, str): image_paths = [image_paths]

    # auto device selection
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # load model and processor
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # prepare text inputs
    text_inputs = processor(text=[caption], return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = F.normalize(text_features, p=2, dim=-1)

    results = {}
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = F.normalize(image_features, p=2, dim=-1)

        similarity = (image_features @ text_features.T).item()
        results[img_path] = round(similarity, 4)

    return results

# === GPT-based reasoning scoring ===
def gpt5_evaluate(img_path, prompt):
    # resize & encode image
    img = Image.open(img_path)
    img.thumbnail((512, 512))
    buf = BytesIO(); img.save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    eval_prompt = (
        "You are an image evaluation expert.\n"
        "Rate the image (0–5) on:\n"
        "1. AttributeBinding\n2. Spatial\n3. NonSpatial\n\n"
        f"Prompt: {prompt}\n\n"
        "Return ONLY valid JSON like "
        '{"AttributeBinding": float, "Spatial": float, "NonSpatial": float}'
    )

    r = gpt_client.responses.create(
        model="gpt-5-pro",           # or gpt-5, gpt-5-mini
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": eval_prompt},
                {"type": "input_image",
                 "image_url": f"data:image/png;base64,{b64}"}
            ],
        }],
        max_output_tokens=300,
        #temperature=0,
    )

    # GPT-5 responses object
    output = r.output[0].content[0].text
    try:
        return json.loads(output)
    except Exception:
        print("⚠️ Could not parse output:\n", output)
        return {"AttributeBinding": 0, "Spatial": 0, "NonSpatial": 0}


def gpt_evaluate(img_path, prompt, max_retries=3):

    def encode_image(img_path, max_size=512):
        img = Image.open(img_path)
        img.thumbnail((max_size, max_size))
        buf = BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode()
    b64 = encode_image(img_path)

    eval_prompt = (
        "You are an image evaluation expert. Images given are generated for Iran's culture and ecosystem.\n"
        "Rate the image (0–5) on these aspects:\n"
        "1. AttributeBinding (attribute–object consistency)\n"
        "2. Spatial (correct spatial relationships)\n"
        "3. NonSpatial (semantic or causal relationships)\n\n"
        f"Prompt: {prompt}\n\n"
        "Return ONLY valid JSON, no text commentary:\n"
        '{"AttributeBinding": <float>, "Spatial": <float>, "NonSpatial": <float>}'
    )

    for attempt in range(max_retries):
        try:
            r = gpt_client.chat.completions.create(
                model= "gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": eval_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        ],
                    }
                ],
                #temperature=0.0,
                max_completion_tokens=300,
                response_format={"type": "json_object"},  # ⬅️ Force JSON mode
            )

            content = r.choices[0].message.content.strip()
            if not content:
                raise ValueError("Empty content returned.")
            data = json.loads(content)
            return data

        except Exception as e:
            print(f"[Retry {attempt+1}] Error: {e}")
            time.sleep(1)

    print("❌ GPT evaluation failed after retries.")
    return {"AttributeBinding": 0, "Spatial": 0, "NonSpatial": 0}


# === 6️⃣ Run evaluation ===
results = []

for e in tqdm(evaluations):
    f = os.path.join(IMAGES_DIR, e["file"])
    p = e["prompt"]

    # --- CLIP sub-prompts for low-level metrics ---
    clip_subs = [f"{p[:75]}, focus on {feature}" for feature in ['color','shape','texture']]
    clip_scores = clip_similarity(f, clip_subs)

    # ---SigLip Similarity
    #sigLip = siglip_similarity(e['prompt'], e['file'])

    # --- GPT reasoning metrics ---
    gpt_scores = gpt_evaluate(f, p) #gpt5_evaluate(f,p)

    results.append({
        "file": e["file"], "prompt": p,
        "Color": clip_scores[clip_subs[0]],
        "Shape": clip_scores[clip_subs[1]],
        "Texture": clip_scores[clip_subs[2]],
        #"SigLip": sigLip.values[0],  #AA#> SigLip Also has a 64 token limit while that of CLIP is 77.
        **gpt_scores
    })
    print(results[-1],'\n')

# === 7️⃣ Save results to CSV ===
print(f'Evaluation results were generated for {len(results)} images')
df = pd.DataFrame(results)
df["Complex"] = df[["AttributeBinding","Color","Shape","Texture","Spatial","NonSpatial"]].mean(axis=0)
outCsv = pd.concat([outCsv,df])
outCsv.to_csv(output_path, index=False,encoding='utf-8-sig')

print("\n✅ Evaluation complete! CSV saved at:", output_path)


"""
# === Visualization: Radar Chart ===
metrics = ["AttributeBinding", "Color", "Shape", "Texture", "Spatial", "NonSpatial"]
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # close the circle

plt.figure(figsize=(8,8))
ax = plt.subplot(111, polar=True)

for model_name, group in df.groupby("Model"):
    avg_scores = [group[m].mean() for m in metrics]
    avg_scores += avg_scores[:1]
    ax.plot(angles, avg_scores, label=model_name)
    ax.fill(angles, avg_scores, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=12)
ax.set_yticklabels([])
ax.set_title("🕸️ Image Evaluation Radar Chart", fontsize=16, pad=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
plt.show()

"""
