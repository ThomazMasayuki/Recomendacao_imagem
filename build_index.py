import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import faiss
import open_clip

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
model = model.to(DEVICE).eval()

def img_embedding(path):
    img = Image.open(path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = model.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu().numpy().astype("float32")

catalog_dir = "data/catalog"
paths = [os.path.join(catalog_dir, f) for f in os.listdir(catalog_dir)
         if f.lower().endswith((".jpg",".jpeg",".png",".webp"))]

embs = []
for p in tqdm(paths):
    embs.append(img_embedding(p))

embs = np.vstack(embs).astype("float32")

d = embs.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embs)

faiss.write_index(index, "image_index.faiss")
np.save("image_paths.npy", np.array(paths))

print("Index criado com", len(paths), "imagens.")
