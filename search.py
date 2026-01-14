import numpy as np
import torch
import faiss
from PIL import Image
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

index = faiss.read_index("image_index.faiss")
paths = np.load("image_paths.npy", allow_pickle=True)

def search(query_path, k=5):
    q = img_embedding(query_path).reshape(1, -1).astype("float32")
    D, I = index.search(q, k)
    return [(paths[i], float(D[0][j])) for j, i in enumerate(I[0])]

if __name__ == "__main__":
    results = search("query.jpg", k=5)
    for r in results:
        print(r)
