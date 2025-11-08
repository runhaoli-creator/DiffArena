import argparse, os, numpy as np
from PIL import Image

def make_noise(h, w, c, kind="gauss", strength=8/255, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "gauss":
        z = rng.normal(0, 1, size=(c, h, w)).astype(np.float32)
        z = z / (np.abs(z).max() + 1e-6) * strength
    elif kind == "stripe":
        z = np.zeros((c, h, w), np.float32)
        z[:, :, ::8] = strength
        z[:, ::8, :] -= strength
    elif kind == "circle":
        y, x = np.ogrid[:h, :w]
        cy, cx = h//2, w//2
        r = np.sqrt((y-cy)**2 + (x-cx)**2)
        z = (np.cos(r/6.0)[None, ...] * strength).astype(np.float32)
        z = np.repeat(z, c, axis=0)
    else:
        raise ValueError(f"Unknown kind: {kind}")
    return z

def save_noise(noise, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, noise)
    print(f"✅ Saved {path} | shape={noise.shape} | range=({noise.min():.4f},{noise.max():.4f})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path to condition image (to get H,W)")
    ap.add_argument("--out", required=True, help="output .npy path")
    ap.add_argument("--kind", default="gauss", choices=["gauss","stripe","circle"])
    ap.add_argument("--strength", type=float, default=8/255)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    img = np.array(Image.open(args.image).convert("RGB"))
    h, w, c = img.shape[0], img.shape[1], 3
    noise = make_noise(h, w, c, kind=args.kind, strength=args.strength, seed=args.seed)
    save_noise(noise, args.out)
