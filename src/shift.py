"""Simulated field-shift proxy sets (NOT real field photographs).
Set A: JPEG, colour cast, haze, noise, blur (overlaps E7's training degradations).
Set B: motion blur, salt-and-pepper, harsh JPEG, low light (related but different).
Seeds used in the project: val A=SEED, val B=SEED+1, test A=SEED+2, test B=SEED+3."""
import io

import cv2
import numpy as np
from PIL import Image, ImageFilter


def degrade_A(img_uint8, rng):
    img = Image.fromarray(img_uint8)
    kind = rng.choice(["jpeg", "cast", "haze", "noise", "blur"])
    if kind == "jpeg":
        buf = io.BytesIO(); img.save(buf, "JPEG", quality=int(rng.integers(8, 25)))
        img = Image.open(buf).convert("RGB")
    elif kind == "cast":
        a = np.asarray(img, dtype=np.float32) * rng.uniform(0.75, 1.25, size=3)
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    elif kind == "haze":
        a = np.asarray(img, dtype=np.float32)
        t = rng.uniform(0.25, 0.5)
        img = Image.fromarray(np.clip(a * (1 - t) + 200 * t, 0, 255).astype(np.uint8))
    elif kind == "noise":
        a = np.asarray(img, dtype=np.float32)
        a += rng.normal(0, rng.uniform(10, 25), a.shape)
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    else:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(1.5, 3.0)))
    return np.asarray(img, dtype=np.uint8), str(kind)


def degrade_B(img_uint8, rng):
    kind = rng.choice(["motion_blur", "salt_pepper", "jpeg_harsh", "low_light"])
    a = img_uint8.copy()
    if kind == "motion_blur":
        L = 2 * int(rng.integers(3, 8)) + 1
        k = np.zeros((L, L), np.float32); k[L // 2, :] = 1
        M = cv2.getRotationMatrix2D((L // 2, L // 2), rng.uniform(0, 180), 1)
        k = cv2.warpAffine(k, M, (L, L)); k /= k.sum()
        a = cv2.filter2D(a, -1, k, borderType=cv2.BORDER_REFLECT)
    elif kind == "salt_pepper":
        amt = rng.uniform(0.01, 0.04)
        m = rng.random(a.shape[:2])
        a[m < amt / 2] = 0; a[m > 1 - amt / 2] = 255
    elif kind == "jpeg_harsh":
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, "JPEG", quality=int(rng.integers(4, 12)))
        a = np.asarray(Image.open(buf).convert("RGB"))
    else:
        g, s = rng.uniform(1.4, 2.0), rng.uniform(0.6, 0.9)
        a = np.clip(255 * (a / 255.0) ** g * s, 0, 255).astype(np.uint8)
    return a, str(kind)


def make_shift_set(X, which, seed):
    """Degrade every image in X (in order) with its own seeded generator."""
    rng = np.random.default_rng(seed)
    fn = degrade_A if which == "A" else degrade_B
    out = [fn(x, rng) for x in X]
    return np.stack([o[0] for o in out]), np.array([o[1] for o in out])
