"""Data loading, stratified splitting with leakage check, and image caching."""
import hashlib
import pathlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


def build_table(train_dir):
    """List every labelled image (Paddy Doctor: one sub-folder per class)."""
    rows = []
    for cls_dir in sorted(pathlib.Path(train_dir).iterdir()):
        if cls_dir.is_dir():
            for p in sorted(cls_dir.glob("*.jpg")):
                rows.append((str(p), cls_dir.name))
    return pd.DataFrame(rows, columns=["path", "label"])


def _md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def make_splits(df, seed):
    """70/15/15 stratified split. Exact-duplicate files (MD5) that appear in
    more than one split are removed from train/val; test is never changed.
    NOTE: the split files saved in splits/*.csv are the authoritative split
    used for all reported results. Re-creating the split can differ slightly
    because file listing order can differ between machines."""
    df = df.copy()
    classes = sorted(df["label"].unique())
    df["y"] = df["label"].map({c: i for i, c in enumerate(classes)})

    train_val, test = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=seed)
    train, val = train_test_split(train_val, test_size=0.15 / 0.85,
                                  stratify=train_val["label"], random_state=seed)
    for d in (train, val, test):
        d["md5"] = d["path"].map(_md5)

    info = {
        "duplicates_inside_test": int(test["md5"].duplicated().sum()),
        "trainval_identical_to_test": int(train["md5"].isin(test["md5"]).sum() + val["md5"].isin(test["md5"]).sum()),
        "train_identical_to_val": int(train["md5"].isin(val["md5"]).sum()),
    }
    train = train[~train["md5"].isin(test["md5"])]
    val = val[~val["md5"].isin(test["md5"])]
    train = train[~train["md5"].isin(val["md5"])]
    assert not set(train["md5"]) & set(test["md5"])
    assert not set(val["md5"]) & set(test["md5"])
    assert not set(train["md5"]) & set(val["md5"])
    info["final_sizes"] = (len(train), len(val), len(test))
    return train, val, test, classes, info


def _load_resized(path, img):
    im = Image.open(path).convert("RGB").resize((img, img), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def build_arrays(d, img=224, workers=8):
    """Resize every image once and keep it as a uint8 array (fast experiments)."""
    with ThreadPoolExecutor(workers) as ex:
        X = np.stack(list(ex.map(lambda p: _load_resized(p, img), d["path"])))
    return X, d["y"].values.astype("int32")
