"""Metrics, reports, confusion matrices and TFLite conversion / latency."""
import statistics
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, recall_score)

from augment import make_ds


def metrics_on(model, X, y, classes, rare_classes):
    pred = model.predict(make_ds(X, y), verbose=0).argmax(1)
    rec = recall_score(y, pred, average=None, labels=np.arange(len(classes)))
    out = {"acc": round(accuracy_score(y, pred), 3),
           "macroF1": round(f1_score(y, pred, average="macro"), 3)}
    for c in rare_classes:
        out[f"rec_{c}"] = round(rec[classes.index(c)], 3)
    return out


def full_report(model, X, y, tag, classes, out_dir, title="Confusion matrix"):
    """Per-class precision/recall/F1 (csv) and confusion matrix (png)."""
    pred = model.predict(make_ds(X, y), verbose=0).argmax(1)
    rep = classification_report(y, pred, target_names=classes, digits=3, output_dict=True)
    print(f"\n=== {tag} ===")
    print(classification_report(y, pred, target_names=classes, digits=3))
    pd.DataFrame(rep).T.round(3).to_csv(f"{out_dir}/per_class_{tag}.csv")
    cm = confusion_matrix(y, pred, labels=np.arange(len(classes)))
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"{title}: {tag}")
    plt.tight_layout(); plt.savefig(f"{out_dir}/confusion_{tag}.png", dpi=150); plt.show()
    return rep


def tflite_convert(model, mode="float32"):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if mode == "dynamic":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    return conv.convert()


def _interp(b):
    it = tf.lite.Interpreter(model_content=b, num_threads=1)   # ONE CPU thread
    it.allocate_tensors()
    return it, it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]


def tflite_predict(b, X):
    it, i_in, i_out = _interp(b)
    out = []
    for i in range(len(X)):
        it.set_tensor(i_in, X[i:i + 1].astype(np.float32)); it.invoke()
        out.append(it.get_tensor(i_out)[0].argmax())
    return np.array(out)


def latency_ms(b, X, n=100, warmup=10):
    """Median and p95 single-image inference time (model only; no JPEG decode/resize)."""
    it, i_in, i_out = _interp(b)
    for i in range(warmup):
        it.set_tensor(i_in, X[i:i + 1].astype(np.float32)); it.invoke()
    t = []
    for i in range(n):
        x = X[i:i + 1].astype(np.float32); t0 = time.perf_counter()
        it.set_tensor(i_in, x); it.invoke(); _ = it.get_tensor(i_out)
        t.append((time.perf_counter() - t0) * 1000)
    t.sort()
    return statistics.median(t), t[int(0.95 * n) - 1]
