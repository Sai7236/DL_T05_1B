"""Experiment runner: trains, evaluates on VALIDATION only, appends to a CSV log."""
import pathlib
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

from augment import make_ds


class Experiment:
    def __init__(self, X_train, y_train, X_val, y_val, classes, rare_classes,
                 log_path, augment=None, seed=42):
        self.X_train, self.y_train, self.X_val, self.y_val = X_train, y_train, X_val, y_val
        self.num_classes = len(classes)
        self.rare_classes = list(rare_classes)
        self.rare_idx = [classes.index(c) for c in self.rare_classes]
        self.augment, self.seed = augment, seed
        cw = compute_class_weight("balanced", classes=np.arange(self.num_classes), y=y_train)
        self.class_weights = {i: float(w) for i, w in enumerate(cw)}
        self.log_path = pathlib.Path(log_path)
        self.results = pd.read_csv(self.log_path).to_dict("records") if self.log_path.exists() else []

    def evaluate_val(self, model):
        pred = model.predict(make_ds(self.X_val, self.y_val), verbose=0).argmax(1)
        rec = recall_score(self.y_val, pred, average=None, labels=np.arange(self.num_classes))
        out = {"val_acc": accuracy_score(self.y_val, pred),
               "val_macroF1": f1_score(self.y_val, pred, average="macro")}
        for c, i in zip(self.rare_classes, self.rare_idx):
            out[f"recall_{c}"] = rec[i]
        return out

    def fit(self, name, model, epochs, lr, augment_on=True, use_cw=True, patience=4):
        model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                               restore_best_weights=True)]
        t0 = time.time()
        aug = self.augment if augment_on else None
        hist = model.fit(make_ds(self.X_train, self.y_train, True, aug, seed=self.seed),
                         validation_data=make_ds(self.X_val, self.y_val),
                         epochs=epochs, callbacks=cb,
                         class_weight=self.class_weights if use_cw else None, verbose=2)
        row = {"experiment": name, "augment": augment_on, "class_weights": use_cw,
               "params": model.count_params(),
               "approx_size_MB_fp32": round(model.count_params() * 4 / 1e6, 2),
               "epochs_run": len(hist.history["loss"]),
               "final_train_acc": hist.history["accuracy"][-1],
               "train_minutes": round((time.time() - t0) / 60, 1)}
        row.update(self.evaluate_val(model))
        self.results.append(row)
        pd.DataFrame(self.results).to_csv(self.log_path, index=False)
        return hist
