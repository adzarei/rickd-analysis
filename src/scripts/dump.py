# Test Set
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X_ts, y, groups=subject_id))

X_ts_train, X_ts_test = X_ts[train_idx], X_ts[test_idx]
X_meta_train, X_meta_test = X_meta[train_idx], X_meta[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
sub_id_train, sub_id_test = subject_id[train_idx], subject_id[test_idx]

# Validation Set
cv_gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE_2)
cv_train_idx, cv_val_idx = next(cv_gss.split(X_ts_train, y_train, groups=sub_id_train))
X_ts_cv_train, X_ts_cv_val = X_ts_train[cv_train_idx], X_ts_train[cv_val_idx]
X_meta_cv_train, X_meta_cv_val = X_meta_train[cv_train_idx], X_meta_train[cv_val_idx]
y_cv_train, y_cv_val = y_train[cv_train_idx], y_train[cv_val_idx]
sub_id_cv_train, sub_id_cv_val = sub_id_train[cv_train_idx], sub_id_train[cv_val_idx]




# Train-only normalization
Xts_train = X_ts[tr_idx].astype("float32")
Xts_val = X_ts[va_idx].astype("float32")
Xts_test = X_ts[te_idx].astype("float32")

Xmeta_tr = X_meta[tr_idx].astype("float32")
Xmeta_va = X_meta[va_idx].astype("float32")
Xmeta_te = X_meta[te_idx].astype("float32")

y_tr = y[tr_idx].astype("float32")
y_va = y[va_idx].astype("float32")
y_te = y[te_idx].astype("float32")

# Per-channel, per-time normalization (broadcast-safe):
ts_mean = Xts_tr.mean(axis=0, keepdims=True)  # (1,101,54)
ts_std  = Xts_tr.std(axis=0, keepdims=True) + 1e-6
Xts_tr = (Xts_tr - ts_mean) / ts_std
Xts_va = (Xts_va - ts_mean) / ts_std
Xts_te = (Xts_te - ts_mean) / ts_std

meta_mean = Xmeta_tr.mean(axis=0, keepdims=True)
meta_std  = Xmeta_tr.std(axis=0, keepdims=True) + 1e-6
Xmeta_tr = (Xmeta_tr - meta_mean) / meta_std
Xmeta_va = (Xmeta_va - meta_mean) / meta_std
Xmeta_te = (Xmeta_te - meta_mean) / meta_std

# 3) Class weights for imbalance (70/30)
cw = compute_class_weight(class_weight='balanced', classes=np.array([0,1]), y=y_tr.astype(int))
class_weight = {0: float(cw[0]), 1: float(cw[1])}

# 4) Train in batches (just set batch_size)
callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor='val_pr_auc', patience=4, factor=0.5, min_lr=1e-5, verbose=1),
    keras.callbacks.EarlyStopping(monitor='val_pr_auc', patience=10, restore_best_weights=True, verbose=1),
]

history = model.fit(
    {"ts": Xts_tr, "meta": Xmeta_tr},
    y_tr,
    validation_data=({"ts": Xts_va, "meta": Xmeta_va}, y_va),
    epochs=60,
    batch_size=256,                 # ← pick what fits your GPU/CPU
    class_weight=class_weight,      # or use sample_weight if you prefer
    shuffle=True,
    callbacks=callbacks,
    verbose=2,
)

# 5) Test
model.evaluate({"ts": Xts_te, "meta": Xmeta_te}, y_te, verbose=0)