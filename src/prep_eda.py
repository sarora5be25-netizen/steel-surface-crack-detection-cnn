import os, glob, hashlib, json, shutil, sys
import numpy as np, cv2, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

RAW = sys.argv[1]          # folder containing all original NEU-DET jpgs
OUT = sys.argv[2]          # work folder
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
SIZE = 128
plt.rcParams.update({"font.size": 10, "figure.dpi": 130})

# 1. load + clean
rows, seen = [], {}
for p in sorted(glob.glob(os.path.join(RAW, "*.jpg"))):
    name = os.path.basename(p)
    cls = name.rsplit("_", 1)[0]
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    ok = img is not None
    h = hashlib.md5(open(p, "rb").read()).hexdigest()
    dup = h in seen
    seen.setdefault(h, name)
    rows.append(dict(file=name, path=p, label=cls, readable=ok, duplicate=dup,
                     h=img.shape[0] if ok else 0, w=img.shape[1] if ok else 0,
                     mean=float(img.mean()) if ok else np.nan, std=float(img.std()) if ok else np.nan))
df = pd.DataFrame(rows)
report = dict(total=len(df), unreadable=int((~df.readable).sum()), duplicates=int(df.duplicate.sum()),
              sizes=df.groupby(["h", "w"]).size().rename("n").reset_index().to_dict("records"))
df = df[df.readable & ~df.duplicate].reset_index(drop=True)
report["after_cleaning"] = len(df)
report["per_class"] = df.label.value_counts().reindex(CLASSES).to_dict()
report["intensity"] = df.groupby("label")[["mean", "std"]].mean().round(2).reindex(CLASSES).to_dict("index")

# 2. EDA figures
cnt = df.label.value_counts().reindex(CLASSES)
fig, ax = plt.subplots(figsize=(7, 3.4))
bars = ax.bar(cnt.index, cnt.values, color=["#C0392B"] + ["#7F8C8D"] * 5)
ax.bar_label(bars); ax.set_ylabel("Images"); ax.set_title("Images per class (crazing = crack class)")
plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(f"{FIG}/01_class_distribution.png"); plt.close()

fig, axs = plt.subplots(len(CLASSES), 5, figsize=(8, 10))
rng = np.random.default_rng(0)
for i, c in enumerate(CLASSES):
    picks = rng.choice(df[df.label == c].path.values, 5, replace=False)
    for j, p in enumerate(picks):
        axs[i, j].imshow(cv2.imread(p, 0), cmap="gray"); axs[i, j].axis("off")
    axs[i, 0].set_title(c, loc="left", fontsize=10)
plt.tight_layout(); plt.savefig(f"{FIG}/02_sample_grid.png"); plt.close()

fig, ax = plt.subplots(figsize=(7, 3.6))
for c in CLASSES:
    hist = np.zeros(256)
    for p in df[df.label == c].path.values[:100]:
        hist += cv2.calcHist([cv2.imread(p, 0)], [0], None, [256], [0, 256]).ravel()
    ax.plot(hist / hist.sum(), label=c, lw=2.2 if c == "crazing" else 1.2)
ax.set_xlabel("Pixel intensity"); ax.set_ylabel("Fraction of pixels"); ax.set_title("Pixel intensity distribution per class")
ax.legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{FIG}/03_intensity_histograms.png"); plt.close()

fig, axs = plt.subplots(1, 6, figsize=(11, 2.3))
for a, c in zip(axs, CLASSES):
    m = np.mean([cv2.imread(p, 0).astype(float) for p in df[df.label == c].path], axis=0)
    a.imshow(m, cmap="gray"); a.set_title(c, fontsize=9); a.axis("off")
plt.suptitle("Mean image per class"); plt.tight_layout(); plt.savefig(f"{FIG}/04_mean_images.png"); plt.close()

# 3. pre-processing
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
def preprocess(gray):
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    g = clahe.apply(g)
    return cv2.resize(g, (SIZE, SIZE), interpolation=cv2.INTER_AREA)

ex = df[df.label == "crazing"].path.values[3]
raw = cv2.imread(ex, 0); blur = cv2.GaussianBlur(raw, (3, 3), 0); cl = clahe.apply(blur); fin = cv2.resize(cl, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
fig, axs = plt.subplots(2, 4, figsize=(11, 5.2))
for a, im, t in zip(axs[0], [raw, blur, cl, fin], ["1. Raw (200x200)", "2. Gaussian blur 3x3", "3. CLAHE contrast", f"4. Resized {SIZE}x{SIZE}"]):
    a.imshow(im, cmap="gray", vmin=0, vmax=255); a.set_title(t, fontsize=9); a.axis("off")
for a, im in zip(axs[1], [raw, blur, cl, fin]):
    a.hist(im.ravel(), 64, color="#34495E"); a.set_yticks([])
plt.suptitle("Pre-processing steps on a crazing (crack) image, with pixel histograms"); plt.tight_layout()
plt.savefig(f"{FIG}/05_preprocessing_steps.png"); plt.close()

# 4. stratified split 70/15/15
tr, tmp = train_test_split(df, test_size=0.30, stratify=df.label, random_state=42)
va, te = train_test_split(tmp, test_size=0.50, stratify=tmp.label, random_state=42)
DS = os.path.join(OUT, "NEU_steel_crack_dataset")
if os.path.exists(DS): shutil.rmtree(DS)
split_rows = []
for split, part in [("train", tr), ("val", va), ("test", te)]:
    for _, r in part.iterrows():
        d = os.path.join(DS, "processed", split, r.label); os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, r.file.replace(".jpg", ".png")), preprocess(cv2.imread(r.path, 0)))
        d2 = os.path.join(DS, "original", split, r.label); os.makedirs(d2, exist_ok=True)
        shutil.copy(r.path, os.path.join(d2, r.file))
        split_rows.append(dict(file=r.file, label=r.label, split=split, is_crack=int(r.label == "crazing")))
sp = pd.DataFrame(split_rows); sp.to_csv(os.path.join(DS, "labels.csv"), index=False)
report["split"] = sp.groupby(["split", "label"]).size().unstack().reindex(columns=CLASSES).to_dict("index")

fig, ax = plt.subplots(figsize=(7, 3.4))
tab = sp.groupby(["label", "split"]).size().unstack().reindex(CLASSES)[["train", "val", "test"]]
tab.plot.bar(ax=ax, color=["#2C3E50", "#7F8C8D", "#C0392B"]); ax.set_ylabel("Images"); ax.set_title("Stratified 70/15/15 split")
plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(f"{FIG}/06_split.png"); plt.close()

json.dump(report, open(os.path.join(OUT, "eda_report.json"), "w"), indent=1)
print(json.dumps(report, indent=1))
