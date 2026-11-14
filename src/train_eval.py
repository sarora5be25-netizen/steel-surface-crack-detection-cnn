import os, sys, json, time, glob
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, confusion_matrix,
                             roc_auc_score, classification_report, ConfusionMatrixDisplay)

torch.manual_seed(42); np.random.seed(42); torch.set_num_threads(2)
DS = sys.argv[1]; OUT = sys.argv[2]; EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 30
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

def load(split):
    X, y = [], []
    for i, c in enumerate(CLASSES):
        for p in sorted(glob.glob(f"{DS}/processed/{split}/{c}/*.png")):
            X.append(cv2.imread(p, 0)); y.append(i)
    X = torch.tensor(np.stack(X), dtype=torch.float32).unsqueeze(1) / 255.0
    return X, torch.tensor(y)
Xtr, ytr = load("train"); Xva, yva = load("val"); Xte, yte = load("test")
MEAN, STD = Xtr.mean().item(), Xtr.std().item()
norm = lambda x: (x - MEAN) / STD
print("train/val/test", len(ytr), len(yva), len(yte), "mean/std", round(MEAN, 4), round(STD, 4))

def augment(x):
    # random flips, 90-degree rotations, brightness/contrast jitter (on real images only)
    if np.random.rand() < 0.5: x = x.flip(3)
    if np.random.rand() < 0.5: x = x.flip(2)
    k = np.random.randint(4); x = torch.rot90(x, k, (2, 3))
    b = (torch.rand(x.size(0), 1, 1, 1) - 0.5) * 0.2
    c = 1 + (torch.rand(x.size(0), 1, 1, 1) - 0.5) * 0.3
    return ((x - 0.5) * c + 0.5 + b).clamp(0, 1)

class SteelCNN(nn.Module):
    def __init__(s, n=6):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        s.features = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128), blk(128, 256))
        s.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.4), nn.Linear(256, n))
    def forward(s, x): return s.head(s.features(x))

def predict(model, X):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 128): out.append(F.softmax(model(norm(X[i:i+128])), 1))
    return torch.cat(out).numpy()

def run(name, use_aug):
    torch.manual_seed(42); np.random.seed(42)
    model = SteelCNN(); opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    hist = dict(train_loss=[], val_loss=[], train_acc=[], val_acc=[]); best, best_state = -1, None
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(len(ytr)); tl, tc = 0, 0
        for i in range(0, len(perm), 32):
            idx = perm[i:i+32]; xb = Xtr[idx]; yb = ytr[idx]
            if use_aug: xb = augment(xb)
            out = model(norm(xb)); loss = F.cross_entropy(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(idx); tc += (out.argmax(1) == yb).sum().item()
        sched.step()
        pv = predict(model, Xva); vl = F.nll_loss(torch.log(torch.tensor(pv) + 1e-9), yva).item(); va = (pv.argmax(1) == yva.numpy()).mean()
        hist["train_loss"].append(tl / len(ytr)); hist["train_acc"].append(tc / len(ytr)); hist["val_loss"].append(vl); hist["val_acc"].append(float(va))
        if va > best: best, best_state, best_ep = va, {k: v.clone() for k, v in model.state_dict().items()}, ep + 1
        print(f"{name} ep{ep+1:02d} loss {tl/len(ytr):.3f} acc {tc/len(ytr):.3f} | val loss {vl:.3f} acc {va:.3f}", flush=True)
    model.load_state_dict(best_state); torch.save(best_state, f"{OUT}/{name}.pth")
    return model, hist, dict(best_val_acc=float(best), best_epoch=best_ep, train_time_s=round(time.time() - t0, 1),
                             params=sum(p.numel() for p in model.parameters()))

def evaluate(model, name):
    P = predict(model, Xte); yp = P.argmax(1); yt = yte.numpy()
    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=range(6), zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
    # crack (crazing) vs everything else, as a plant alarm would use it
    bt, bp = (yt == 0).astype(int), (yp == 0).astype(int)
    bpr, brc, bf, _ = precision_recall_fscore_support(bt, bp, average="binary", zero_division=0)
    m = dict(accuracy=accuracy_score(yt, yp), macro_precision=mp, macro_recall=mr, macro_f1=mf,
             roc_auc_ovr_macro=roc_auc_score(yt, P, multi_class="ovr", average="macro"),
             crack_precision=bpr, crack_recall=brc, crack_f1=bf, crack_auc=roc_auc_score(bt, P[:, 0]),
             per_class={c: dict(precision=float(p[i]), recall=float(r[i]), f1=float(f[i]), support=int(s[i])) for i, c in enumerate(CLASSES)},
             confusion=confusion_matrix(yt, yp).tolist())
    m = {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in m.items()}
    print(classification_report(yt, yp, target_names=CLASSES, digits=4))
    return m, P

results, hists, models = {}, {}, {}
for name, aug in [("cnn_baseline_no_aug", False), ("cnn_augmented", True)]:
    model, h, info = run(name, aug)
    m, P = evaluate(model, name)
    results[name] = dict(**info, test=m); hists[name] = h; models[name] = (model, P)

json.dump(dict(results=results, history=hists, norm=dict(mean=MEAN, std=STD)), open(f"{OUT}/results.json", "w"), indent=1)

# figures: learning curves
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
for name, ls in [("cnn_baseline_no_aug", "--"), ("cnn_augmented", "-")]:
    h = hists[name]; e = range(1, EPOCHS + 1); lab = "baseline" if "base" in name else "augmented"
    axs[0].plot(e, h["train_loss"], ls, color="#2C3E50", label=f"{lab} train"); axs[0].plot(e, h["val_loss"], ls, color="#C0392B", label=f"{lab} val")
    axs[1].plot(e, h["train_acc"], ls, color="#2C3E50", label=f"{lab} train"); axs[1].plot(e, h["val_acc"], ls, color="#C0392B", label=f"{lab} val")
axs[0].set_title("Loss"); axs[1].set_title("Accuracy"); axs[0].set_ylim(0, 2)
for a in axs: a.set_xlabel("Epoch"); a.legend(fontsize=8); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{FIG}/07_learning_curves.png", dpi=130); plt.close()

best = max(results, key=lambda k: results[k]["test"]["macro_f1"])
model, P = models[best]; yp = P.argmax(1)
fig, ax = plt.subplots(figsize=(6.4, 5.4))
ConfusionMatrixDisplay(confusion_matrix(yte.numpy(), yp), display_labels=CLASSES).plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=35)
ax.set_title(f"Confusion matrix on test set ({best})"); plt.tight_layout(); plt.savefig(f"{FIG}/08_confusion_matrix.png", dpi=130); plt.close()

from sklearn.metrics import roc_curve
fig, ax = plt.subplots(figsize=(5.6, 4.6))
for i, c in enumerate(CLASSES):
    fpr, tpr, _ = roc_curve((yte.numpy() == i).astype(int), P[:, i]); ax.plot(fpr, tpr, lw=2.2 if i == 0 else 1.2, label=c)
ax.plot([0, 1], [0, 1], "k:", lw=.8); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title("ROC curves (one vs rest)"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{FIG}/09_roc_curves.png", dpi=130); plt.close()

# Grad-CAM on test crazing images
def gradcam(model, x, cls):
    model.eval(); acts = {}
    hk = model.features[-1].register_forward_hook(lambda m, i, o: acts.setdefault("a", o))
    x = norm(x).requires_grad_(False); out = model(x); hk.remove()
    a = acts["a"]; g = torch.autograd.grad(out[0, cls], a)[0]
    cam = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1))[0].detach().numpy()
    cam = cv2.resize(cam, (x.shape[-1], x.shape[-2])); return cam / (cam.max() + 1e-8)
idx = [i for i in range(len(yte)) if yte[i] == 0][:4] + [i for i in range(len(yte)) if yte[i] == 5][:2]
fig, axs = plt.subplots(2, len(idx), figsize=(2.1 * len(idx), 4.4))
for j, i in enumerate(idx):
    img = Xte[i, 0].numpy(); cam = gradcam(model, Xte[i:i+1], int(yp[i]))
    axs[0, j].imshow(img, cmap="gray"); axs[0, j].set_title(f"true: {CLASSES[yte[i]]}", fontsize=8)
    axs[1, j].imshow(img, cmap="gray"); axs[1, j].imshow(cam, cmap="jet", alpha=.45); axs[1, j].set_title(f"pred: {CLASSES[yp[i]]}", fontsize=8)
    axs[0, j].axis("off"); axs[1, j].axis("off")
plt.suptitle("Grad-CAM: where the CNN looks"); plt.tight_layout(); plt.savefig(f"{FIG}/10_gradcam.png", dpi=130); plt.close()

wrong = np.where(yp != yte.numpy())[0][:8]
if len(wrong):
    fig, axs = plt.subplots(1, len(wrong), figsize=(2 * len(wrong), 2.4), squeeze=False)
    for a, i in zip(axs[0], wrong):
        a.imshow(Xte[i, 0], cmap="gray"); a.axis("off"); a.set_title(f"T:{CLASSES[yte[i]]}\nP:{CLASSES[yp[i]]}", fontsize=7)
    plt.suptitle("Misclassified test images"); plt.tight_layout(); plt.savefig(f"{FIG}/11_misclassified.png", dpi=130); plt.close()
print(json.dumps({k: {kk: v[kk] for kk in ["best_val_acc", "best_epoch", "train_time_s", "params"]} | {kk: v["test"][kk] for kk in ["accuracy", "macro_f1", "crack_recall", "crack_precision", "roc_auc_ovr_macro"]} for k, v in results.items()}, indent=1))
print("best", best, "wrong", len(np.where(yp != yte.numpy())[0]))
