import argparse, torch, numpy as np, json
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support, accuracy_score
from data import build_loaders
from models import create_model

def test(ckpt, data_root, img_size, bs, run_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, te = build_loaders(data_root, img_size, bs, augment=False)
    m = create_model("efficientnet_b0", pretrained=False).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"]); m.eval()

    probs, labels = [], []
    with torch.no_grad():
        for x,y in te:
            x = x.to(device); p = m(x).softmax(1)[:,1].cpu().numpy()
            probs.append(p); labels.append(y.numpy())
    probs = np.concatenate(probs); labels = np.concatenate(labels)
    preds = (probs>=0.5).astype(int)

    auc = roc_auc_score(labels, probs); acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    spec = tn / (tn + fp + 1e-12)
    metrics = dict(roc_auc=float(auc), accuracy=float(acc), precision=float(prec),
                   recall_sensitivity=float(rec), specificity=float(spec), f1=float(f1),
                   tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    os.makedirs(run_dir, exist_ok=True)
    with open(f"{run_dir}/metrics_test.json","w") as f: json.dump(metrics, f, indent=2)
    print(metrics)

if __name__ == "__main__":
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", default="../chest_xray")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--run-dir", default="runs/effb0_224")
    a = ap.parse_args(); test(a.ckpt, a.data_root, a.img_size, a.batch_size, a.run_dir)
