import argparse, yaml, torch, torch.nn.functional as F, os, json
from torchmetrics.classification import BinaryAUROC
from models import create_model
from data import build_loaders

def train(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(cfg["ckpt"]), exist_ok=True)
    os.makedirs(cfg["run_dir"], exist_ok=True)

    tr, va, _ = build_loaders(cfg["data_root"], cfg["img_size"], cfg["batch_size"], cfg["augment"])
    model = create_model("efficientnet_b0").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    auroc = BinaryAUROC().to(device)

    best, best_state, bad, patience = 0.0, None, 0, cfg["patience"]
    for ep in range(cfg["epochs"]):
        model.train()
        for x,y in tr:
            x,y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()

        model.eval(); auroc.reset()
        with torch.no_grad():
            for x,y in va:
                x,y = x.to(device), y.to(device)
                p1 = model(x).softmax(1)[:,1]; auroc.update(p1, y)
        vauc = float(auroc.compute().item())
        print(f"epoch {ep}: val ROC-AUC={vauc:.4f}")
        if vauc > best: best, best_state, bad = vauc, model.state_dict(), 0
        else:
            bad += 1
            if bad > patience: break

    model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "best_val_auc": best}, cfg["ckpt"])
    with open(os.path.join(cfg["run_dir"], "train_summary.json"), "w") as f: json.dump({"best_val_auc": best}, f, indent=2)
    print("best val ROC-AUC:", best)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cfg", required=True)
    args = ap.parse_args(); train(yaml.safe_load(open(args.cfg)))
