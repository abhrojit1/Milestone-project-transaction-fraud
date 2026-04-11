import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix,
)
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = Path("data.csv")
OUTPUT_DIR  = Path("model_output")
EPOCHS      = 5
BATCH_SIZE  = 32
LR          = 2e-5
MAX_LEN     = 96       # longer than before to fit all 7 features
TEST_SIZE   = 0.2
SEED        = 42

# Column names in data.csv
COL_AMOUNT      = "amount"
COL_TYPE        = "transaction_type"
COL_CATEGORY    = "merchant_category"
COL_COUNTRY     = "country"
COL_HOUR        = "hour"
COL_DEVICE_RISK = "device_risk_score"
COL_IP_RISK     = "ip_risk_score"
COL_LABEL       = "is_fraud"
# ─────────────────────────────────────────────────────────────────────────────


def row_to_text(row):
    """
    Serialize one transaction row into a natural-language string for BERT.
    Uses ALL informative columns from this dataset.
    """
    hour = int(row[COL_HOUR])
    if hour < 5:
        time_desc = "late night"
    elif hour < 12:
        time_desc = "morning"
    elif hour < 17:
        time_desc = "afternoon"
    elif hour < 21:
        time_desc = "evening"
    else:
        time_desc = "night"

    device_risk = float(row[COL_DEVICE_RISK])
    ip_risk     = float(row[COL_IP_RISK])

    if device_risk > 0.7:
        dev_desc = "high risk"
    elif device_risk > 0.4:
        dev_desc = "medium risk"
    else:
        dev_desc = "low risk"

    if ip_risk > 0.7:
        ip_desc = "high risk"
    elif ip_risk > 0.4:
        ip_desc = "medium risk"
    else:
        ip_desc = "low risk"

    return (
        f"amount: {round(float(row[COL_AMOUNT]), 2)} | "
        f"type: {str(row[COL_TYPE]).lower()} | "
        f"category: {str(row[COL_CATEGORY]).lower()} | "
        f"country: {str(row[COL_COUNTRY]).lower()} | "
        f"hour: {hour} ({time_desc}) | "
        f"device risk: {dev_desc} ({round(device_risk, 3)}) | "
        f"ip risk: {ip_desc} ({round(ip_risk, 3)})"
    )


class TransactionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    for batch in loader:
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        lbls  = batch["label"].to(device)
        optimizer.zero_grad()
        out   = model(input_ids=ids, attention_mask=mask, labels=lbls)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += out.loss.item()
        all_preds.extend(torch.argmax(out.logits, 1).cpu().numpy())
        all_labels.extend(lbls.cpu().numpy())
    return total_loss / len(loader), accuracy_score(all_labels, all_preds)


def eval_epoch(model, loader, device):
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []
    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            lbls  = batch["label"].to(device)
            out   = model(input_ids=ids, attention_mask=mask, labels=lbls)
            total_loss += out.loss.item()
            probs = torch.softmax(out.logits, 1).cpu().numpy()
            all_preds.extend(np.argmax(probs, 1))
            all_labels.extend(lbls.cpu().numpy())
            all_probs.extend(probs[:, 1])
    return (
        total_loss / len(loader),
        accuracy_score(all_labels, all_preds),
        all_preds, all_labels, all_probs,
    )


def main():
    print("=" * 60)
    print("  FraudSense AI — Training on synthetic_fraud_dataset.csv")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n⚙  Device: {device}")

    # ── Load dataset ──────────────────────────────────────────────
    print(f"\n📂 Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[COL_AMOUNT, COL_TYPE, COL_CATEGORY, COL_COUNTRY,
                           COL_HOUR, COL_DEVICE_RISK, COL_IP_RISK, COL_LABEL])
    print(f"   Rows: {len(df):,}")
    n_fraud = int(df[COL_LABEL].sum())
    print(f"   Legit: {len(df)-n_fraud:,}  |  Fraud: {n_fraud:,}  "
          f"({n_fraud/len(df)*100:.1f}%)")

    texts  = [row_to_text(row) for _, row in df.iterrows()]
    labels = df[COL_LABEL].astype(int).tolist()

    print("\n📝 Sample encodings:")
    for i in range(3):
        print(f"   [{labels[i]}] {texts[i]}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        texts, labels, test_size=TEST_SIZE, stratify=labels, random_state=SEED
    )
    print(f"\n✅ Train: {len(X_tr):,}  |  Val: {len(X_val):,}")

    # ── Model ─────────────────────────────────────────────────────
    print("\n🤖 Loading DistilBERT ...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model     = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    model.to(device)

    train_loader = DataLoader(
        TransactionDataset(X_tr,  y_tr,  tokenizer),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        TransactionDataset(X_val, y_val, tokenizer),
        batch_size=BATCH_SIZE * 2,
    )

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    # ── Train ─────────────────────────────────────────────────────
    print(f"\n🚀 Training for {EPOCHS} epochs ...\n")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        t_loss, t_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        v_loss, v_acc, val_preds, val_labels, _ = eval_epoch(model, val_loader, device)

        f1   = f1_score(val_labels, val_preds, average="binary", zero_division=0)
        prec = precision_score(val_labels, val_preds, average="binary", zero_division=0)
        rec  = recall_score(val_labels, val_preds, average="binary", zero_division=0)

        history.append(dict(
            epoch=epoch,
            train_loss=round(t_loss, 4), train_acc=round(t_acc, 4),
            val_loss=round(v_loss, 4),   val_acc=round(v_acc, 4),
            f1=round(f1, 4), precision=round(prec, 4), recall=round(rec, 4),
        ))
        print(f"  Epoch {epoch:02d}/{EPOCHS}  "
              f"TrLoss:{t_loss:.4f} TrAcc:{t_acc*100:.2f}%  |  "
              f"ValLoss:{v_loss:.4f} ValAcc:{v_acc*100:.2f}%  F1:{f1:.4f}")

        # ── Save per-epoch checkpoint ──
        ckpt_path = OUTPUT_DIR / f"checkpoint_epoch_{epoch:02d}.pt"
        torch.save({
            "epoch": epoch, "val_acc": v_acc,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, ckpt_path)
        print(f"    💾 {ckpt_path.name}")

        # ── Save best model ──
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            model.save_pretrained(OUTPUT_DIR / "best_model")
            tokenizer.save_pretrained(OUTPUT_DIR / "best_model")
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model_weights.pt")
            print(f"    ✅ New best! Saved to model_output/best_model/")

    # ── Final model ───────────────────────────────────────────────
    model.save_pretrained(OUTPUT_DIR / "final_model")
    tokenizer.save_pretrained(OUTPUT_DIR / "final_model")
    torch.save(model.state_dict(), OUTPUT_DIR / "final_model_weights.pt")

    # ── Final evaluation ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Final Evaluation")
    print("=" * 60)
    _, _, final_preds, final_labels, _ = eval_epoch(model, val_loader, device)
    print(classification_report(final_labels, final_preds, target_names=["Legit", "Fraud"]))
    cm = confusion_matrix(final_labels, final_preds)
    print(f"Confusion Matrix:\n  TN={cm[0][0]}  FP={cm[0][1]}\n  FN={cm[1][0]}  TP={cm[1][1]}")

    # ── Save metadata (read by app.py) ────────────────────────────
    meta = {
        "trained_at":    datetime.now().isoformat(),
        "base_model":    "distilbert-base-uncased",
        "dataset":       "synthetic_fraud_dataset.csv",
        "rows":          len(df),
        "fraud_count":   n_fraud,
        "legit_count":   len(df) - n_fraud,
        "epochs":        EPOCHS,
        "batch_size":    BATCH_SIZE,
        "lr":            LR,
        "max_len":       MAX_LEN,
        "best_val_acc":  round(best_val_acc, 4),
        "columns": {
            "amount":       COL_AMOUNT,
            "type":         COL_TYPE,
            "category":     COL_CATEGORY,
            "country":      COL_COUNTRY,
            "hour":         COL_HOUR,
            "device_risk":  COL_DEVICE_RISK,
            "ip_risk":      COL_IP_RISK,
            "label":        COL_LABEL,
        },
        "transaction_types": ["ATM", "QR", "Online", "POS"],
        "merchant_categories": ["Travel", "Food", "Clothing", "Grocery", "Electronics"],
        "countries": ["TR", "US", "FR", "DE", "UK", "NG"],
        "confusion_matrix": {
            "TN": int(cm[0][0]), "FP": int(cm[0][1]),
            "FN": int(cm[1][0]), "TP": int(cm[1][1]),
        },
        "history": history,
    }
    with open(OUTPUT_DIR / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n💾 Saved to {OUTPUT_DIR.resolve()}/")
    print(f"   best_model/            ← used by app.py")
    print(f"   final_model/")
    print(f"   best_model_weights.pt")
    print(f"   final_model_weights.pt")
    print(f"   checkpoint_epoch_XX.pt")
    print(f"   training_meta.json     ← loaded by app.py on startup")
    print(f"\n🎉 Done!  Best val accuracy: {best_val_acc*100:.2f}%")
    print(f"\n▶  Now run:  python app.py")


if __name__ == "__main__":
    main()
