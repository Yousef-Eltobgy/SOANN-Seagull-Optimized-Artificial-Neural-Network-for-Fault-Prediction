# soann_paper_replica_with_sim.py
import os, glob, random, math, json # os, glob → file handling (find all CSVs in a folder). 
#random, math → randomness, math functions (used in optimization & simulation).
#json → saving experiment results to JSON.
import numpy as np #numpy (np) → fast arrays & math.
import pandas as pd #pandas (pd) → read CSV chunks, data processing.
from tqdm import tqdm #tqdm → progress bars when fitting scaler.

import torch #torch → PyTorch framework for neural networks.
import torch.nn as nn #nn → defines neural network layers (Linear, ReLU, Dropout).
import torch.optim as optim #optim → optimizers like Adam for training.

from sklearn.preprocessing import StandardScaler #StandardScaler → normalize SMART features across files.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve #metrics → compute ML performance (accuracy, precision, recall, F1, ROC AUC).

# --------------------------
# CONFIG
# --------------------------
DATA_DIR = r"Data"                # change to your dataset folder
BATCH_SIZE = 5000
EPOCHS_BASELINE = 8
LR_BASELINE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SOA params
POP_SIZE = 12
SOA_ITERS = 20
ALPHA_MIGRATE = 1.8
BETA_EXPL = 0.6
SIGMA_SPIRAL = 0.08
FINE_TUNE_LAST_EPOCHS = 6
SEED = 42

# Validation / sampling
VAL_ROWS_LIMIT = 15000
VAL_POS_MIN = 150

# Attribute selection (paper critical attributes)
USE_ALL_ATTRIBUTES = False
CRITICAL_SMART_ATTRIBUTES = [
    'smart_5_raw', 'smart_187_raw', 'smart_188_raw', 'smart_197_raw',
    'smart_198_raw', 'smart_199_raw', 'smart_241_raw', 'smart_242_raw'
]
    # smart_5_raw: Tracks how many bad sectors have been replaced with spare ones.
    # smart_187_raw: Counts read/write errors that couldn’t be recovered by ECC (error-correction).
    # smart_188_raw: Number of operations that exceeded the time limit.
    # smart_197_raw: Sectors waiting to be reallocated because they couldn’t be read successfully.
    # smart_198_raw: Number of sectors that failed during offline scanning.
    # smart_199_raw: Data transfer errors between drive and controller (usually cabling/power issues).
    # smart_241_raw / smart_242_raw: Tracks total data written/read in lifetime (in logical block addresses).

SECONDARY_SMART_ATTRIBUTES = [
    'smart_1_raw', 'smart_7_raw', 'smart_9_raw', 'smart_12_raw',
    'smart_194_raw', 'smart_195_raw'
]
    # smart_1_raw: Errors in reading from the disk surface.
    # smart_7_raw: Failures in positioning the read/write heads.
    # smart_9_raw: Total hours powered on.
    # smart_12_raw: Times the drive was powered on/off.
    # smart_194_raw: Operating temperature.
    # smart_195_raw: Number of errors corrected by error-correction logic.

SOA_FITNESS = "hybrid"  # "f1", "roc_auc", or "hybrid"

# Simulation defaults (paper-like scale)
SIM_REQ_RATE = 196
SIM_MINUTES = 60
SIM_DEADLINE = 3000.0
SIM_RECOVERY_PENALTY = 600.0
SIM_FALSE_ALARM_PENALTY = 45.0

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --------------------------
# Utilities
# --------------------------
def safe_roc_auc(y_true, y_score):
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return roc_auc_score(y_true, y_score)
    except Exception:
        return float("nan")

def evaluate_fitness(metrics):
    f1 = metrics.get("f1", 0.0) or 0.0
    roc = metrics.get("roc_auc", 0.0) or 0.0
    if SOA_FITNESS == "f1":
        return f1
    elif SOA_FITNESS == "roc_auc":
        return roc
    else:
        return 0.5 * f1 + 0.5 * (roc if not math.isnan(roc) else 0.0)

def get_useful_columns(sample_path):
    df_sample = pd.read_csv(sample_path, nrows=2)
    available_cols = df_sample.columns.tolist()
    if USE_ALL_ATTRIBUTES:
        smart_cols = [c for c in available_cols if c.startswith("smart_")]
    else:
        selected = [c for c in CRITICAL_SMART_ATTRIBUTES + SECONDARY_SMART_ATTRIBUTES if c in available_cols]
        smart_cols = selected
    useful_cols = ["failure"] + smart_cols
    print(f"Selected {len(smart_cols)} SMART attributes (USE_ALL_ATTRIBUTES={USE_ALL_ATTRIBUTES})")
    return useful_cols

# --------------------------
# Streaming scaler & data loaders
# --------------------------
def fit_scaler(files, useful_cols, batch_size=BATCH_SIZE):
    scaler = StandardScaler()
    smart_cols = [c for c in useful_cols if c.startswith("smart_")]
    n = 0
    for f in tqdm(files, desc="Fitting scaler"):
        for chunk in pd.read_csv(f, chunksize=batch_size, usecols=useful_cols):
            chunk = chunk.dropna(subset=["failure"]).fillna(0)
            if chunk.empty: 
                continue
            X_chunk = chunk[smart_cols].astype("float32").values
            if X_chunk.size == 0: 
                continue
            scaler.partial_fit(X_chunk)
            n += 1
    if n == 0:
        raise RuntimeError("Scaler could not be fitted — check files/columns.")
    print(f"Scaler fitted on {n} batches.")
    return scaler

def data_stream(files, useful_cols, scaler, batch_size=BATCH_SIZE):
    smart_cols = [c for c in useful_cols if c.startswith("smart_")]
    for f in files:
        for chunk in pd.read_csv(f, chunksize=batch_size, usecols=useful_cols):
            chunk = chunk.dropna(subset=["failure"]).fillna(0)
            if chunk.empty:
                continue
            X = chunk[smart_cols].astype("float32").values
            y = chunk["failure"].astype("int64").values
            if X.size == 0:
                continue
            X = scaler.transform(X)
            yield X, y

def build_validation_sample(files, useful_cols, scaler, limit_rows=VAL_ROWS_LIMIT, min_pos=VAL_POS_MIN):
    smart_cols = [c for c in useful_cols if c.startswith("smart_")]
    Xs, ys = [], []
    pos_collected = 0
    total_collected = 0
    for f in files:
        for chunk in pd.read_csv(f, chunksize=10000, usecols=useful_cols):
            chunk = chunk.dropna(subset=["failure"]).fillna(0)
            if chunk.empty:
                continue
            y_chunk = chunk["failure"].astype(int).values
            X_chunk = chunk[smart_cols].astype("float32").values
            if X_chunk.size == 0:
                continue
            # keep all positives
            mask_pos = (y_chunk == 1)
            if mask_pos.sum() > 0:
                Xs.append(X_chunk[mask_pos])
                ys.append(y_chunk[mask_pos])
                pos_collected += mask_pos.sum()
                total_collected += mask_pos.sum()
            # sample some negatives
            mask_neg = (y_chunk == 0)
            if mask_neg.sum() > 0:
                negs = X_chunk[mask_neg]
                negy = y_chunk[mask_neg]
                sample_size = max(1, int(0.02 * len(negs)))  # 2% negatives each chunk
                idx = np.random.choice(len(negs), size=min(sample_size, len(negs)), replace=False)
                Xs.append(negs[idx])
                ys.append(negy[idx])
                total_collected += len(idx)
            if total_collected >= limit_rows and pos_collected >= min_pos:
                X = np.vstack(Xs)[:limit_rows]
                y = np.concatenate(ys)[:limit_rows]
                X = scaler.transform(X)
                print(f"Validation sample built: {len(y)} rows, positives={int(np.sum(y))}")
                return X, y
    if len(Xs) == 0:
        raise RuntimeError("Could not build validation sample (no rows).")
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    X = scaler.transform(X)
    print(f"Validation sample built (fallback): {len(y)} rows, positives={int(np.sum(y))}")
    return X, y

# --------------------------
# ANN model
# --------------------------
class ANN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.dropout1(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu(self.bn2(self.fc2(x))))
        x = self.relu(self.bn3(self.fc3(x)))
        logits = self.fc4(x).squeeze(-1)
        return logits

# --------------------------
# Training + Evaluation
# --------------------------
def train_baseline(model, files, useful_cols, scaler, epochs=EPOCHS_BASELINE, lr=LR_BASELINE):
    model.to(DEVICE)
    # approximate pos/neg ratio from a small sample
    pos = 1; neg = 1; count = 0
    for Xs, ys in data_stream(files[:2], useful_cols, scaler, batch_size=BATCH_SIZE):
        pos += (ys == 1).sum()
        neg += (ys == 0).sum()
        count += len(ys)
        if count >= 10000:
            break
    pos_weight = torch.tensor([float(neg) / float(pos)], dtype=torch.float32).to(DEVICE)
    print(f"Estimated pos_weight = {pos_weight.item():.3f} (neg={neg}, pos={pos})")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.train()
    for epoch in range(1, epochs+1):
        losses = []
        for X_batch, y_batch in data_stream(files, useful_cols, scaler, batch_size=BATCH_SIZE):
            Xb = torch.tensor(X_batch, dtype=torch.float32).to(DEVICE)
            yb = torch.tensor(y_batch, dtype=torch.float32).to(DEVICE)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch}/{epochs} loss: {np.nanmean(losses):.5f}")
    return model, pos_weight

def evaluate_model_and_preds(model, X_sample, y_sample):
    model.to(DEVICE).eval()
    with torch.no_grad():
        X_t = torch.tensor(X_sample, dtype=torch.float32).to(DEVICE)
        logits = model(X_t).cpu().numpy().flatten()
        probs = 1.0 / (1.0 + np.exp(-logits))  # stable sigmoid on numpy
    # safe ROC AUC
    roc = safe_roc_auc(y_sample, probs)
    # pick threshold maximizing F1 on precision-recall curve
    prec, rec, th = precision_recall_curve(y_sample, probs)
    if len(prec) == 0 or len(rec) == 0:
        best_thresh = 0.5
    else:
        f1s = 2 * (prec * rec) / (prec + rec + 1e-12)
        best_idx = np.nanargmax(f1s)
        best_thresh = float(th[best_idx]) if best_idx < len(th) else 0.5
    preds = (probs >= best_thresh).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_sample, preds)),
        "precision": float(precision_score(y_sample, preds, zero_division=0)),
        "recall": float(recall_score(y_sample, preds, zero_division=0)),
        "f1": float(f1_score(y_sample, preds, zero_division=0)),
        "roc_auc": float(roc),
        "best_thresh": best_thresh
    }
    return metrics, probs, preds

# --------------------------
# SOA helpers (last layer)
# --------------------------
def get_last_layer_vector(model):
    w = model.fc4.weight.detach().cpu().numpy().ravel()
    b = model.fc4.bias.detach().cpu().numpy().ravel()
    return np.concatenate([w, b])

def set_last_layer_from_vector(model, vec):
    vec = np.asarray(vec)
    w_size = model.fc4.weight.numel()
    w = vec[:w_size].reshape(model.fc4.weight.shape)
    b = vec[w_size:w_size + model.fc4.bias.numel()]
    with torch.no_grad():
        model.fc4.weight.copy_(torch.tensor(w, dtype=torch.float32))
        model.fc4.bias.copy_(torch.tensor(b, dtype=torch.float32))

# --------------------------
# SOA optimize (seagull-like updates on last layer)
# --------------------------
def soa_optimize_last_layer(model, X_val, y_val, pop_size=POP_SIZE, iters=SOA_ITERS):
    dim = get_last_layer_vector(model).shape[0]
    base = get_last_layer_vector(model)
    pop = np.array([base + 0.1 * np.random.randn(dim) for _ in range(pop_size)])
    best_score = -np.inf
    best_vec = base.copy()
    # initial evaluate
    for i in range(pop_size):
        set_last_layer_from_vector(model, pop[i])
        metrics, _, _ = evaluate_model_and_preds(model, X_val, y_val)
        score = evaluate_fitness(metrics)
        if score > best_score:
            best_score = score
            best_vec = pop[i].copy()
    print(f"SOA init best score ({SOA_FITNESS}) = {best_score:.5f}")
    # main loop
    for t in range(1, iters + 1):
        a = 2.0 * (1 - t/iters)
        for i in range(pop_size):
            r1 = np.random.rand(dim); r2 = np.random.rand(dim)
            migration = pop[i] + ALPHA_MIGRATE * r1 * (best_vec - np.abs(pop[i]))
            exploration = migration + BETA_EXPL * a * np.random.randn(dim)
            dist = np.linalg.norm(best_vec - pop[i]) + 1e-12
            spiral = best_vec + SIGMA_SPIRAL * dist * np.concatenate([np.cos(2*np.pi*r2), np.sin(2*np.pi*r2)])[:dim]
            if np.random.rand() < (1 - t/iters):
                new_pos = exploration
            else:
                new_pos = spiral
            new_pos += 0.01 * np.random.randn(dim)
            # evaluate
            set_last_layer_from_vector(model, new_pos)
            metrics, _, _ = evaluate_model_and_preds(model, X_val, y_val)
            score = evaluate_fitness(metrics)
            if score > best_score:
                best_score = score
                best_vec = new_pos.copy()
                pop[i] = new_pos.copy()
            elif np.random.rand() < 0.05:
                pop[i] = new_pos.copy()
        if t % max(1, iters // 5) == 0 or t == 1 or t == iters:
            print(f"Iter {t}/{iters} - best score ({SOA_FITNESS}) = {best_score:.5f}")
    # load best
    set_last_layer_from_vector(model, best_vec)
    return model

# --------------------------
# Optional fine-tune (small GD on last layer)
# --------------------------
def fine_tune_last_layer(model, files, useful_cols, scaler, pos_weight, epochs=FINE_TUNE_LAST_EPOCHS, lr=1e-4):
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("fc4.")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    model.train()
    for epoch in range(1, epochs+1):
        losses = []
        for X_batch, y_batch in data_stream(files, useful_cols, scaler, batch_size=BATCH_SIZE):
            Xb = torch.tensor(X_batch, dtype=torch.float32).to(DEVICE)
            yb = torch.tensor(y_batch, dtype=torch.float32).to(DEVICE)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward(); optimizer.step()
            losses.append(loss.item())
        print(f"Fine-tune epoch {epoch}/{epochs} loss: {np.nanmean(losses):.5f}")
    return model

# --------------------------
# Simulation (paper-scale)
# --------------------------
def simulate_cloud_service(y_true, y_pred, req_rate=SIM_REQ_RATE, sim_minutes=SIM_MINUTES,
                           deadline=SIM_DEADLINE, recovery_penalty=SIM_RECOVERY_PENALTY, false_alarm_penalty=SIM_FALSE_ALARM_PENALTY):
    total_requests = req_rate * sim_minutes
    ast_times = []
    success_count = 0
    total_downtime = 0.0
    operational_time = sim_minutes * 60.0
    for i in range(total_requests):
        idx = random.randint(0, len(y_true) - 1)
        true = int(y_true[idx]); pred = int(y_pred[idx])
        service_time = random.uniform(1.0, 3.0)
        if true == 1:
            if pred == 1:
                service_time += random.uniform(60.0, 300.0)
            else:
                downtime = random.uniform(recovery_penalty - 300.0, recovery_penalty + 300.0)
                service_time += downtime
                total_downtime += downtime
        else:
            if pred == 1:
                service_time += random.uniform(false_alarm_penalty - 15.0, false_alarm_penalty + 15.0)
        if service_time <= deadline:
            success_count += 1
            ast_times.append(service_time)
        else:
            total_downtime += max(0.0, (service_time - deadline))
    AST = np.mean(ast_times) if ast_times else float("inf")
    throughput = success_count / sim_minutes
    success_rate = (success_count / total_requests) * 100.0
    availability = 100.0 * max(0.0, (1.0 - total_downtime / operational_time))
    return {"AST": round(float(AST),2), "throughput_rpm": round(float(throughput),1), "success_rate_pct": round(float(success_rate),2), "availability_pct": round(float(availability),2)}

# --------------------------
# Comparison printing helper
# --------------------------
def print_comparison(baseline_metrics, baseline_sim, soann_metrics, soann_sim):
    print("\n" + "="*80)
    print("COMPARISON: BASELINE ANN vs SOANN (paper replica)")
    print("="*80)
    print("\nML METRICS:")
    for k in ["accuracy","precision","recall","f1","roc_auc"]:
        bv = baseline_metrics.get(k, float("nan"))
        sv = soann_metrics.get(k, float("nan"))
        print(f"{k:10s} | ANN = {bv:7.4f} | SOANN = {sv:7.4f} | Δ = {sv-bv:+7.4f}")
    print("\nSERVICE METRICS:")
    for k in ["AST","throughput_rpm","success_rate_pct","availability_pct"]:
        bv = baseline_sim.get(k, float("nan"))
        sv = soann_sim.get(k, float("nan"))
        if k == "AST":
            print(f"{k:14s} | ANN = {bv:7.2f} | SOANN = {sv:7.2f} | Δ = {(bv-sv):+7.2f} (lower better)")
        else:
            print(f"{k:14s} | ANN = {bv:7.2f} | SOANN = {sv:7.2f} | Δ = {(sv-bv):+7.2f}")

# --------------------------
# MAIN
# --------------------------
if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if len(files) == 0:
        raise RuntimeError(f"No CSVs found in {DATA_DIR}")
    print(f"Found {len(files)} data files")
    useful_cols = get_useful_columns(files[0])
    scaler = fit_scaler(files, useful_cols)
    X_val, y_val = build_validation_sample(files, useful_cols, scaler)

    # Baseline ANN
    print("\n=== Training Baseline ANN ===")
    baseline = ANN(X_val.shape[1])
    baseline, pos_weight = train_baseline(baseline, files, useful_cols, scaler)
    baseline_metrics, baseline_probs, baseline_preds = evaluate_model_and_preds(baseline, X_val, y_val)
    baseline_sim = simulate_cloud_service(y_val, baseline_preds)

    # SOANN (ANN + SOA)
    print("\n=== Training SOANN (ANN + SOA) ===")
    soann = ANN(X_val.shape[1])
    soann, _ = train_baseline(soann, files, useful_cols, scaler)
    soann = soa_optimize_last_layer(soann, X_val, y_val, pop_size=POP_SIZE, iters=SOA_ITERS)
    # optional small fine-tune (kept but can be disabled)
    soann = fine_tune_last_layer(soann, files, useful_cols, scaler, pos_weight, epochs=FINE_TUNE_LAST_EPOCHS, lr=1e-4)
    soann_metrics, soann_probs, soann_preds = evaluate_model_and_preds(soann, X_val, y_val)
    soann_sim = simulate_cloud_service(y_val, soann_preds)

    # Print comparison
    print_comparison(baseline_metrics, baseline_sim, soann_metrics, soann_sim)

    # Save results
    out = {
        "config": {
            "pop_size": POP_SIZE, "soa_iters": SOA_ITERS, "soa_fitness": SOA_FITNESS,
            "use_all_attributes": USE_ALL_ATTRIBUTES
        },
        "baseline_metrics": baseline_metrics, "soann_metrics": soann_metrics,
        "baseline_sim": baseline_sim, "soann_sim": soann_sim
    }
    with open("soann_paper_replica_results.json", "w") as f:
        json.dump(out, f, indent=2)

    # Save model weights
    torch.save(baseline.state_dict(), "baseline_model.pth")
    torch.save(soann.state_dict(), "soann_model.pth")

    print("\nSaved results to 'soann_paper_replica_results.json' and model weights.")



"""
================================================================================
COMPARISON: BASELINE ANN vs SOANN (paper replica)
================================================================================

ML METRICS:
accuracy   | ANN =  0.9977 | SOANN =  0.9980 | Δ = +0.0003
precision  | ANN =  0.4483 | SOANN =  0.5417 | Δ = +0.0934
recall     | ANN =  0.4062 | SOANN =  0.4062 | Δ = +0.0000
f1         | ANN =  0.4262 | SOANN =  0.4643 | Δ = +0.0381
roc_auc    | ANN =  0.9723 | SOANN =  0.9751 | Δ = +0.0028

SERVICE METRICS:
AST            | ANN =    2.94 | SOANN =    2.62 | Δ =   +0.32 (lower better)
throughput_rpm | ANN =  196.00 | SOANN =  196.00 | Δ =   +0.00
success_rate_pct | ANN =  100.00 | SOANN =  100.00 | Δ =   +0.00
availability_pct | ANN =    0.00 | SOANN =    0.00 | Δ =   +0.00

++++++++++++++++++++++++++++++++++++++++++++++++++++++
full attributes:

================================================================================
COMPARISON: BASELINE ANN vs SOANN (paper replica)
================================================================================

ML METRICS:
accuracy   | ANN =  0.9971 | SOANN =  0.9982 | Δ = +0.0011
precision  | ANN =  0.3659 | SOANN =  0.6190 | Δ = +0.2532
recall     | ANN =  0.4688 | SOANN =  0.4062 | Δ = -0.0625
f1         | ANN =  0.4110 | SOANN =  0.4906 | Δ = +0.0796
roc_auc    | ANN =  0.9785 | SOANN =  0.9799 | Δ = +0.0014

SERVICE METRICS:
AST            | ANN =    2.95 | SOANN =    2.59 | Δ =   +0.36 (lower better)
throughput_rpm | ANN =  196.00 | SOANN =  196.00 | Δ =   +0.00
success_rate_pct | ANN =  100.00 | SOANN =  100.00 | Δ =   +0.00
availability_pct | ANN =    0.00 | SOANN =    0.00 | Δ =   +0.00

"""
