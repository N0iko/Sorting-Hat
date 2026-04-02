import torch
import torch.nn.functional as F
import numpy as np
import os
import pickle
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, BatchNorm


# ──────────────────────── Model architecture (must match training exactly) ───────────────────────

class ContractGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(ContractGNN, self).__init__()
        hidden = 64
        self.conv1 = GATConv(num_node_features, hidden)
        self.bn1   = BatchNorm(hidden)
        self.conv2 = GATConv(hidden, hidden)
        self.bn2   = BatchNorm(hidden)
        self.conv3 = GATConv(hidden, hidden)
        self.bn3   = BatchNorm(hidden)
        self.lin   = torch.nn.Linear(hidden * 2, num_classes)

    def forward(self, x, edge_index, batch):
        # In eval mode Dropout is disabled automatically, no need to specify it
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.bn3(self.conv3(x, edge_index))
        x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return self.lin(x)


# ──────────────────────────── Path configuration ────────────────────────────────────────

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Fix: training script saves model_fold_X_best.pth, and model_best_overall.pth is also supported
MODEL_PATHS = [os.path.join(PROJECT_ROOT, f'model_fold_{i}_best.pth') for i in range(1, 6)]

# feature_scaler.pkl is located in the same directory as the training data
SCALER_PATH = os.path.join(
    os.path.dirname(CURRENT_DIR),
    "facts file convertor", "processed_fusion", "feature_scaler.pkl"
)

TARGET_FACTS = r""  # Default input path when running standalone

# Classification thresholds (consistent with pipeline.py)
MALICIOUS_THRESHOLD = 0.75   # 75% and above considered highly malicious
SUSPICIOUS_THRESHOLD = 0.25  # 25% to 75% considered suspicious


# ──────────────────────── Utility functions ────────────────────────────────────────────

def load_scaler(scaler_path=SCALER_PATH):
    """Load the StandardScaler saved during training. Return None and warn if missing."""
    if not os.path.exists(scaler_path):
        print(f"[!] Warning: feature_scaler.pkl not found -> {scaler_path}")
        print(f"[!] Skipping feature normalization; detection results may be inaccurate!")
        return None
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print(f"[✓] Loaded feature scaler: {scaler_path}")
    return scaler


def apply_scaler(data, scaler):
    """
    Transform node features using the same scaler used during training.
    If scaler is None, return unchanged.
    """
    if scaler is None:
        return data
    x_np = data.x.cpu().numpy()
    x_scaled = scaler.transform(x_np)
    data.x = torch.tensor(x_scaled, dtype=torch.float)
    return data


from facts_to_pyg import convert_contract


# ──────────────────────── Core inference logic ───────────────────────────────────────

def _infer(facts_dir, scaler, device):
    """
    Internal inference function shared by run_detection / run_detection_result.
    Returns result dict, or None on failure.
    """
    if not os.path.exists(facts_dir):
        print(f"[!] Facts directory not found: {facts_dir}")
        return None

    # 1. Convert to PyG Data
    data = convert_contract(facts_dir, label=0)
    if data is None:
        print("[!] Facts conversion failed")
        return None

    # 2. ── Key: apply the training scaler to features ──
    data = apply_scaler(data, scaler)
    data = data.to(device)

    # 3. Perform inference for each model and collect malicious probabilities
    all_probs = []
    loaded_count = 0

    for path in MODEL_PATHS:
        if not os.path.exists(path):
            print(f"[?] Model file not found, skipped: {path}")
            continue

        model = ContractGNN(data.num_node_features, 2).to(device)
        model.load_state_dict(
            torch.load(path, map_location=device, weights_only=True)
        )
        model.eval()

        with torch.no_grad():
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
            out   = model(data.x, data.edge_index, batch)
            prob  = F.softmax(out, dim=1)[0][1].item()
            all_probs.append(prob)
            loaded_count += 1

    if not all_probs:
        print("[!] No available models, inference failed")
        return None

    # 4. Aggregate scores
    final_score = float(np.mean(all_probs))

    # consistency: use 1 - clamp(std, 0, 1) to prevent negatives; single model yields 1.0
    if len(all_probs) > 1:
        consistency = float(1.0 - min(np.std(all_probs), 1.0))
    else:
        consistency = 1.0

    # 5. Classification (covers full range: [0.75, 1.0], [0.25, 0.75), [0.0, 0.25))
    if final_score >= MALICIOUS_THRESHOLD:
        conclusion     = "Highly Malicious Contract"
        conclusion_key = "malicious"
    elif final_score >= SUSPICIOUS_THRESHOLD:
        conclusion     = "Suspicious Contract"
        conclusion_key = "suspicious"
    else:
        conclusion     = "Benign Contract"
        conclusion_key = "benign"

    return {
        "contract_name"   : os.path.basename(facts_dir),
        "score"           : final_score,
        "score_pct"       : f"{final_score:.2%}",
        "consistency"     : consistency,
        "consistency_pct" : f"{consistency:.2%}",
        "conclusion"      : conclusion,
        "conclusion_key"  : conclusion_key,
        "models_used"     : loaded_count,
    }


# ──────────────────────── Public interface ───────────────────────────────────────────

def run_detection(target_facts_dir=None):
    """Print a detection report (for command line / pipeline invocation)."""
    facts_dir = target_facts_dir if target_facts_dir is not None else TARGET_FACTS
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Using device: {device}")

    scaler = load_scaler()
    result = _infer(facts_dir, scaler, device)

    if result is None:
        print("[✗] Detection failed")
        return None

    print("\n" + "=" * 45)
    print(f"  🔍 Malicious contract detection report")
    print("-" * 45)
    print(f"  Contract name: {result['contract_name']}")
    print(f"  Malicious score: {result['score_pct']}  (raw: {result['score']:.6f})")
    print(f"  Model consistency: {result['consistency_pct']}")
    print(f"  Models used: {result['models_used']}")
    print("-" * 45)
    print(f"  Conclusion: {result['conclusion']}")
    print("=" * 45)

    return result


def run_detection_result(target_facts_dir=None):
    """
    Silent mode: only return the result dict (for pipeline.py API mode).
    Reuse _infer directly without duplicating logic.
    """
    facts_dir = target_facts_dir if target_facts_dir is not None else TARGET_FACTS
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler    = load_scaler()
    return _infer(facts_dir, scaler, device)


if __name__ == "__main__":
    run_detection()
