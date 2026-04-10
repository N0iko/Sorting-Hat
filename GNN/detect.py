import torch
import torch.nn.functional as F
import numpy as np
import os
import pickle
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_max_pool, BatchNorm, GlobalAttention



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
        
        gate_nn = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden // 2, 1)
        )
        self.attention_pool = GlobalAttention(gate_nn=gate_nn)
        self.lin = torch.nn.Linear(hidden * 2, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        
        x_attn = self.attention_pool(x, batch)
        x_max = global_max_pool(x, batch)
        
        x = torch.cat([x_attn, x_max], dim=1)
        return self.lin(x)



CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

MODEL_PATHS = [os.path.join(PROJECT_ROOT, f'model_fold_{i}_best.pth') for i in range(1, 6)]
SCALER_PATH = os.path.join(os.path.dirname(CURRENT_DIR), "facts file convertor", "processed_fusion", "feature_scaler.pkl")

TARGET_FACTS = r""
MALICIOUS_THRESHOLD = 0.60
SUSPICIOUS_THRESHOLD = 0.30


def load_scaler(scaler_path=SCALER_PATH):
    if not os.path.exists(scaler_path): return None
    with open(scaler_path, "rb") as f: return pickle.load(f)

def apply_scaler(data, scaler):
    if scaler is None: return data
    data.x = torch.tensor(scaler.transform(data.x.cpu().numpy()), dtype=torch.float)
    return data

from facts_to_pyg import convert_contract

def _infer(facts_dir, scaler, device):
    if not os.path.exists(facts_dir): return None
    data = convert_contract(facts_dir, label=0)
    if data is None: return None

    data = apply_scaler(data, scaler).to(device)
    all_probs, loaded_count = [], 0

    for path in MODEL_PATHS:
        if not os.path.exists(path): continue
        model = ContractGNN(data.num_node_features, 2).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()

        with torch.no_grad():
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
            prob  = F.softmax(model(data.x, data.edge_index, batch), dim=1)[0][1].item()
            all_probs.append(prob)
            loaded_count += 1

    if not all_probs: return None

    final_score = float(np.mean(all_probs))
    consistency = float(1.0 - min(np.std(all_probs), 1.0)) if len(all_probs) > 1 else 1.0

    if final_score >= MALICIOUS_THRESHOLD:
        conclusion, conclusion_key = "Highly Malicious Contract", "malicious"
    elif final_score >= SUSPICIOUS_THRESHOLD:
        conclusion, conclusion_key = "Suspicious Contract", "suspicious"
    else:
        conclusion, conclusion_key = "Benign Contract", "benign"

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

def run_detection(target_facts_dir=None):
    facts_dir = target_facts_dir if target_facts_dir is not None else TARGET_FACTS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result = _infer(facts_dir, load_scaler(), device)
    
    if result is None: return None
    print(f"\n=============================================")
    print(f"  Contract name: {result['contract_name']}")
    print(f"  Malicious score: {result['score_pct']}  (raw: {result['score']:.6f})")
    print(f"  Conclusion: {result['conclusion']}")
    print(f"=============================================")
    return result

def run_detection_result(target_facts_dir=None):
    facts_dir = target_facts_dir if target_facts_dir is not None else TARGET_FACTS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return _infer(facts_dir, load_scaler(), device)

if __name__ == "__main__":
    run_detection()