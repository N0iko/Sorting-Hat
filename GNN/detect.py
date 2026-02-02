import torch
import torch.nn.functional as F
import numpy as np
import os
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool


class ContractGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(ContractGNN, self).__init__()
        hidden = 64 
        self.conv1 = GATConv(num_node_features, hidden)
        self.conv2 = GATConv(hidden, hidden)
        self.conv3 = GATConv(hidden, hidden)
        self.lin = torch.nn.Linear(hidden * 2, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return self.lin(x)


MODEL_DIR = r"C:\Users\john\Desktop\FYP" 
MODEL_PATHS = [os.path.join(MODEL_DIR, f'model_fold_{i}.pth') for i in range(1, 6)]


TARGET_FACTS = r"  " #input path


THRESHOLD = 0.75


from facts_to_pyg import convert_contract

def run_detection(target_facts_dir=None):
   
    facts_dir = target_facts_dir if target_facts_dir is not None else TARGET_FACTS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Using device: {device}")


    if not os.path.exists(facts_dir):
        print(f"[!] Error: Could not find folder {facts_dir}")
        return

    data = convert_contract(facts_dir, label=0)
    if data is None:
        print("[!] Error: Facts conversion failed")
        return
    
    data = data.to(device)
    num_features = data.num_node_features
    

    all_probs = []
    print(f"[*] Performing cross-validation detection through 5 models...")
    
    for path in MODEL_PATHS:
        if not os.path.exists(path):
            print(f"[?] Warning: Could not find model weights {path}, skipping")
            continue
            
        model = ContractGNN(num_features, 2).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        
        with torch.no_grad():
            batch = torch.zeros(data.num_nodes, dtype=torch.long).to(device)
            out = model(data.x, data.edge_index, batch)
            prob = F.softmax(out, dim=1)[0][1].item()
            all_probs.append(prob)

    if not all_probs:
        print("[!] Error: No available models for inference")
        return None


    final_score = np.mean(all_probs)
    consistency = 1 - np.std(all_probs)  

    if final_score >= THRESHOLD:
        conclusion = "Highly Malicious Contract"
        conclusion_key = "malicious"
    elif final_score >= 0.5:
        conclusion = "Suspicious Contract"
        conclusion_key = "suspicious"
    else:
        conclusion = "Benign Contract"
        conclusion_key = "benign"

    result = {
        "contract_name": os.path.basename(facts_dir),
        "score": float(final_score),
        "score_pct": f"{final_score:.2%}",
        "consistency": float(consistency),
        "consistency_pct": f"{consistency:.2%}",
        "conclusion": conclusion,
        "conclusion_key": conclusion_key,
    }

    print("\n" + "="*40)
    print(f"🔍 Malicious Contract Detection Report")
    print("-" * 40)
    print(f"Contract Name: {result['contract_name']}")
    print(f"Malicious Score: {result['score_pct']}")
    print(f"Model Confidence: {result['consistency_pct']}")
    print("-" * 40)
    print(f"Conclusion: {conclusion}")
    print("="*40)

    return result


def run_detection_result(target_facts_dir=None):
    facts_dir = target_facts_dir if target_facts_dir is not None else TARGET_FACTS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(facts_dir):
        return None
    data = convert_contract(facts_dir, label=0)
    if data is None:
        return None
    data = data.to(device)
    num_features = data.num_node_features
    all_probs = []
    for path in MODEL_PATHS:
        if not os.path.exists(path):
            continue
        model = ContractGNN(num_features, 2).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        with torch.no_grad():
            batch = torch.zeros(data.num_nodes, dtype=torch.long).to(device)
            out = model(data.x, data.edge_index, batch)
            prob = F.softmax(out, dim=1)[0][1].item()
            all_probs.append(prob)
    if not all_probs:
        return None
    final_score = np.mean(all_probs)
    consistency = 1 - np.std(all_probs)
    if final_score >= THRESHOLD:
        conclusion, conclusion_key = "Highly Malicious Contract", "malicious"
    elif final_score >= 0.5:
        conclusion, conclusion_key = "Suspicious Contract", "suspicious"
    else:
        conclusion, conclusion_key = "Benign Contract", "benign"
    return {
        "contract_name": os.path.basename(facts_dir),
        "score": float(final_score),
        "score_pct": f"{final_score:.2%}",
        "consistency": float(consistency),
        "consistency_pct": f"{consistency:.2%}",
        "conclusion": conclusion,
        "conclusion_key": conclusion_key,
    }


if __name__ == "__main__":
    run_detection()