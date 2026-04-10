import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_max_pool, BatchNorm, GlobalAttention
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, classification_report

FILE_PATH         = 'graphs_fusion_final.pt'
K_FOLDS           = 5
BATCH_SIZE        = 32
EPOCHS            = 150
LEARNING_RATE     = 0.0005
PATIENCE          = 30
HIDDEN_CHANNELS   = 64
DEVICE            = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BENIGN_ALPHA      = 1.0
MALICIOUS_ALPHA   = 10.0

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce_loss   = F.cross_entropy(logits, targets, reduction='none')
        pt        = torch.exp(-ce_loss)
        focal     = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None: focal = self.alpha[targets] * focal
        return focal.mean()

class ContractGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.conv1 = GATConv(num_node_features, HIDDEN_CHANNELS)
        self.bn1   = BatchNorm(HIDDEN_CHANNELS)
        self.conv2 = GATConv(HIDDEN_CHANNELS, HIDDEN_CHANNELS)
        self.bn2   = BatchNorm(HIDDEN_CHANNELS)
        self.conv3 = GATConv(HIDDEN_CHANNELS, HIDDEN_CHANNELS)
        self.bn3   = BatchNorm(HIDDEN_CHANNELS)
        
        gate_nn = torch.nn.Sequential(
            torch.nn.Linear(HIDDEN_CHANNELS, HIDDEN_CHANNELS // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_CHANNELS // 2, 1)
        )
        self.attention_pool = GlobalAttention(gate_nn=gate_nn)
        self.lin = torch.nn.Linear(HIDDEN_CHANNELS * 2, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.dropout(F.relu(self.bn1(self.conv1(x, edge_index))), p=0.2, training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x, edge_index))), p=0.2, training=self.training)
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x_attn = self.attention_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_attn, x_max], dim=1)
        x = F.dropout(x, p=0.1, training=self.training)
        return self.lin(x)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(data.x, data.edge_index, data.batch), data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, verbose=False):
    model.eval()
    y_true, y_pred = [], []
    for data in loader:
        data = data.to(DEVICE)
        pred = model(data.x, data.edge_index, data.batch).argmax(dim=1)
        y_true.extend(data.y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    metrics = {
        'accuracy'  : accuracy_score(y_true, y_pred),
        'f1_macro'  : f1_score(y_true, y_pred, average='macro',  zero_division=0),
        'mal_prec'  : precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'mal_rec'   : recall_score(   y_true, y_pred, pos_label=1, zero_division=0),
        'mal_f1'    : f1_score(       y_true, y_pred, pos_label=1, zero_division=0),
    }
    if verbose: print(classification_report(y_true, y_pred, target_names=['benign', 'malicious'], zero_division=0))
    return metrics

def main():
    all_data = torch.load(FILE_PATH)
    labels = np.array([d.y.item() for d in all_data])
    alpha = torch.tensor([BENIGN_ALPHA, MALICIOUS_ALPHA], dtype=torch.float).to(DEVICE)
    skf, fold_results = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42), []
    global_best_mal_f1, global_best_fold = 0.0, -1

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_data, labels)):
        print(f"\n{'='*55}\n  Fold {fold+1}/{K_FOLDS}\n{'='*55}")
        train_loader = DataLoader([all_data[i] for i in train_idx], batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader([all_data[i] for i in val_idx], batch_size=BATCH_SIZE)

        model = ContractGNN(all_data[0].num_node_features, 2).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        criterion = FocalLoss(alpha=alpha, gamma=2.0)

        best_mal_f1, best_metrics, patience_counter = 0.0, {}, 0
        for epoch in range(1, EPOCHS + 1):
            loss = train_one_epoch(model, train_loader, optimizer, criterion)
            metrics = evaluate(model, val_loader)

            if metrics['mal_f1'] > best_mal_f1:
                best_mal_f1, best_metrics, patience_counter = metrics['mal_f1'], metrics, 0
                torch.save(model.state_dict(), f'model_fold_{fold+1}_best.pth')
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                print(f"  Epoch {epoch:03d} | Loss: {loss:.4f} | Mal-F1: {metrics['mal_f1']:.4f} | Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE: break

        fold_results.append(best_metrics)
        model.load_state_dict(torch.load(f'model_fold_{fold+1}_best.pth', weights_only=True))
        if best_mal_f1 > global_best_mal_f1:
            global_best_mal_f1, global_best_fold = best_mal_f1, fold + 1
            torch.save(model.state_dict(), 'model_best_overall.pth')

    print(f"\n  Best overall model comes from Fold {global_best_fold}, Malicious F1 = {global_best_mal_f1:.4f}")

if __name__ == "__main__":
    main()