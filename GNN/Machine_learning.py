import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, BatchNorm, GlobalAttention
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, recall_score, accuracy_score, precision_score,
    classification_report
)

# ================= Configuration =================
FILE_PATH         = 'graphs_fusion_final.pt'
K_FOLDS           = 5
BATCH_SIZE        = 32
EPOCHS            = 150
LEARNING_RATE     = 0.0005
PATIENCE          = 30
HIDDEN_CHANNELS   = 64
DEVICE            = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Sample ratio: 5000/798 ≈ 6.27
# Suggested malicious alpha starts from 4.0~6.0, adjustable
BENIGN_ALPHA    = 1.0
MALICIOUS_ALPHA = 5.0
# ========================================


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce_loss   = F.cross_entropy(logits, targets, reduction='none')
        pt        = torch.exp(-ce_loss)
        focal     = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            focal = self.alpha[targets] * focal
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
        
        # 1. Declare the gate network required for Attention Pooling
        # This small neural network computes an importance score for each node (basic block)
        gate_nn = torch.nn.Sequential(
            torch.nn.Linear(HIDDEN_CHANNELS, HIDDEN_CHANNELS // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_CHANNELS // 2, 1)
        )
        self.attention_pool = GlobalAttention(gate_nn=gate_nn)
        
        # 2. Adjust the classifier input dimension
        # attention_pool outputs HIDDEN_CHANNELS, global_max_pool outputs HIDDEN_CHANNELS
        self.lin = torch.nn.Linear(HIDDEN_CHANNELS * 2, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.dropout(F.relu(self.bn1(self.conv1(x, edge_index))), p=0.2, training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x, edge_index))), p=0.2, training=self.training)
        # Note: add ReLU so the last convolution also has non-linear activation
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        
        # 3. Replace mean pooling with Attention Pooling
        # x_attn will focus on nodes that contain malicious opcodes
        x_attn = self.attention_pool(x, batch)
        # Keep Max Pooling as well to preserve extreme-value features, then concatenate both
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
    """
    Return overall metrics plus malicious-class-specific metrics.
    Early stopping is based on malicious-class F1 (mal_f1).
    """
    model.eval()
    y_true, y_pred = [], []
    for data in loader:
        data = data.to(DEVICE)
        pred = model(data.x, data.edge_index, data.batch).argmax(dim=1)
        y_true.extend(data.y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {
        # Overall metrics
        'accuracy'  : accuracy_score(y_true, y_pred),
        'f1_macro'  : f1_score(y_true, y_pred, average='macro',  zero_division=0),
        # Malicious class metrics (label=1) — used for early stopping/model selection
        'mal_prec'  : precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'mal_rec'   : recall_score(   y_true, y_pred, pos_label=1, zero_division=0),
        'mal_f1'    : f1_score(       y_true, y_pred, pos_label=1, zero_division=0),
    }

    if verbose:
        print(classification_report(y_true, y_pred,
                                    target_names=['benign', 'malicious'],
                                    zero_division=0))
    return metrics


def main():
    print(f"Device: {DEVICE}")
    print(f"[*] Loading data: {FILE_PATH}")
    all_data = torch.load(FILE_PATH)
    labels   = np.array([d.y.item() for d in all_data])

    n_benign   = int((labels == 0).sum())
    n_malicious = int((labels == 1).sum())
    print(f"[*] Data distribution -> benign: {n_benign}, malicious: {n_malicious}, "
          f"ratio: {n_benign/n_malicious:.2f}:1")
    print(f"[*] Focal Loss alpha -> benign={BENIGN_ALPHA}, malicious={MALICIOUS_ALPHA}\n")

    alpha = torch.tensor([BENIGN_ALPHA, MALICIOUS_ALPHA], dtype=torch.float).to(DEVICE)

    skf          = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_results = []

    # Track the overall best model to save the final checkpoint
    global_best_mal_f1 = 0.0
    global_best_fold   = -1

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_data, labels)):
        print(f"{'='*55}")
        print(f"  Fold {fold+1}/{K_FOLDS}")
        print(f"{'='*55}")

        train_set    = [all_data[i] for i in train_idx]
        val_set      = [all_data[i] for i in val_idx]
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE)

        model     = ContractGNN(all_data[0].num_node_features, 2).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        criterion = FocalLoss(alpha=alpha, gamma=2.0)

        best_mal_f1      = 0.0
        best_metrics     = {}
        patience_counter = 0

        for epoch in range(1, EPOCHS + 1):
            loss    = train_one_epoch(model, train_loader, optimizer, criterion)
            metrics = evaluate(model, val_loader)

            # ── Early stopping criterion: malicious-class F1 ──
            if metrics['mal_f1'] > best_mal_f1:
                best_mal_f1      = metrics['mal_f1']
                best_metrics     = metrics
                patience_counter = 0
                torch.save(model.state_dict(), f'model_fold_{fold+1}_best.pth')
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                print(
                    f"  Epoch {epoch:03d} | Loss: {loss:.4f} | "
                    f"Mal-F1: {metrics['mal_f1']:.4f} | "
                    f"Mal-Rec: {metrics['mal_rec']:.4f} | "
                    f"Mal-Prec: {metrics['mal_prec']:.4f} | "
                    f"Patience: {patience_counter}/{PATIENCE}"
                )

            if patience_counter >= PATIENCE:
                print(f"  [!] Early stopping at epoch {epoch}. Best Mal-F1: {best_mal_f1:.4f}")
                break

        fold_results.append(best_metrics)
        print(f"\n  Fold {fold+1} Best -> "
              f"Mal-F1={best_metrics.get('mal_f1',0):.4f} | "
              f"Mal-Prec={best_metrics.get('mal_prec',0):.4f} | "
              f"Mal-Rec={best_metrics.get('mal_rec',0):.4f} | "
              f"Acc={best_metrics.get('accuracy',0):.4f}")

        # Print the classification report for this fold
        print(f"\n  [Fold {fold+1}] Detailed classification report on validation set:")
        model.load_state_dict(torch.load(f'model_fold_{fold+1}_best.pth'))
        evaluate(model, val_loader, verbose=True)

        # Update overall best model
        if best_mal_f1 > global_best_mal_f1:
            global_best_mal_f1 = best_mal_f1
            global_best_fold   = fold + 1
            torch.save(model.state_dict(), 'model_best_overall.pth')

    # ── Summary ──
    print("\n" + "="*55)
    print("  5-Fold cross-validation summary")
    print("="*55)
    for key, label in [
        ('mal_f1',   'Malicious F1'),
        ('mal_prec', 'Malicious Prec'),
        ('mal_rec',  'Malicious Rec'),
        ('f1_macro', 'Macro F1'),
        ('accuracy', 'Accuracy'),
    ]:
        vals = [r.get(key, 0) for r in fold_results]
        print(f"  {label}: {np.mean(vals):.4f}  (+/- {np.std(vals):.4f})")

    print(f"\n  Best overall model comes from Fold {global_best_fold}, "
          f"Malicious F1 = {global_best_mal_f1:.4f}")
    print(f"  Saved as: model_best_overall.pth")
    print("="*55)


if __name__ == "__main__":
    main()
