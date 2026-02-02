import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


FILE_PATH = 'all_data_for_cv.pt'
K_FOLDS = 5
BATCH_SIZE = 32
EPOCHS = 40  
LEARNING_RATE = 0.001
HIDDEN_CHANNELS = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ContractGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(ContractGNN, self).__init__()
        self.conv1 = GATConv(num_node_features, HIDDEN_CHANNELS)
        self.conv2 = GATConv(HIDDEN_CHANNELS, HIDDEN_CHANNELS)
        self.conv3 = GATConv(HIDDEN_CHANNELS, HIDDEN_CHANNELS)
        self.lin = torch.nn.Linear(HIDDEN_CHANNELS * 2, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return self.lin(x)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(DEVICE)
            out = model(data.x, data.edge_index, data.batch)
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(out.argmax(dim=1).cpu().numpy())
    return {
        'acc': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'prec': precision_score(y_true, y_pred, zero_division=0),
        'rec': recall_score(y_true, y_pred, zero_division=0)
    }


def run_cv():
    all_data = torch.load(FILE_PATH)
    labels = [d.y.item() for d in all_data]
    
    # Initialize stratified K-fold cross-validation
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    fold_results = []
    print(f"[*] Starting {K_FOLDS}-Fold cross-validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n--- Fold {fold+1} training ---")
        
        # Prepare the data for this fold
        train_set = [all_data[i] for i in train_idx]
        val_set = [all_data[i] for i in val_idx]
        
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

        # Initialize the model and optimizer
        num_features = all_data[0].num_node_features
        model = ContractGNN(num_features, 2).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0]).to(DEVICE))

        best_f1 = 0
        best_metrics = {}

        for epoch in range(1, EPOCHS + 1):
            train_one_epoch(model, train_loader, optimizer, criterion)
            metrics = evaluate(model, val_loader)
            
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_metrics = metrics
        
        fold_results.append(best_metrics)
        torch.save(model.state_dict(), f'model_fold_{fold+1}.pth')
        print(f"Best results for fold {fold+1}: F1={best_metrics['f1']:.4f}, Recall={best_metrics['rec']:.4f}")

    # Output the average results
    print("\n" + "="*30)
    print("Final average results (Final Summary)")
    print("="*30)
    for m in ['acc', 'prec', 'rec', 'f1']:
        vals = [r[m] for r in fold_results]
        print(f"{m.upper()}: {np.mean(vals):.4f} (+/- {np.std(vals):.4f})")

if __name__ == "__main__":
    run_cv()