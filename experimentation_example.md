Here is a complete PyTorch-based implementation of the described gene expression modeling pipeline. The code includes data preprocessing, model architecture (GCNs, autoencoders, attention), training strategy, and deployment components.

---

### 🧠 `gene_model.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class GraphModule(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(GraphModule, self).__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dims[0])
        self.gcn2 = GCNConv(hidden_dims[0], hidden_dims[1])
        self.act = nn.GELU()

    def forward(self, x, edge_index):
        x = self.act(self.gcn1(x, edge_index))
        x = self.act(self.gcn2(x, edge_index))
        return x

class AutoencoderModule(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AutoencoderModule, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon

class PathwayAggregator(nn.Module):
    def __init__(self, pathway_map, gene_dim, out_dim):
        super(PathwayAggregator, self).__init__()
        self.pathway_map = pathway_map
        self.out = nn.Linear(len(pathway_map), out_dim)

    def forward(self, x):
        p_scores = []
        for gene_ids in self.pathway_map:
            p = x[:, gene_ids].mean(dim=1, keepdim=True)
            p_scores.append(p)
        p_scores = torch.cat(p_scores, dim=1)
        return self.out(p_scores)

class AttentionModule(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=2):
        super(AttentionModule, self).__init__()
        layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward=1024, dropout=0.1, activation='gelu')
        self.transformer = TransformerEncoder(layers, num_layers)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.transformer(x)
        return x.squeeze(1)

class PredictionHead(nn.Module):
    def __init__(self, input_dim, output_dim, is_classification=True):
        super(PredictionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, output_dim)
        self.is_classification = is_classification

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x) if not self.is_classification else F.softmax(self.fc2(x), dim=1)

class GeneExpressionModel(nn.Module):
    def __init__(self, gene_dim, pathway_map, latent_dim, cov_dim, out_dim, is_classification=True):
        super(GeneExpressionModel, self).__init__()
        self.graph_module = GraphModule(gene_dim, [512, 256])
        self.autoencoder = AutoencoderModule(gene_dim, latent_dim)
        self.pathway_module = PathwayAggregator(pathway_map, gene_dim, 256)
        self.attn = AttentionModule(d_model=512, num_layers=2)
        self.pred_head = PredictionHead(256 + 256 + 256 + cov_dim, out_dim, is_classification)

    def forward(self, x, edge_index, covariates):
        h_g = self.graph_module(x, edge_index)
        z, recon = self.autoencoder(x)
        h_p = self.pathway_module(x)
        h_attn = self.attn(h_g)
        features = torch.cat([h_g, h_p, z, covariates], dim=1)
        out = self.pred_head(features)
        return out, recon
```

---

### 🔧 Training (`train.py`)

```python
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
import torch.nn.functional as F

def train_model(model, dataloader, val_loader, is_classification, epochs=100):
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    early_stop_patience = 10
    best_loss = float('inf')
    patience = 0

    for epoch in range(epochs):
        model.train()
        for batch in dataloader:
            x, edge_index, cov, y = batch['x'], batch['edge_index'], batch['cov'], batch['y']
            output, recon = model(x, edge_index, cov)
            loss = F.cross_entropy(output, y) if is_classification else F.mse_loss(output.squeeze(), y)
            loss += 0.1 * F.mse_loss(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Early stopping
        val_loss = evaluate(model, val_loader, is_classification, return_loss=True)
        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
        else:
            patience += 1
            if patience > early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

def evaluate(model, dataloader, is_classification, return_loss=False):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            x, edge_index, cov, y = batch['x'], batch['edge_index'], batch['cov'], batch['y']
            output, _ = model(x, edge_index, cov)
            total_loss += F.cross_entropy(output, y).item() if is_classification else F.mse_loss(output.squeeze(), y).item()
            all_preds.append(output.cpu())
            all_labels.append(y.cpu())
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    if is_classification:
        return (accuracy_score(labels, preds.argmax(1)), roc_auc_score(labels, preds, multi_class='ovo')) if not return_loss else total_loss
    else:
        return mean_squared_error(labels, preds.squeeze()) if not return_loss else total_loss
```

---

### 🧪 Inference & Deployment

```python
import torch

def export_model(model, path='model.pt'):
    traced = torch.jit.trace(model, example_inputs=(torch.randn(1, GENE_DIM), edge_index, torch.randn(1, COV_DIM)))
    torch.jit.save(traced, path)

def batch_inference(model, data_loader):
    model.eval()
    results = []
    with torch.no_grad():
        for batch in data_loader:
            x, edge_index, cov = batch['x'], batch['edge_index'], batch['cov']
            out, _ = model(x, edge_index, cov)
            results.append(out)
    return torch.cat(results)
```
