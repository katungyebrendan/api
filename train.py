import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import networkx as nx


df = pd.read_csv("balanced_dataset.csv")

X = df[['tick', 'cape', 'cattle', 'bio5']].values
y = df['ECF'].values


smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


node_features = torch.tensor(X_resampled, dtype=torch.float)
labels = torch.tensor(y_resampled, dtype=torch.long)


num_nodes = len(X_resampled)
G = nx.erdos_renyi_graph(num_nodes, p=0.1)
edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()


data = Data(x=node_features, edge_index=edge_index, y=labels)

class StudentGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(StudentGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StudentGNN(node_features.shape[1], 32, len(torch.unique(labels))).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x.to(device), data.edge_index.to(device))
    loss = F.nll_loss(out, data.y.to(device))
    loss.backward()
    optimizer.step()
    return loss.item()


print("Training Student Model...")
for epoch in range(100):
    loss = train()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

torch.save(model, "student_model.pth")  
print("✅ Model saved as student_model.pth")
