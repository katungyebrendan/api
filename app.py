from fastapi import FastAPI
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import uvicorn

# FastAPI app
app = FastAPI()

# Define Student Model
class StudentGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(StudentGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
student_model = StudentGNN(5, 32, 2)  # Adjust input size
student_model.load_state_dict(torch.load("student_model.pth", map_location=device, weights_only=False))
student_model.eval()

@app.post("/predict/")
async def predict(data: dict):
    try:
        # Convert input to tensor
        X = torch.tensor([data["features"]], dtype=torch.float)
        edge_index = torch.tensor(data["edges"], dtype=torch.long).t().contiguous()

        with torch.no_grad():
            prediction = student_model(X, edge_index).argmax(dim=1).item()
        
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}

# Run locally for testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
