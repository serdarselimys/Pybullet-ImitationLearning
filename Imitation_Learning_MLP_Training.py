import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

DATA_PATH = "gait_dataset_pybullet_merged.csv"
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("Columns:", list(df.columns))

feature_cols = [
    'joy_fwd', 'joy_side', 'joy_spin',
    'phase', 'body_height',
    'lin_vel_x', 'lin_vel_y', 'lin_vel_z',
    'ang_vel_x', 'ang_vel_y', 'ang_vel_z'
]

X = df[feature_cols].values

LEG_ORDER = ["FL", "ML", "RL", "FR", "MR", "RR"]

# ---- Stateless feature set: matches the regenerated dataset ----
# raw phase (not sin/cos), dirt_mode included, NO per-leg phase encodings,
# NO prev_joints history.
feature_cols = [
    'joy_fwd', 'joy_side', 'joy_spin',
    'phase', 'body_height', 'dirt_mode',
    'lin_vel_x', 'lin_vel_y', 'lin_vel_z',
    'ang_vel_x', 'ang_vel_y', 'ang_vel_z',
]

# ---- Targets: current (actual) joint angles ----
target_cols = []
for leg in LEG_ORDER:
    target_cols += [f'{leg}3', f'{leg}2', f'{leg}1']

X = df[feature_cols].values
y = df[target_cols].values

print("Input shape:", X.shape)   # Expect (N, 12)
print("Output shape:", y.shape)  # Expect (N, 18)
print("Num input features:", len(feature_cols))
print("Num target dims:", len(target_cols))

scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test  = torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=512,
    shuffle=True
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=512,
    shuffle=False
)


class GaitNet(nn.Module):
    def __init__(self, input_size, output_size=18):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.model(x)

net = GaitNet(input_size=len(feature_cols), output_size=len(target_cols)).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

# Optional: learning rate scheduler (helps stability)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

EPOCHS = 50

for epoch in range(EPOCHS):

    # ---------------- TRAIN ----------------
    net.train()
    train_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        pred = net(xb)
        loss = criterion(pred, yb)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * xb.size(0)

    train_loss /= len(train_loader.dataset)

    # ---------------- EVAL ----------------
    net.eval()
    test_loss = 0.0

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)

            pred = net(xb)
            loss = criterion(pred, yb)

            test_loss += loss.item() * xb.size(0)

    test_loss /= len(test_loader.dataset)

    scheduler.step()

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train: {train_loss:.6f} | Test: {test_loss:.6f}")


MODEL_PATH = "gait_nn.pth"
FULL_MODEL_PATH = "gait_nn_full.pth"

torch.save(net.state_dict(), MODEL_PATH)
torch.save(net, FULL_MODEL_PATH)

with open("scaler_X.pkl", "wb") as f:
    pickle.dump(scaler_X, f)

with open("scaler_y.pkl", "wb") as f:
    pickle.dump(scaler_y, f)

# Save the exact feature/target column order used for training so that
# inference code can reconstruct the input vector correctly.
with open("feature_columns.pkl", "wb") as f:
    pickle.dump({"feature_cols": feature_cols, "target_cols": target_cols}, f)

print("Model, scalers, and column metadata saved")


net.eval()

sample_input = X_test[0].unsqueeze(0).to(device)

with torch.no_grad():
    pred_norm = net(sample_input).cpu().numpy()
    pred = scaler_y.inverse_transform(pred_norm)

print("\nSample prediction (first robot):")
print(pred.reshape(6, 3))  # 6 legs x 3 joints


# Check how many parameters the model has
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")
