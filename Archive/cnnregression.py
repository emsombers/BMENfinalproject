import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class FC3DVolumeDataset(Dataset):
    def __init__(self, fc_dir, label_dir, target_column='mean_scaled_anxiety'):
        self.samples = []
        self.targets = []
        self.subject_ids = []
        file_list = sorted([f for f in os.listdir(fc_dir) if f.endswith('_fc.npy')])
        for fname in file_list:
            subj_id = fname.replace('_fc.npy', '')
            label_path = os.path.join(label_dir, f"{subj_id}_labels.csv")
            if not os.path.exists(label_path):
                continue
            df = pd.read_csv(label_path)
            if target_column not in df.columns:
                continue
            fc_path = os.path.join(fc_dir, fname)
            fc_array = np.load(fc_path)  # shape: (91, 100, 100)
            if fc_array.shape != (91, 100, 100):
                continue
            # Add a channel dimension: (1, 91, 100, 100)
            self.samples.append(fc_array[None, ...])
            self.targets.append(df[target_column].values[0])
            self.subject_ids.append(subj_id)
        self.samples = torch.tensor(np.array(self.samples), dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]

class FC3DCNNRegressor(nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.1):
        super().__init__()
        # Input: (B, 1, 91, 100, 100)

        self.conv1 = nn.Conv3d(1, 16, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2))  # → (B, 16, 91, 100, 100)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))  # → (B, 16, 45, 50, 50)

        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=1)  # → (B, 32, 45, 50, 50)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))  # → (B, 32, 22, 25, 25)

        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1)  # → (B, 64, 22, 25, 25)
        self.pool3 = nn.AdaptiveAvgPool3d((1, 1, 1))  # → (B, 64, 1, 1, 1)

        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(64, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # (B, 16, 91, 100, 100)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)  # (B, 64, 1, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten → (B, 64)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x).squeeze(1)
        return x

def train(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


def plot_predictions(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds.extend(model(x).cpu().numpy())
            targets.extend(y.numpy())
    plt.figure(figsize=(6,6))
    plt.scatter(targets, preds, alpha=0.7)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compute_global_pca(fc_dir, n_components=1000):
    all_data = []
    for fname in sorted([f for f in os.listdir(fc_dir) if f.endswith('_fc.npy')]):
        fc_array = np.load(os.path.join(fc_dir, fname))  # (91, 100, 100)
        if fc_array.shape != (91, 100, 100):
            continue
        all_data.append(fc_array.reshape(91, -1))  # (91, 10000)
    stacked = np.concatenate(all_data, axis=0)  # (91 * num_subjects, 10000)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(stacked)
    pca = PCA(n_components=n_components)
    pca.fit(scaled)
    return scaler, pca

class FC3DPCADataset(Dataset):
    def __init__(self, fc_dir, label_dir, scaler, pca, target_column='mean_scaled_anxiety'):
        self.samples = []
        self.targets = []
        self.subject_ids = []
        for fname in sorted([f for f in os.listdir(fc_dir) if f.endswith('_fc.npy')]):
            subj_id = fname.replace('_fc.npy', '')
            label_path = os.path.join(label_dir, f"{subj_id}_labels.csv")
            if not os.path.exists(label_path):
                continue
            df = pd.read_csv(label_path)
            if target_column not in df.columns:
                continue
            fc_array = np.load(os.path.join(fc_dir, fname))
            if fc_array.shape != (91, 100, 100):
                continue
            flat = fc_array.reshape(91, -1)
            scaled = scaler.transform(flat)
            reduced = pca.transform(scaled)  # (91, n_components)
            n_comp = reduced.shape[1]
            side = int(np.ceil(np.sqrt(n_comp)))
            padded = np.zeros((91, side * side))
            padded[:, :n_comp] = reduced
            vol = padded.reshape(1, 91, side, side)  # (1, 91, H, W)
            self.samples.append(vol)
            self.targets.append(df[target_column].values[0])
            self.subject_ids.append(subj_id)
        self.samples = torch.tensor(np.array(self.samples), dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]

def main():
    fc_dir = 'fc_matrices'
    label_dir = 'fc_matrices'
    target_column = 'mean_scaled_anxiety'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    scaler, pca = compute_global_pca(fc_dir, n_components=1000)

    dataset = FC3DPCADataset(fc_dir, label_dir, scaler, pca, target_column)

    unique_subjects = list(sorted(set(dataset.subject_ids)))
    train_subjects, test_subjects = train_test_split(unique_subjects, test_size=0.2, random_state=42)

    train_indices = [i for i, sid in enumerate(dataset.subject_ids) if sid in train_subjects]
    test_indices = [i for i, sid in enumerate(dataset.subject_ids) if sid in test_subjects]

    train_ds = torch.utils.data.Subset(dataset, train_indices)
    test_ds = torch.utils.data.Subset(dataset, test_indices)

    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=4)

    model = FC3DCNNRegressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    for epoch in range(1, 31):
        train_loss = train(model, train_dl, optimizer, loss_fn, device)
        test_loss = evaluate(model, test_dl, loss_fn, device)
        print(f"Epoch {epoch:02d}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")


    plot_predictions(model, test_dl, device)

if __name__ == '__main__':
    main()
