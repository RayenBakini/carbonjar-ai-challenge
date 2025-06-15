# ============================================
# LSTM Autoencoder for Anomaly Detection
# Author: Rayen Bakini
# ============================================

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 1) Simulate time-series data: noisy sine waves
def generate_sine_data(n_samples=500, seq_len=30):
    """
    Returns data shape (n_samples, seq_len, 1)
    """
    x = np.linspace(0, 4 * np.pi, seq_len)
    data = []
    for _ in range(n_samples):
        phase = np.random.rand() * 2 * np.pi
        series = np.sin(x + phase) + 0.1 * np.random.randn(seq_len)
        data.append(series)
    data = np.array(data).reshape(n_samples, seq_len, 1)
    return data

# 2) Prepare dataset & dataloader
raw_data = generate_sine_data()
scaler = MinMaxScaler(feature_range=(0, 1))
# flatten for scaler then reshape back
ns, sl, nf = raw_data.shape
flat = raw_data.reshape(-1, nf)
scaled = scaler.fit_transform(flat).reshape(ns, sl, nf)
dataset = TensorDataset(torch.from_numpy(scaled).float())
loader  = DataLoader(dataset, batch_size=32, shuffle=True)

# 3) Define the LSTM Autoencoder
class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len: int, n_features: int, embedding_dim=64):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        # encoder: many-to-one
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )
        # decoder: one-to-many
        self.decoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=n_features,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        # x: [B, T, F]
        # 1) encode entire sequence → take final hidden state
        _, (h_n, c_n) = self.encoder(x)
        # h_n: [1, B, E] → squeeze layer dimension
        h_n = h_n.permute(1, 0, 2)  # → [B, 1, E]
        # 2) prepare decoder input: repeat h_n for each time step
        dec_input = h_n.repeat(1, self.seq_len, 1)  # → [B, T, E]
        # 3) decode to reconstruct sequence
        decoded, _ = self.decoder(dec_input)
        # decoded: [B, T, F]
        return decoded

# 4) Instantiate model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMAutoencoder(seq_len=sl, n_features=nf, embedding_dim=64).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 5) Training loop
n_epochs = 20
history = []

for epoch in range(1, n_epochs + 1):
    epoch_loss = 0.0
    model.train()
    for (batch,) in loader:
        batch = batch.to(device)  # [B, T, F]
        optimizer.zero_grad()
        # forward
        recon = model(batch)
        # compute reconstruction error
        loss = criterion(recon, batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch.size(0)
    epoch_loss /= len(loader.dataset)
    history.append(epoch_loss)
    print(f"Epoch {epoch:02d}/{n_epochs} — Loss: {epoch_loss:.6f}")

# 6) Plot training loss
plt.figure(figsize=(6,4))
plt.plot(history, marker='o')
plt.title("Training Loss (MSE) over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.tight_layout()
plt.show()
