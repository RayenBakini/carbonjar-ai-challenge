# Question 1 – LSTM Autoencoder for Anomaly Detection

## Problem
We need a real‐time pipeline to detect anomalies in time‐series emissions data. An LSTM autoencoder is a good choice: it learns to reconstruct normal sequences, and high reconstruction error flags anomalies.

---

## Implementation Details

1. **Data Simulation & Preprocessing**  
   - Generated noisy sine waves as surrogate emissions data (`generate_sine_data`).  
   - Scaled values to [0,1] with `MinMaxScaler`.  
   - Packed into a PyTorch `DataLoader`.

2. **Model Architecture**  
   - **Encoder**: LSTM that ingests the full sequence and outputs a final hidden state (`embedding_dim=64`).  
   - **Decoder**: LSTM that starts from the hidden state (repeated for each timestep) and reconstructs the original sequence.  

3. **Training Loop**  
   - Loss = MSE between input and reconstruction.  
   - Optimizer = Adam (lr=1e-3), 20 epochs, batch size 32.  
   - Printed epoch‐wise loss and plotted convergence.

---

## Preprocessing & Deployment Considerations

- **Scaling**: In production, apply the same `MinMaxScaler` fit on training data to new streams.  
- **Detrending**: If a strong seasonal component exists, remove trend/seasonality (e.g., via rolling‐window detrend) before encoding.  
- **Windowing**: Real‐time pipeline should buffer streaming data into fixed‐length windows (here 30 timesteps) before passing to the autoencoder.  
- **Thresholding**: Choose a reconstruction‐error threshold (e.g., 95th percentile on validation) to flag anomalies.


