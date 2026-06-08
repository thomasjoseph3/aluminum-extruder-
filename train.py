"""
Training script for the extrusion PINN.

Usage
-----
    python train.py

Checkpoints are saved to checkpoints/ every 10 epochs.
Loss curves are saved to loss_curve.png at the end.
"""

import os
import time
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from models.pinn       import ExtrusionPINN, PINNLoss
from utils.preprocessing import load_and_split

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CSV_PATH    = os.path.join('data', 'extrusion_mock_data.csv')
CHECKPOINT_DIR = 'checkpoints'

EPOCHS      = 200
LR          = 1e-3
HIDDEN_DIM  = 128
N_LAYERS    = 6

# Physics weight starts small; we anneal it up after epoch 50
W_DATA_INIT = 1.0
W_PHYS_INIT = 0.01
W_PHYS_FINAL = 0.5   # reached gradually from epoch 50 → 150
W_BC        = 1.0

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_phys_weight(epoch: int) -> float:
    """Linear ramp from W_PHYS_INIT to W_PHYS_FINAL between epochs 50–150."""
    if epoch < 50:
        return W_PHYS_INIT
    if epoch > 150:
        return W_PHYS_FINAL
    t = (epoch - 50) / 100.0
    return W_PHYS_INIT + t * (W_PHYS_FINAL - W_PHYS_INIT)


def denormalise(tensor_norm, scaler, device):
    """Convert normalised tensor back to raw units using scaler."""
    arr = tensor_norm.detach().cpu().numpy()
    arr_raw = scaler.inverse_transform(arr)
    return torch.tensor(arr_raw, dtype=torch.float32).to(device)


def denormalise_inputs(X_norm, x_scaler, device):
    arr = X_norm.detach().cpu().numpy()
    arr_raw = x_scaler.inverse_transform(arr)
    return torch.tensor(arr_raw, dtype=torch.float32).to(device)


def run_epoch(model, loader, loss_fn, x_scaler, y_scaler, optimizer=None):
    """Single train or eval epoch. Returns dict of mean losses."""
    is_train = optimizer is not None
    model.train(is_train)

    totals = {}
    n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for X_norm, y_norm in loader:
            X_norm = X_norm.to(DEVICE)
            y_norm = y_norm.to(DEVICE)

            y_pred_norm = model(X_norm)

            # De-normalise for physics residuals
            X_raw     = denormalise_inputs(X_norm, x_scaler, DEVICE)
            y_pred_raw = denormalise(y_pred_norm, y_scaler, DEVICE)

            loss, components = loss_fn(y_pred_norm, y_norm, X_raw, y_pred_raw)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            for k, v in components.items():
                totals[k] = totals.get(k, 0.0) + v
            n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # --- Data ---
    loaders, scalers, _ = load_and_split(CSV_PATH)
    x_scaler = scalers['X']
    y_scaler = scalers['y']

    # --- Model ---
    model = ExtrusionPINN(
        input_dim  = 13,
        hidden_dim = HIDDEN_DIM,
        n_layers   = N_LAYERS,
        output_dim = 4,
    ).to(DEVICE)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=15, factor=0.5
    )

    loss_fn = PINNLoss(w_data=W_DATA_INIT, w_phys=W_PHYS_INIT, w_bc=W_BC)

    # --- Training loop ---
    history = {'train': [], 'val': []}
    best_val = float('inf')
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        # Anneal physics weight
        loss_fn.w_phys = get_phys_weight(epoch)

        train_losses = run_epoch(model, loaders['train'], loss_fn, x_scaler, y_scaler, optimizer)
        val_losses   = run_epoch(model, loaders['val'],   loss_fn, x_scaler, y_scaler)

        scheduler.step(val_losses['total'])

        history['train'].append(train_losses)
        history['val'].append(val_losses)

        # Checkpoint
        if val_losses['total'] < best_val:
            best_val = val_losses['total']
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'best_model.pt'))

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:3d}/{EPOCHS} | "
                f"train {train_losses['total']:.4f} "
                f"(data {train_losses['data']:.4f} "
                f"phys {train_losses['phys_temp']:.4f}+{train_losses['phys_press']:.4f}) | "
                f"val {val_losses['total']:.4f} | "
                f"w_phys {loss_fn.w_phys:.3f} | "
                f"{elapsed:.0f}s"
            )

        if epoch % 50 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(CHECKPOINT_DIR, f'epoch_{epoch:04d}.pt')
            )

    # --- Save scalers ---
    import pickle
    with open(os.path.join(CHECKPOINT_DIR, 'scalers.pkl'), 'wb') as f:
        pickle.dump({'X': x_scaler, 'y': y_scaler}, f)
    print("Scalers saved to checkpoints/scalers.pkl")

    # --- Final evaluation on test set ---
    test_losses = run_epoch(model, loaders['test'], loss_fn, x_scaler, y_scaler)
    print(f"\nTest loss: {test_losses}")

    # --- Plot loss curves ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    keys = [('total', 'Total loss'), ('data', 'Data loss'), ('phys_temp', 'Physics (temp) residual')]
    for ax, (key, title) in zip(axes, keys):
        ax.plot([d[key] for d in history['train']], label='train')
        ax.plot([d[key] for d in history['val']],   label='val')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=150)
    print("Loss curve saved to loss_curve.png")


if __name__ == '__main__':
    main()
