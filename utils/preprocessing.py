"""
Data loading and normalisation for extrusion PINN.

All variables are scaled to [0, 1] before entering the network.
Normalisation stats are fitted on training data and reused for val/test.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Material property lookup — these become extra input features so the model
# can generalise across alloys without needing one-hot encoding.
ALLOY_FEATURES = {
    'AA6063': [2700, 900,  167, 149_776, 2.41e9, 0.045, 5.385],
    'AA6061': [2700, 896,  152, 152_000, 3.12e9, 0.048, 5.200],
    'AA7075': [2810, 960,  130, 156_000, 4.57e9, 0.050, 4.800],
    #          rho   cp    k    Q_act     A       alpha  n
}

INPUT_COLS = [
    'billet_temperature_C', 'container_temperature_C', 'die_temperature_C',
    'ram_speed_mm_per_sec', 'billet_diameter_mm', 'extrusion_ratio',
    # material features appended dynamically
]

OUTPUT_COLS = [
    'exit_temperature_C',
    'ram_pressure_MPa',
    'exit_speed_mm_per_sec',
    'surface_quality_score',
]


class MinMaxScaler:
    """Fits min/max on training data; transforms any split consistently."""

    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X: np.ndarray):
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        rng = self.max_ - self.min_
        rng[rng == 0] = 1.0          # avoid divide-by-zero for constant cols
        return (X - self.min_) / rng

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        rng = self.max_ - self.min_
        rng[rng == 0] = 1.0
        return X * rng + self.min_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class ExtrusionDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_and_split(
    csv_path: str,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    seed:       int   = 42,
):
    """
    Load CSV, attach material features, split into train/val/test,
    fit scalers on train only, return DataLoaders + scaler objects.

    Returns
    -------
    loaders : dict  {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    scalers : dict  {'X': MinMaxScaler,   'y': MinMaxScaler}
    raw     : dict  {'train': (X,y), 'val': (X,y), 'test': (X,y)}   (unscaled numpy)
    """
    rng = np.random.default_rng(seed)

    df = pd.read_csv(csv_path)
    df = df[df['run_is_valid'] == True].reset_index(drop=True)

    # Attach alloy material features as numeric columns
    mat_cols = ['rho', 'cp', 'k', 'Q_act', 'A', 'alpha', 'n']
    mat_df = df['alloy_grade'].map(ALLOY_FEATURES).apply(pd.Series)
    mat_df.columns = mat_cols
    df = pd.concat([df, mat_df], axis=1)

    feature_cols = INPUT_COLS + mat_cols

    X = df[feature_cols].values.astype(np.float64)
    y = df[OUTPUT_COLS].values.astype(np.float64)

    # Shuffle
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    n = len(X)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    splits = {
        'train': (X[:n_train],           y[:n_train]),
        'val':   (X[n_train:n_train+n_val], y[n_train:n_train+n_val]),
        'test':  (X[n_train+n_val:],     y[n_train+n_val:]),
    }

    # Fit scalers on training data only
    x_scaler = MinMaxScaler().fit(splits['train'][0])
    y_scaler = MinMaxScaler().fit(splits['train'][1])

    loaders = {}
    for split, (Xs, ys) in splits.items():
        Xn = x_scaler.transform(Xs)
        yn = y_scaler.transform(ys)
        ds = ExtrusionDataset(Xn, yn)
        loaders[split] = DataLoader(
            ds,
            batch_size=256,
            shuffle=(split == 'train'),
            num_workers=0,
        )

    return loaders, {'X': x_scaler, 'y': y_scaler}, splits
