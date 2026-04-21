import os, gc, time, warnings
from pathlib import Path
from report_nn import report_nn, report_nn_yearly

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings('ignore')

try:
    from google.colab import drive
    IN_COLAB = True
except Exception:
    IN_COLAB = False

if IN_COLAB:
    drive.mount('/content/drive')

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print('GPU:', torch.cuda.get_device_name(0))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    DEVICE = torch.device('cpu')
    print('Using CPU')

USE_AMP = DEVICE.type == 'cuda'
print('Device:', DEVICE)
print('AMP enabled:', USE_AMP)
print('Torch:', torch.__version__)

# -------- paths --------
PARQUET_PATH = '/content/drive/MyDrive/industry_project/preprocess_data.parquet'
OUTPUT_DIR   = '/content/drive/MyDrive/backtest'

NN_ARCH = 'NN1'  # options: 'NN1', 'NN2', 'NN3', 'NN4', 'NN5'
EXP     = 'exp1' # experiment name — change per run

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------- load data --------
df = pd.read_parquet(PARQUET_PATH)
df['DATE'] = pd.to_datetime(df['DATE'])

print('shape:', df.shape)
print('date range:', df['DATE'].min().date(), 'to', df['DATE'].max().date())
print('permno count:', df['permno'].nunique())
print(f'\nRunning architecture: {NN_ARCH}, exp: {EXP}')
df.head(3)

# -------- column sets --------
NON_CHAR_COLS = {
    'permno', 'DATE', 'ret', 'rf', 'exret', 'exret_lead1', 'sic2',
    'tbl', 'b/m', 'd/p', 'e/p', 'ntis', 'tms', 'dfy', 'svar'
}
MACRO_COLS = ['tbl', 'b/m', 'd/p', 'e/p', 'ntis', 'tms', 'dfy', 'svar']
CHAR_COLS_94 = sorted([c for c in df.columns if c not in NON_CHAR_COLS])

sic2_dummies = pd.get_dummies(df['sic2'].astype(str), prefix='sic2', drop_first=False)
SIC2_DUMMY_COLS = sorted(sic2_dummies.columns.tolist())
df = pd.concat([df, sic2_dummies], axis=1)
del sic2_dummies
gc.collect()

print('94 stock characteristics:', len(CHAR_COLS_94))
print('8 macro variables:', MACRO_COLS)
print('SIC2 dummy cols:', len(SIC2_DUMMY_COLS))
print('total features:', len(CHAR_COLS_94) * 9 + len(SIC2_DUMMY_COLS))

# -------- model config --------
TARGET = 'exret_lead1'
VALIDATION_END = pd.Timestamp('1986-12-31')
TEST_END = pd.Timestamp('2016-12-31')
VALIDATION_YEARS = 12

NN_ARCHITECTURES = {
    'NN1': [32],
    'NN2': [32, 16],
    'NN3': [32, 16, 8],
    'NN4': [32, 16, 8, 4],
    'NN5': [32, 16, 8, 4, 2],
}

HIDDEN_LAYERS = NN_ARCHITECTURES[NN_ARCH]

# HP grid — 20-point L1 grid from 1e-5 to 1e-3
L1_GRID   = list(np.logspace(-5, -3, 20))
LR_GRID   = [0.01, 0.1]
BATCH_SIZE  = 10_000
MAX_EPOCHS  = 100
PATIENCE    = 5
N_ENSEMBLE  = 10
GRAD_CLIP   = 1.0

print(f'L1 grid: {len(L1_GRID)} points from {L1_GRID[0]:.1e} to {L1_GRID[-1]:.1e}')
print(f'LR grid: {LR_GRID}')
print(f'HP combos: {len(L1_GRID)*len(LR_GRID)} x {N_ENSEMBLE} seeds = {len(L1_GRID)*len(LR_GRID)*N_ENSEMBLE} models/year')

# -------- clean data (no winsorization) --------
REQUIRED = CHAR_COLS_94 + MACRO_COLS + [TARGET, 'mvel1']
df_clean = df.dropna(subset=REQUIRED).copy()
df_clean = df_clean.sort_values(['DATE', 'permno']).reset_index(drop=True)

test_years = sorted(
    df_clean.loc[
        (df_clean['DATE'] > VALIDATION_END) & (df_clean['DATE'] <= TEST_END),
        'DATE'
    ].dt.year.unique()
)

print('clean rows:', f'{len(df_clean):,}')
print('test years:', test_years[0], 'to', test_years[-1], f'({len(test_years)})')

# -------- precompute feature matrix --------
print('Precomputing 920-dim feature matrix...')
t0 = time.time()

chars_all = df_clean[CHAR_COLS_94].to_numpy(dtype=np.float32)
macro_with_const_all = np.column_stack([
    np.ones(len(df_clean), dtype=np.float32),
    df_clean[MACRO_COLS].to_numpy(dtype=np.float32)
])
interactions_all = (chars_all[:, :, None] * macro_with_const_all[:, None, :]).reshape(len(df_clean), -1)
sic2_all = df_clean[SIC2_DUMMY_COLS].to_numpy(dtype=np.float32)
X_all = np.hstack([interactions_all, sic2_all])
y_all = df_clean[TARGET].to_numpy(dtype=np.float64)

year_col    = df_clean['DATE'].dt.year.to_numpy()
meta_date   = df_clean['DATE'].to_numpy()
meta_permno = df_clean['permno'].to_numpy()
meta_mvel1  = df_clean['mvel1'].to_numpy(dtype=np.float64)

del chars_all, macro_with_const_all, interactions_all, sic2_all
del df, df_clean
gc.collect()

print(f'X_all shape: {X_all.shape}, dtype: {X_all.dtype}')
print(f'Precompute time: {time.time()-t0:.1f}s')
print(f'Memory: {X_all.nbytes / 1e9:.2f} GB')

# -------- helpers --------

def oos_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.sum(y_true ** 2)
    if denom == 0:
        return np.nan
    return 1.0 - np.sum((y_true - y_pred) ** 2) / denom

# -------- NN model --------

class AssetPricingNN(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)

test_model = AssetPricingNN(920, HIDDEN_LAYERS)
print(f'{NN_ARCH} architecture:')
print(test_model)
print(f'Total parameters: {sum(p.numel() for p in test_model.parameters()):,}')
del test_model

# -------- training function --------

def train_single_nn(X_train, y_train, X_val, y_val,
                    hidden_layers, l1_lambda, lr,
                    batch_size, max_epochs, patience, seed, device,
                    grad_clip=1.0, use_amp=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)

    input_dim = X_train.shape[1]
    n_train = X_train.shape[0]
    model = AssetPricingNN(input_dim, hidden_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler(enabled=use_amp)

    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)

        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            X_batch = X_train[idx]
            y_batch = y_train[idx]

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                y_pred = model(X_batch)
                mse_loss = nn.functional.mse_loss(y_pred, y_batch)

                l1_loss = torch.tensor(0.0, device=device)
                for name, param in model.named_parameters():
                    if 'weight' in name and 'bn' not in name.lower() and 'norm' not in name.lower():
                        l1_loss = l1_loss + param.abs().sum()

                loss = mse_loss + l1_lambda * l1_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        with torch.no_grad():
            with autocast(enabled=use_amp):
                val_pred = model(X_val)
            val_loss = nn.functional.mse_loss(val_pred.float(), y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {'best_epoch': best_epoch, 'best_val_loss': best_val_loss,
                   'total_epochs': epoch + 1}


def predict_ensemble(models, X, device, use_amp=False):
    preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            with autocast(enabled=use_amp):
                pred = model(X).float().cpu().numpy()
            preds.append(pred)
    return np.mean(preds, axis=0)

print('training functions ready')

# ======================================================================
# main loop
# ======================================================================

def main(exp):
    save_dir = f"{OUTPUT_DIR}/{NN_ARCH}/{exp}/"
    os.makedirs(save_dir, exist_ok=True)

    # detect completed years
    done_years = set()
    for fname in os.listdir(save_dir):
        if fname.endswith('_pred.parquet'):
            try:
                done_years.add(int(fname.split('_')[0]))
            except ValueError:
                pass

    if done_years:
        print(f"Checkpoint: {len(done_years)} years already done: {sorted(done_years)}")
    else:
        print("Checkpoint: starting fresh")

    all_predictions = []
    yearly_records  = []

    for year in test_years:

        # --- checkpoint: load if done ---
        pred_path = f"{save_dir}/{year}_pred.parquet"
        info_path = f"{save_dir}/{year}_info.parquet"

        if year in done_years:
            print(f"Skipping {year}, already done.")
            pred_df = pd.read_parquet(pred_path)
            all_predictions.append(pred_df)
            if os.path.exists(info_path):
                info_row = pd.read_parquet(info_path).iloc[0].to_dict()
                yearly_records.append(info_row)
            continue

        t0 = time.time()

        # --- sample split ---
        train_end_year = year - VALIDATION_YEARS - 1
        val_start_year = year - VALIDATION_YEARS
        val_end_year   = year - 1

        mask_train = year_col <= train_end_year
        mask_val   = (year_col >= val_start_year) & (year_col <= val_end_year)
        mask_test  = year_col == year

        X_train_np = X_all[mask_train].copy()
        X_val_np   = X_all[mask_val].copy()
        X_test_np  = X_all[mask_test].copy()
        y_train_np = y_all[mask_train]
        y_val_np   = y_all[mask_val]
        y_test_np  = y_all[mask_test]

        # --- demean target (no standardization on X) ---
        y_mu   = float(y_train_np.mean())
        y_tr_c = (y_train_np - y_mu).astype(np.float32)
        y_va_c = (y_val_np   - y_mu).astype(np.float32)

        # --- move to GPU ---
        X_tr = torch.tensor(X_train_np, dtype=torch.float32, device=DEVICE); del X_train_np; gc.collect()
        X_va = torch.tensor(X_val_np,   dtype=torch.float32, device=DEVICE); del X_val_np;   gc.collect()
        X_te = torch.tensor(X_test_np,  dtype=torch.float32, device=DEVICE); del X_test_np;  gc.collect()
        y_tr = torch.tensor(y_tr_c,     dtype=torch.float32, device=DEVICE)
        y_va = torch.tensor(y_va_c,     dtype=torch.float32, device=DEVICE)
        del y_tr_c, y_va_c; gc.collect()

        # --- HP search ---
        best_hp_mse    = float('inf')
        best_l1        = L1_GRID[0]
        best_lr        = LR_GRID[0]
        best_models    = None
        best_epochs_list = None

        for l1_lambda in L1_GRID:
            for lr in LR_GRID:
                hp_models = []
                hp_epochs = []
                for seed_idx in range(N_ENSEMBLE):
                    seed = seed_idx * 1000 + year
                    model, hist = train_single_nn(
                        X_tr, y_tr, X_va, y_va,
                        hidden_layers=HIDDEN_LAYERS,
                        l1_lambda=l1_lambda, lr=lr,
                        batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS,
                        patience=PATIENCE, seed=seed, device=DEVICE,
                        grad_clip=GRAD_CLIP, use_amp=USE_AMP
                    )
                    hp_models.append(model)
                    hp_epochs.append(hist['best_epoch'])

                val_pred   = predict_ensemble(hp_models, X_va, DEVICE, USE_AMP)
                val_pred_t = torch.tensor(val_pred, dtype=torch.float32, device=DEVICE)
                hp_mse     = nn.functional.mse_loss(val_pred_t, y_va).item()

                if hp_mse < best_hp_mse:
                    best_hp_mse = hp_mse
                    best_l1     = l1_lambda
                    best_lr     = lr
                    if best_models is not None:
                        del best_models
                    best_models      = hp_models
                    best_epochs_list = hp_epochs
                else:
                    del hp_models

                del val_pred, val_pred_t
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        best_avg_epochs = float(np.mean(best_epochs_list))

        # --- predict on test ---
        y_pred_ensemble = predict_ensemble(best_models, X_te, DEVICE, USE_AMP)
        y_pred = y_pred_ensemble.astype(np.float64) + y_mu

        elapsed = time.time() - t0

        # --- build result df ---
        res = pd.DataFrame({
            'DATE':   meta_date[mask_test],
            'permno': meta_permno[mask_test],
            'mvel1':  meta_mvel1[mask_test],
            'y_true': y_test_np,
            'y_pred': y_pred,
        })

        test_r2 = oos_r2(y_test_np, y_pred)

        info = {
            'year':           year,
            'n_train':        int(mask_train.sum()),
            'n_val':          int(mask_val.sum()),
            'n_test':         int(mask_test.sum()),
            'best_l1':        best_l1,
            'best_lr':        best_lr,
            'best_val_mse':   best_hp_mse,
            'avg_best_epoch': best_avg_epochs,
            'test_r2':        test_r2,
            'sec':            elapsed,
        }

        # --- checkpoint save ---
        res.to_parquet(pred_path, index=False)
        pd.DataFrame([info]).to_parquet(info_path, index=False)

        all_predictions.append(res)
        yearly_records.append(info)

        print(
            f'Year {year} | train {int(mask_train.sum()):>8,} | val {int(mask_val.sum()):>8,} | '
            f'test {int(mask_test.sum()):>7,} | '
            f'l1={best_l1:.1e} lr={best_lr} '
            f'avg_epoch={best_avg_epochs:.0f} '
            f'r2={test_r2*100:+.4f}% | {elapsed:.1f}s [saved]'
        )

        del X_tr, X_va, X_te, y_tr, y_va, best_models, res, y_pred, y_pred_ensemble
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- aggregate & report ---
    if all_predictions:
        results = pd.concat(all_predictions, ignore_index=True)
        results['DATE'] = pd.to_datetime(results['DATE'])
        results = results.sort_values(['DATE', 'permno']).reset_index(drop=True)

        # save aggregated outputs
        out_pred_path = f"{save_dir}/all_predictions.parquet"
        out_info_path = f"{save_dir}/year_info.csv"
        results[['permno', 'DATE', 'y_true', 'y_pred', 'mvel1']].to_parquet(out_pred_path, index=False)
        pd.DataFrame(yearly_records).to_csv(out_info_path, index=False)
        print(f'Saved: {out_pred_path}')
        print(f'Saved: {out_info_path}')

        # report
        fig_path = f"{save_dir}/figure_{NN_ARCH.lower()}_{exp}.png"
        report_nn(results, arch=NN_ARCH, save_path=fig_path)
        report_nn_yearly(yearly_records, arch=NN_ARCH, save_path=fig_path)
    else:
        print("No predictions collected.")


# ======================================================================
# run
# ======================================================================
main(EXP)
