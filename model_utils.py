import numpy as np
import torch


# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_features_920(data, char_cols, macro_cols, sic2_dummies_cols):
    """
    Build the 920-dimensional feature matrix:
    [characteristics × (1 + macro variables)] + SIC2 dummies.

    Parameters
    ----------
    data : pd.DataFrame
    char_cols : list[str]
    macro_cols : list[str]
    sic2_dummies_cols : list[str]

    Returns
    -------
    X : np.ndarray of shape (N, P), dtype float32
    """
    chars = data[char_cols].to_numpy(dtype=np.float32)

    macro_with_const = np.column_stack([
        np.ones(len(data), dtype=np.float32),
        data[macro_cols].to_numpy(dtype=np.float32)
    ])

    interactions = (
        chars[:, :, None] * macro_with_const[:, None, :]
    ).reshape(len(data), -1)

    sic2_vals = data[sic2_dummies_cols].to_numpy(dtype=np.float32)

    return np.hstack([interactions, sic2_vals])


def huber_loss_t(residuals, xi):
    """
    Elementwise Huber loss for torch tensors.
    """
    abs_r = residuals.abs()
    return torch.where(abs_r <= xi, residuals**2, 2 * xi * abs_r - xi**2)


def huber_grad_t(X, y, theta, xi):
    """
    Gradient of mean Huber loss:
        (1/N) * sum H(y - X theta; xi)
    """
    n = y.shape[0]
    residuals = y - X @ theta
    mask = residuals.abs() <= xi
    dH = torch.where(mask, 2.0 * residuals, 2.0 * xi * residuals.sign())
    return -(1.0 / n) * (X.T @ dH)


def prox_enet_t(theta, gamma, lam, rho):
    """
    Proximal operator for elastic net penalty:
        lam * [(1-rho)*|theta| + (rho/2)*||theta||^2]
    """
    tau = gamma * lam * (1.0 - rho)
    shrunk = theta.sign() * torch.clamp(theta.abs() - tau, min=0.0)
    return shrunk / (1.0 + gamma * lam * rho)


def enet_objective_t(X, y, theta, lam, rho, xi):
    """
    Full objective:
        mean Huber loss + elastic-net penalty
    """
    n = y.shape[0]
    loss = huber_loss_t(y - X @ theta, xi).sum() / n
    l1 = (1.0 - rho) * theta.abs().sum()
    l2 = 0.5 * rho * (theta ** 2).sum()
    return (loss + lam * (l1 + l2)).item()


def apg_huber_enet_gpu(
    X,
    y,
    lam,
    rho,
    xi,
    gamma,
    max_iter=2000,
    tol=1e-5,
    theta_init=None,
    return_info=False,
):
    """
    Accelerated proximal gradient for Huber + elastic net, with adaptive restart.

    Parameters
    ----------
    X : torch.Tensor, shape (N, P)
    y : torch.Tensor, shape (N,)
    lam : float
    rho : float
    xi : float
    gamma : float
        Step size.
    max_iter : int
    tol : float
    theta_init : torch.Tensor or None
    return_info : bool

    Returns
    -------
    theta : torch.Tensor
    info : dict, optional
    """
    p = X.shape[1]

    theta = theta_init.clone() if theta_init is not None else torch.zeros(
        p, dtype=torch.float32, device=X.device
    )
    theta_old = theta.clone()

    converged = False
    final_diff = float("inf")
    iters_used = 0
    n_restarts = 0
    momentum_age = 0

    prev_obj = enet_objective_t(X, y, theta, lam, rho, xi)

    for m in range(max_iter):
        grad = huber_grad_t(X, y, theta, xi)
        theta_tilde = theta - gamma * grad
        theta_bar = prox_enet_t(theta_tilde, gamma, lam, rho)

        beta = momentum_age / (momentum_age + 3.0)
        theta_new = theta_bar + beta * (theta_bar - theta_old)

        curr_obj = enet_objective_t(X, y, theta_new, lam, rho, xi)

        if curr_obj > prev_obj:
            theta_new = theta_bar
            curr_obj = enet_objective_t(X, y, theta_new, lam, rho, xi)
            momentum_age = 0
            n_restarts += 1
        else:
            momentum_age += 1

        prev_obj = curr_obj

        diff = (theta_new - theta).norm().item()
        final_diff = diff
        iters_used = m + 1

        if diff < tol * (1.0 + theta.norm().item()):
            converged = True
            theta = theta_new
            break

        theta_old = theta.clone()
        theta = theta_new

    if return_info:
        return theta, {
            "iters": iters_used,
            "converged": converged,
            "final_diff": final_diff,
            "n_restarts": n_restarts,
        }

    return theta


def standardize(X, mean=None, std=None):
    """
    Standardize columns of X using provided or estimated mean/std.
    """
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)

    std = np.where(std < 1e-8, 1.0, std)
    return (X - mean) / std, mean, std


def oos_r2(y_true, y_pred, benchmark_var=None):
    """
    Out-of-sample R^2.

    If benchmark_var is None, denominator is sum(y_true^2).
    Otherwise denominator is N * benchmark_var.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum(y_true ** 2) if benchmark_var is None else len(y_true) * benchmark_var

    if ss_tot == 0:
        return np.nan

    return 1.0 - ss_res / ss_tot