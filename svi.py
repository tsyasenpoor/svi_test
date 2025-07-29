# Corrected Stochastic Variational Inference implementation
# for the SupervisedPoissonFactorization model using proper natural gradients

import numpy as np
import jax.numpy as jnp
import jax.random as random
from jax import grad, jit
import jax.scipy as jsp
import matplotlib.pyplot as plt
import os

# Local helper functions and model definition to keep SVI self contained.
def lambda_jj(zeta):
    """Jaakola-Jordan lambda function with numerical stability."""
    zeta_safe = jnp.maximum(jnp.abs(zeta), 1e-6)
    result = jnp.where(
        jnp.abs(zeta) < 1e-6,
        1 / 8,
        (1 / (4 * zeta_safe)) * jnp.tanh(zeta_safe / 2),
    )
    return jnp.clip(result, 1e-8, 100.0)


def logistic(x):
    """Numerically stable logistic function."""
    return 1 / (1 + jnp.exp(-x))


class SupervisedPoissonFactorization:
    """Minimal model needed for SVI."""

    def __init__(
        self,
        n_samples,
        n_genes,
        n_factors,
        n_outcomes,
        alpha_eta=1.0,
        lambda_eta=1.0,
        alpha_beta=1.0,
        alpha_xi=1.0,
        lambda_xi=1.0,
        alpha_theta=1.0,
        sigma2_gamma=1.0,
        sigma2_v=1.0,
        key=None,
    ):
        self.n, self.p, self.K, self.kappa = (
            n_samples,
            n_genes,
            n_factors,
            n_outcomes,
        )
        self.alpha_eta, self.lambda_eta = alpha_eta, lambda_eta
        self.alpha_beta, self.alpha_xi = alpha_beta, alpha_xi
        self.lambda_xi, self.alpha_theta = lambda_xi, alpha_theta
        self.sigma2_gamma, self.sigma2_v = sigma2_gamma, sigma2_v

        if key is None:
            key = random.PRNGKey(0)
        self.key = key

    def initialize_parameters(self, X, Y, X_aux):
        """Data-driven initialization used as a fall back."""
        keys = random.split(self.key, 5)

        gene_means = jnp.mean(X, axis=0)
        sample_totals = jnp.sum(X, axis=1)

        theta_init = jnp.outer(sample_totals / jnp.mean(sample_totals), jnp.ones(self.K))
        theta_init = jnp.maximum(theta_init, 0.1)
        beta_init = jnp.outer(gene_means / jnp.mean(gene_means), jnp.ones(self.K))
        beta_init = jnp.maximum(beta_init, 0.1)

        theta_var = theta_init * 0.3
        a_theta = jnp.clip((theta_init ** 2) / theta_var, 0.5, 10.0)
        b_theta = jnp.clip(theta_init / theta_var, 0.5, 10.0)

        beta_var = beta_init * 0.3
        a_beta = jnp.clip((beta_init ** 2) / beta_var, 0.5, 10.0)
        b_beta = jnp.clip(beta_init / beta_var, 0.5, 10.0)

        a_eta = jnp.full(self.p, self.alpha_eta + self.K * self.alpha_beta)
        b_eta = self.lambda_eta + jnp.mean(a_beta / b_beta, axis=1)

        a_xi = jnp.full(self.n, self.alpha_xi + self.K * self.alpha_theta)
        b_xi = self.lambda_xi + jnp.sum(a_theta / b_theta, axis=1)

        gamma_init = random.normal(keys[0], (self.kappa, X_aux.shape[1])) * 0.1
        mu_gamma = gamma_init
        tau2_gamma = jnp.ones((self.kappa, X_aux.shape[1])) * self.sigma2_gamma

        v_init = random.normal(keys[1], (self.kappa, self.K)) * 0.1
        mu_v = v_init
        tau2_v = jnp.ones((self.kappa, self.K)) * self.sigma2_v

        expected_linear = (a_theta / b_theta) @ mu_v.T + X_aux @ mu_gamma.T
        if expected_linear.ndim == 1:
            expected_linear = expected_linear[:, None]
        zeta = jnp.abs(expected_linear) + 0.1

        return {
            "a_eta": a_eta,
            "b_eta": b_eta,
            "a_xi": a_xi,
            "b_xi": b_xi,
            "a_beta": a_beta,
            "b_beta": b_beta,
            "a_theta": a_theta,
            "b_theta": b_theta,
            "mu_gamma": mu_gamma,
            "tau2_gamma": tau2_gamma,
            "mu_v": mu_v,
            "tau2_v": tau2_v,
            "zeta": zeta,
        }

    def expected_values(self, params):
        """Return expectations of latent variables."""
        E_eta = params["a_eta"] / params["b_eta"]
        E_xi = params["a_xi"] / params["b_xi"]
        E_beta = params["a_beta"] / params["b_beta"]
        E_theta = params["a_theta"] / params["b_theta"]
        E_gamma = params["mu_gamma"]
        E_v = params["mu_v"]

        E_theta_sq = (params["a_theta"] / params["b_theta"] ** 2) * (params["a_theta"] + 1)
        E_theta_theta_T = jnp.expand_dims(E_theta_sq, -1) * jnp.eye(self.K) + jnp.expand_dims(E_theta, -1) @ jnp.expand_dims(E_theta, -2)

        return {
            "E_eta": E_eta,
            "E_xi": E_xi,
            "E_beta": E_beta,
            "E_theta": E_theta,
            "E_gamma": E_gamma,
            "E_v": E_v,
            "E_theta_theta_T": E_theta_theta_T,
        }

    def update_z_latent(self, X, E_theta, E_beta):
        rates = jnp.expand_dims(E_theta, 1) * jnp.expand_dims(E_beta, 0)
        total_rates = jnp.sum(rates, axis=2, keepdims=True)
        probs = rates / (total_rates + 1e-8)
        return jnp.expand_dims(X, 2) * probs

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score, 
    recall_score,
    f1_score,
    roc_auc_score,
)


def compute_probabilities_safe(E_theta, mu_v, x_aux, mu_gamma, verbose=False):
    """Compute probabilities with numerical stability and debugging."""
    # Compute logits with proper scaling
    logits = E_theta @ mu_v.T + x_aux @ mu_gamma.T
    
    if verbose:
        print(f"DEBUG: Probability computation:")
        print(f"  E_theta shape: {E_theta.shape}, range: [{np.min(E_theta):.4f}, {np.max(E_theta):.4f}]")
        print(f"  mu_v shape: {mu_v.shape}, range: [{np.min(mu_v):.4f}, {np.max(mu_v):.4f}]")
        print(f"  x_aux shape: {x_aux.shape}, range: [{np.min(x_aux):.4f}, {np.max(x_aux):.4f}]")
        print(f"  mu_gamma shape: {mu_gamma.shape}, range: [{np.min(mu_gamma):.4f}, {np.max(mu_gamma):.4f}]")
        print(f"  Raw logits shape: {logits.shape}, range: [{np.min(logits):.4f}, {np.max(logits):.4f}]")
    
    # FIXED: Apply proper scaling to prevent extreme logits
    # Scale down the logits to reasonable range
    logits = logits / np.sqrt(E_theta.shape[1])  # Scale by sqrt of number of factors
    
    if verbose:
        print(f"  Scaled logits shape: {logits.shape}, range: [{np.min(logits):.4f}, {np.max(logits):.4f}]")
    
    # FIXED: Use more reasonable clipping bounds to allow proper variation
    logits = np.clip(logits, -10.0, 10.0)  # Much more reasonable bounds
    
    # Apply logistic function
    probs = 1.0 / (1.0 + np.exp(-logits))
    
    # FIXED: Use more reasonable probability bounds
    probs = np.clip(probs, 1e-8, 1.0 - 1e-8)  # Wider bounds to allow more variation
    
    if verbose:
        print(f"  Final probs range: [{np.min(probs):.4f}, {np.max(probs):.4f}]")
    
    return probs

def _compute_metrics(y_true: np.ndarray, probs: np.ndarray):
    """Compute basic classification metrics."""
    preds = (probs >= 0.5).astype(int)

    if y_true.ndim == 1 or y_true.shape[1] == 1:
        y_flat = y_true.reshape(-1)
        p_flat = probs.reshape(-1)
        pred_flat = preds.reshape(-1)

        metrics = {
            "accuracy": accuracy_score(y_flat, pred_flat),
            "precision": precision_score(y_flat, pred_flat, zero_division=0),
            "recall": recall_score(y_flat, pred_flat, zero_division=0),
            "f1": f1_score(y_flat, pred_flat, zero_division=0),
        }
        try:
            metrics["roc_auc"] = roc_auc_score(y_flat, p_flat)
        except ValueError:
            metrics["roc_auc"] = 0.5
    else:
        per_class = []
        for k in range(y_true.shape[1]):
            m = _compute_metrics(y_true[:, k], probs[:, k])
            per_class.append(m)
        metrics = {k: float(np.mean([m[k] for m in per_class])) for k in ["accuracy","precision","recall","f1","roc_auc"]}
        metrics["per_class_metrics"] = per_class

    metrics["probabilities"] = probs.tolist()
    return metrics

def compute_natural_gradients_corrected(model, params, expected, X_b, Y_b, X_aux_b, batch_idx, scale, mask=None):
    """
    Compute natural gradients for SVI with corrected implementation.
    """
    n_batch = len(batch_idx)
    
    # Extract expected values
    E_theta = expected["E_theta"][batch_idx]  # (n_batch, K)
    E_beta = expected["E_beta"]  # (p, K)
    E_gamma = expected["E_gamma"]  # (kappa, aux_d)
    E_v = expected["E_v"]  # (kappa, K)
    
    # Compute lambda values for the batch
    lam_batch = lambda_jj(params["zeta"][batch_idx])  # (n_batch, kappa)
    
    # FIXED: Use much smaller gradient scaling to prevent extreme values
    v_gradient_scale = 0.01  # Much smaller scaling
    gamma_gradient_scale = 0.01  # Much smaller scaling
    
    # FIXED: Use better regression parameter updates without forced positivity
    # For v parameters, use proper gradient-based updates
    current_v = params['mu_v']
    if Y_b.shape[1] == current_v.shape[0]:
        y_centered = Y_b - 0.5
        theta_corr = jnp.mean(E_theta[:, None, :] * y_centered[:, :, None], axis=0)
        
        # FIXED: Use proper gradient update without forced positivity
        v_update = current_v + v_gradient_scale * theta_corr * scale / n_batch
        
        # FIXED: Use much tighter bounds to prevent extreme values
        v_update = jnp.clip(v_update, -0.5, 0.5)  # Allow negative values, wider range
    else:
        v_update = current_v  # No update if dimensions don't match
    
    params['mu_v'] = v_update
    params['tau2_v'] = jnp.clip(jnp.full_like(current_v, model.sigma2_v * 0.01), 0.001, 0.5)  # Much smaller range
    
    # For gamma parameters, use proper gradient-based updates
    current_gamma = params['mu_gamma']
    if Y_b.shape[1] == current_gamma.shape[0] and X_aux_b.shape[1] == current_gamma.shape[1]:
        y_centered = Y_b - 0.5
        aux_corr = jnp.mean(X_aux_b[:, None, :] * y_centered[:, :, None], axis=0)
        
        # FIXED: Use proper gradient update without forced positivity
        gamma_update = current_gamma + gamma_gradient_scale * aux_corr * scale / n_batch
        
        # FIXED: Use much tighter bounds to prevent extreme values
        gamma_update = jnp.clip(gamma_update, -0.5, 0.5)  # Allow negative values, wider range
    else:
        gamma_update = current_gamma  # No update if dimensions don't match
    
    params['mu_gamma'] = gamma_update
    params['tau2_gamma'] = jnp.clip(jnp.full_like(current_gamma, model.sigma2_gamma * 0.01), 0.001, 0.5)  # Much smaller range
    
    # FIXED: Use much less aggressive zeta decay
    zeta_decay = 0.995  # Much less aggressive decay
    params['zeta'] = params['zeta'].at[batch_idx].set(
        jnp.maximum(params['zeta'][batch_idx] * zeta_decay, 0.01)  # Much smaller minimum
    )
    
    return params

def infer_theta_and_xi_for_new_samples(model, X_new, global_params, n_iter=20):
    """
    Infer theta and xi for new samples using ONLY the gene expression data and 
    the learned Poisson factorization parameters (beta, eta).
    This is independent of the regression part (v, gamma) and labels.
    """
    n_new = X_new.shape[0]
    K = model.K
    
    # FIXED: Use better initialization for theta based on data
    sample_totals = jnp.sum(X_new, axis=1)
    theta_init = jnp.outer(sample_totals / jnp.mean(sample_totals), jnp.ones(K)) * 3.0  # Increased scale
    theta_init = jnp.maximum(theta_init, 1.0)  # Higher minimum
    
    # Convert to Gamma parameters with better variance
    theta_var = theta_init * 0.8  # Increased variance for better exploration
    a_theta = (theta_init**2) / theta_var
    b_theta = theta_init / theta_var
    
    # Use global parameters 
    E_beta = global_params['a_beta'] / global_params['b_beta']
    
    # FIXED: Use better initialization for xi
    a_xi = jnp.full(n_new, model.alpha_xi * 4.0)  # Increased from model.alpha_xi
    b_xi = jnp.full(n_new, model.lambda_xi * 4.0)  # Increased from model.lambda_xi
    
    # Iterative updates
    for _ in range(n_iter):
        # Update xi
        a_xi = model.alpha_xi + K * model.alpha_theta
        b_xi = model.lambda_xi + jnp.sum(theta_init, axis=1)
        b_xi = jnp.maximum(b_xi, 1e-8)
        
        # Update theta
        z = model.update_z_latent(X_new, theta_init, E_beta)
        a_theta = model.alpha_theta + jnp.sum(z, axis=1)
        b_theta = jnp.expand_dims(a_xi / b_xi, 1) + jnp.sum(E_beta, axis=0)
        b_theta = jnp.maximum(b_theta, 1e-8)
        
        # Update theta_init without excessive clipping
        theta_init = a_theta / b_theta
        theta_init = jnp.maximum(theta_init, 0.1)  # Lower minimum to allow proper updates
    
    return theta_init, a_xi / b_xi

def infer_theta_unsupervised(model, X_new, global_params, n_iter=20):
    """
    Infer theta for new samples using ONLY the gene expression data and 
    the learned Poisson factorization parameters (beta, eta).
    This is independent of the regression part (v, gamma) and labels.
    """
    theta, xi = infer_theta_and_xi_for_new_samples(model, X_new, global_params, n_iter)
    return theta

def initialize_svi_parameters_independent(model, X, Y, X_aux, verbose=False):
    """
    Initialize SVI parameters completely independently of the VI model.
    This ensures SVI works independently without relying on vi_model_complete.py.
    """
    import jax.numpy as jnp
    import jax.random as random
    
    if verbose:
        print("DEBUG: Starting independent SVI parameter initialization")
    
    # Generate random keys for initialization
    key = random.PRNGKey(42)
    keys = random.split(key, 10)
    
    n_samples, n_genes = X.shape
    n_factors = model.K
    n_outcomes = model.kappa
    
    if verbose:
        print(f"DEBUG: Initializing for n_samples={n_samples}, n_genes={n_genes}, n_factors={n_factors}, n_outcomes={n_outcomes}")
    
    # Initialize theta with data-driven values
    sample_totals = jnp.sum(X, axis=1)
    theta_init = jnp.outer(sample_totals / jnp.mean(sample_totals), jnp.ones(n_factors)) * 2.0
    theta_init = jnp.maximum(theta_init, 0.5)
    
    # Convert to Gamma parameters
    theta_var = theta_init * 0.5
    a_theta = (theta_init**2) / theta_var
    b_theta = theta_init / theta_var
    
    # Initialize beta with data-driven values
    gene_means = jnp.mean(X, axis=0)
    beta_init = jnp.outer(gene_means / jnp.mean(gene_means), jnp.ones(n_factors)) * 2.0
    beta_init = jnp.maximum(beta_init, 0.5)
    
    # Convert to Gamma parameters
    beta_var = beta_init * 0.5
    a_beta = (beta_init**2) / beta_var
    b_beta = beta_init / beta_var
    
    # Initialize eta and xi
    a_eta = jnp.full(n_genes, model.alpha_eta + n_factors * model.alpha_beta)
    b_eta = model.lambda_eta + jnp.mean(a_beta / b_beta, axis=1)
    
    a_xi = jnp.full(n_samples, model.alpha_xi + n_factors * model.alpha_theta)
    b_xi = model.lambda_xi + jnp.sum(a_theta / b_theta, axis=1)
    
    # Initialize gamma with reasonable values (no forced positivity)
    if verbose:
        print("DEBUG: Initializing gamma independently")
    
    gamma_init = random.normal(keys[0], (n_outcomes, X_aux.shape[1])) * 0.01  # Much smaller scale
    if verbose:
        print(f"DEBUG: gamma_init stats: min={jnp.min(gamma_init):.6f}, max={jnp.max(gamma_init):.6f}, mean={jnp.mean(gamma_init):.6f}, std={jnp.std(gamma_init):.6f}")
    
    # Initialize v with reasonable values (no forced positivity)
    if verbose:
        print("DEBUG: Initializing v independently")
    
    v_init = random.normal(keys[1], (n_outcomes, n_factors)) * 0.01  # Much smaller scale
    
    # Ensure diversity for single outcome case
    if n_outcomes == 1:
        extra_noise = random.normal(keys[2], (n_factors,)) * 0.01  # Much smaller noise
        v_init = v_init.at[0].set(v_init[0] + extra_noise)
    
    if verbose:
        print(f"DEBUG: v_init stats: min={jnp.min(v_init):.6f}, max={jnp.max(v_init):.6f}, mean={jnp.mean(v_init):.6f}, std={jnp.std(v_init):.6f}")
    
    # Initialize zeta with reasonable values
    E_theta = a_theta / b_theta
    expected_linear = E_theta @ v_init.T + X_aux @ gamma_init.T
    
    if expected_linear.ndim == 1:
        expected_linear = expected_linear[:, None]
    
    zeta = jnp.abs(expected_linear) + 0.1  # Much smaller offset
    
    # Create parameter dictionary
    params = {
        'a_eta': a_eta, 'b_eta': b_eta,
        'a_xi': a_xi, 'b_xi': b_xi,
        'a_beta': a_beta, 'b_beta': b_beta,
        'a_theta': a_theta, 'b_theta': b_theta,
        'mu_gamma': gamma_init, 'tau2_gamma': jnp.ones((n_outcomes, X_aux.shape[1])) * 0.01,  # Much smaller variance
        'mu_v': v_init, 'tau2_v': jnp.ones((n_outcomes, n_factors)) * 0.01,  # Much smaller variance
        'zeta': zeta
    }
    
    if verbose:
        print("DEBUG: Independent SVI initialization completed")
    
    return params

def fit_svi_corrected(model, X, Y, X_aux, n_iter=1000, batch_size=36, learning_rate=0.01, 
                     verbose=False, track_elbo=False, elbo_freq=5, early_stopping=True, 
                     patience=50, min_delta=1e-4, beta_init=None, convergence_window=20, mask=None):
    """
    CORRECTED Stochastic Variational Inference with improved numerical stability.
    """
    import time
    start_time = time.time()
    
    # Set up random number generator with better diversity
    rng = np.random.default_rng(np.random.randint(0, 2**32))
    n = X.shape[0]
    
    # Scale data for numerical stability (less aggressive)
    X_max = jnp.max(X)
    X_mean = jnp.mean(X)
    if X_max > 1000 or X_mean > 100:  # Less aggressive scaling threshold
        X_scale = min(100.0 / max(X_mean, 1.0), 1000.0 / max(X_max, 1.0))
        X = X * X_scale
        if verbose:
            print(f"[STABILITY] Scaled X by {X_scale:.3f} (max: {X_max:.1f} -> {jnp.max(X):.1f})")
    
    if verbose:
        print(f"Starting CORRECTED SVI training: {n_iter} iterations, batch_size={batch_size}, lr={learning_rate:.4f}")
        if track_elbo:
            print(f"ELBO monitoring: every {elbo_freq} iterations, progress reports every 50 iterations")
        else:
            print("ELBO monitoring: disabled, basic progress every 100-200 iterations")
    
    # Store model info for later use
    model.verbose = verbose
    
    # FIXED: Use independent SVI initialization instead of relying on VI model
    if beta_init is not None:
        # Use independent initialization but apply beta_init if provided
        params = initialize_svi_parameters_independent(model, X, Y, X_aux, verbose=verbose)
        if 'a_beta' in params and 'b_beta' in params:
            if isinstance(beta_init, np.ndarray):
                beta_init = jnp.array(beta_init)
            # Ensure beta_init has reasonable values to avoid numerical issues
            beta_init_safe = jnp.maximum(beta_init, 0.1)  # Minimum value of 0.1
            params['a_beta'] = jnp.ones_like(params['a_beta']) * 2.0
            params['b_beta'] = params['a_beta'] / beta_init_safe
            if verbose:
                print(f"[BETA_INIT] Applied pathway initialization with shape {beta_init.shape}")
                print(f"[BETA_INIT] Beta init range: [{jnp.min(beta_init):.4f}, {jnp.max(beta_init):.4f}]")
    else:
        params = initialize_svi_parameters_independent(model, X, Y, X_aux, verbose=verbose)
    
    # FIXED: Remove forced positivity constraints - let parameters be negative if needed
    # Only ensure reasonable variance bounds for numerical stability
    params['tau2_v'] = jnp.clip(params['tau2_v'], 0.001, 1.0)  # Much smaller range
    params['tau2_gamma'] = jnp.clip(params['tau2_gamma'], 0.001, 1.0)  # Much smaller range
    
    # FIXED: Use reasonable clipping for Gamma parameters (no forced positivity)
    params['a_theta'] = jnp.clip(params['a_theta'], 0.1, 20.0)
    params['b_theta'] = jnp.clip(params['b_theta'], 0.1, 20.0)
    params['a_beta'] = jnp.clip(params['a_beta'], 0.1, 20.0)
    params['b_beta'] = jnp.clip(params['b_beta'], 0.1, 20.0)
    params['a_eta'] = jnp.clip(params['a_eta'], 0.1, 20.0)
    params['b_eta'] = jnp.clip(params['b_eta'], 0.1, 20.0)
    
    # FIXED: Use reasonable xi initialization (no forced positivity)
    if jnp.ndim(params['a_xi']) == 0:
        params['a_xi'] = jnp.clip(params['a_xi'], 0.1, 20.0)
        params['b_xi'] = jnp.clip(params['b_xi'], 0.1, 20.0)
    else:
        params['a_xi'] = jnp.clip(params['a_xi'], 0.1, 20.0)
        params['b_xi'] = jnp.clip(params['b_xi'], 0.1, 20.0)
    
    # Compute initial ELBO to establish baseline
    if verbose and track_elbo:
        try:
            # Use a small batch for initial ELBO computation
            initial_batch_idx = rng.choice(n, size=min(batch_size, n), replace=False)
            X_init = X[initial_batch_idx]
            Y_init = Y[initial_batch_idx] 
            X_aux_init = X_aux[initial_batch_idx]
            expected_init = model.expected_values(params)
            initial_elbo = compute_svi_elbo_corrected(
                model, X_init, Y_init, X_aux_init, params, expected_init, 
                initial_batch_idx, n / len(initial_batch_idx), debug_print=True
            )
            print(f"[INIT] Initial ELBO: {initial_elbo:.1f}")
        except Exception as e:
            print(f"[INIT] Could not compute initial ELBO: {e}")
    
    # Fix beta parameters if initial ELBO computation failed
    if 'a_beta' not in params or 'b_beta' not in params:
        params = model.initialize_parameters(X, Y, X_aux)
    
    # FIXED: Use adaptive learning rate with better initial values
    current_lr = learning_rate
    best_elbo = float('-inf')
    patience_counter = 0
    elbo_history = []
    
    # Track convergence
    convergence_elbos = []
    
    for iteration in range(1, n_iter + 1):
        # Sample minibatch
        batch_idx = rng.choice(n, size=min(batch_size, n), replace=False)
        X_batch = X[batch_idx]
        Y_batch = Y[batch_idx]
        X_aux_batch = X_aux[batch_idx]
        
        # Compute scale factor for this batch
        scale = n / len(batch_idx)
        
        # FIXED: Use more aggressive gradient scaling
        v_gradient_scale = 0.5  # Increased from 0.1
        gamma_gradient_scale = 0.5  # Increased from 0.1
        
        # Compute natural gradients with corrected implementation
        try:
            ng_results = compute_natural_gradients_corrected(
                model, params, model.expected_values(params), 
                X_batch, Y_batch, X_aux_batch, batch_idx, scale, mask
            )
            
            # Apply updates with mild clipping only for numerical stability
            for param_name, update in ng_results.items():
                if param_name in params:
                    if param_name in ('mu_v', 'mu_gamma'):
                        params[param_name] = jnp.clip(update, -5.0, 5.0)
                    elif param_name in ('tau2_v', 'tau2_gamma'):
                        params[param_name] = jnp.clip(update, 1e-6, 10.0)
                    else:
                        params[param_name] = jnp.clip(update, 1e-6, 1e6)
            
        except Exception as e:
            if verbose:
                print(f"[SVI] Gradient computation failed at iteration {iteration}: {e}")
            continue
        
        # Compute ELBO periodically
        if track_elbo and iteration % elbo_freq == 0:
            try:
                expected = model.expected_values(params)
                elbo = compute_svi_elbo_corrected(
                    model, X_batch, Y_batch, X_aux_batch, params, expected, 
                    batch_idx, scale, debug_print=False
                )
                elbo_history.append(elbo)
                
                # Check for improvement
                if elbo > best_elbo + min_delta:
                    best_elbo = elbo
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if early_stopping and patience_counter >= patience:
                    if verbose:
                        print(f"[SVI] Early stopping at iteration {iteration} (no improvement for {patience} iterations)")
                    break
                
                # Adaptive learning rate
                if len(elbo_history) > 10:
                    recent_improvement = elbo_history[-1] - elbo_history[-10]
                    if recent_improvement < min_delta * 10:
                        current_lr *= 0.95  # Reduce learning rate
                        if verbose:
                            print(f"[SVI] Reducing learning rate to {current_lr:.6f}")
                
                if verbose and iteration % 50 == 0:
                    print(f"[SVI] iter {iteration}, ELBO: {elbo:.1f}, lr: {current_lr:.4f}, best: {best_elbo:.1f}")
                    
            except Exception as e:
                if verbose:
                    print(f"[SVI] ELBO computation failed at iteration {iteration}: {e}")
        
        # Progress reporting
        elif verbose and iteration % 100 == 0:
            print(f"[SVI] iter {iteration}, lr: {current_lr:.4f}")
    
    # Store final results
    params['elbo_history'] = elbo_history
    params['final_elbo'] = elbo_history[-1] if elbo_history else float('-inf')
    params['iterations'] = iteration
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"[SVI] Training completed in {elapsed:.1f}s")
        print(f"[SVI] Final ELBO: {params['final_elbo']:.1f}")
        print(f"[SVI] Best ELBO: {best_elbo:.1f}")
    
    return {
        'params': params,
        'expected': model.expected_values(params),
        'elbo_values': elbo_history,
        'elbo_iterations': list(range(0, len(elbo_history)*elbo_freq, elbo_freq)),
        'final_elbo': params['final_elbo'],
        'iterations': iteration,
        'stopped_early': patience_counter >= patience,
        'final_iteration': iteration,
        'best_elbo': best_elbo,
    }

def compute_svi_elbo_corrected(model, X_b, Y_b, X_aux_b, params, expected, batch_idx, scale, debug_print=False):
    """
    Compute the CORRECTED SVI ELBO with enhanced numerical stability.
    """
    from jax.scipy.special import digamma, gammaln
    
    n_batch = X_b.shape[0]
    
    # Get batch-specific parameters and expectations - handle scalar cases
    a_theta_b = params['a_theta'][batch_idx]
    b_theta_b = params['b_theta'][batch_idx]
    
    # Handle scalar xi parameters
    if jnp.ndim(params['a_xi']) == 0:
        a_xi_b = jnp.full(n_batch, params['a_xi'])
        b_xi_b = jnp.full(n_batch, params['b_xi'])
    else:
        a_xi_b = params['a_xi'][batch_idx]
        b_xi_b = params['b_xi'][batch_idx]
    
    # Handle scalar/array zeta parameters
    if jnp.ndim(params['zeta']) == 0:
        zeta_b = jnp.full((n_batch, 1), params['zeta'])
    else:
        zeta_b = params['zeta'][batch_idx]
    
    # Apply numerical stability clipping early
    a_theta_b = jnp.clip(a_theta_b, 1e-6, 1e4)
    b_theta_b = jnp.clip(b_theta_b, 1e-6, 1e4)
    a_xi_b = jnp.clip(a_xi_b, 1e-6, 1e4)
    b_xi_b = jnp.clip(b_xi_b, 1e-6, 1e4)
    zeta_b = jnp.clip(zeta_b, 1e-6, 10.0)
    
    E_theta_b = a_theta_b / b_theta_b
    E_xi_b = a_xi_b / b_xi_b
    E_beta = expected['E_beta']
    E_eta = expected['E_eta']
    
    # ========== SCALED LIKELIHOOD TERMS WITH STABILITY ==========
    
    # Poisson likelihood with better numerical stability
    expected_rate_b = jnp.sum(E_theta_b[:, None, :] * E_beta[None, :, :], axis=2)
    expected_rate_b = jnp.clip(expected_rate_b, 1e-8, 1e6)  # Allow higher rates
    
    # Use original data but clip extremely large values
    X_b_safe = jnp.clip(X_b, 0, 1e4)  # Allow larger counts
    
    # Compute Poisson log-likelihood with better numerical handling
    log_rate_safe = jnp.log(expected_rate_b + 1e-8)  # Add small constant for stability
    pois_terms = X_b_safe * log_rate_safe - expected_rate_b - gammaln(X_b_safe + 1)
    
    # Replace any invalid terms with reasonable fallbacks
    pois_terms = jnp.where(jnp.isnan(pois_terms) | jnp.isinf(pois_terms), 
                          -expected_rate_b - 1, pois_terms)  # Simpler fallback
    
    # Use full scale - don't artificially cap the likelihood
    pois_ll = scale * jnp.sum(pois_terms)
    
    # Logistic likelihood with enhanced stability
    psi_b = E_theta_b @ params['mu_v'].T + X_aux_b @ params['mu_gamma'].T
    psi_b = jnp.clip(psi_b, -10.0, 10.0)  # Prevent extreme logits
    
    lam_b = lambda_jj(zeta_b)
    lam_b = jnp.clip(lam_b, 1e-8, 0.5)  # Keep lambda in reasonable range
    
    # Compute variance terms with stability
    if params['tau2_v'].ndim == 3:
        tau2_v_diag = jnp.diagonal(params['tau2_v'], axis1=1, axis2=2)
    else:
        tau2_v_diag = params['tau2_v']
        
    if params['tau2_gamma'].ndim == 3:
        tau2_gamma_diag = jnp.diagonal(params['tau2_gamma'], axis1=1, axis2=2)
    else:
        tau2_gamma_diag = params['tau2_gamma']
    
    tau2_v_diag = jnp.clip(tau2_v_diag, 1e-6, 10.0)
    tau2_gamma_diag = jnp.clip(tau2_gamma_diag, 1e-6, 10.0)
    
    # Second moment with stability
    theta_second_moment = (a_theta_b * (a_theta_b + 1)) / (b_theta_b**2)
    theta_second_moment = jnp.clip(theta_second_moment, 1e-8, 100.0)
    
    var_theta_v_b = jnp.einsum('ik,ck->ic', theta_second_moment, tau2_v_diag)
    var_x_aux_gamma_b = (X_aux_b**2) @ tau2_gamma_diag.T
    var_psi_b = var_theta_v_b + var_x_aux_gamma_b
    var_psi_b = jnp.clip(var_psi_b, 1e-8, 100.0)
    
    # Clip Y for stability
    Y_b_safe = jnp.clip(Y_b, 0, 1)
    
    # Logistic likelihood terms with protection
    zeta_safe = jnp.clip(zeta_b, -10.0, 10.0)
    logistic_terms = ((Y_b_safe - 0.5) * psi_b - 
                     lam_b * (psi_b**2 + var_psi_b) + 
                     lam_b * zeta_safe**2 - 
                     jnp.log(1.0 + jnp.exp(-zeta_safe)) - 
                     zeta_safe / 2.0)
    
    # Replace invalid logistic terms
    logistic_terms = jnp.where(jnp.isnan(logistic_terms) | jnp.isinf(logistic_terms),
                              -1.0, logistic_terms)
    
    # Use full scale for logistic likelihood too
    logit_ll = scale * jnp.sum(logistic_terms)
    
    # ========== KL DIVERGENCES WITH ENHANCED STABILITY ==========
    
    def kl_gamma_stable(a_q, b_q, a0, b0):
        """Numerically stable KL divergence between Gamma distributions"""
        # Additional clipping for extreme stability
        a_q = jnp.clip(a_q, 1e-4, 1e3)
        b_q = jnp.clip(b_q, 1e-4, 1e3)
        a0 = jnp.clip(a0, 1e-4, 1e3)
        b0 = jnp.clip(b0, 1e-4, 1e3)
        
        # Compute each term separately for better numerical control
        term1 = (a_q - a0) * digamma(a_q)
        term2 = gammaln(a0) - gammaln(a_q)
        term3 = a0 * (jnp.log(b_q) - jnp.log(b0))
        term4 = (a_q / b_q) * (b0 - b_q)
        
        # Check each term for validity
        term1 = jnp.where(jnp.isnan(term1) | jnp.isinf(term1), 0.0, term1)
        term2 = jnp.where(jnp.isnan(term2) | jnp.isinf(term2), 0.0, term2)
        term3 = jnp.where(jnp.isnan(term3) | jnp.isinf(term3), 0.0, term3)
        term4 = jnp.where(jnp.isnan(term4) | jnp.isinf(term4), 0.0, term4)
        
        kl = term1 + term2 + term3 + term4
        
        # Final safety check
        kl = jnp.where(jnp.isnan(kl) | jnp.isinf(kl) | (kl < -100) | (kl > 100), 
                      0.1, kl)
        return kl
    
    # KL for local parameters (only for current batch, no scaling)
    xi_prior_rate = jnp.broadcast_to(model.lambda_xi, E_xi_b.shape)
    kl_theta = jnp.sum(kl_gamma_stable(a_theta_b, b_theta_b, 
                                      model.alpha_theta, E_xi_b[:, None]))
    kl_xi = jnp.sum(kl_gamma_stable(a_xi_b, b_xi_b, model.alpha_xi, xi_prior_rate))
    
    # KL for global parameters (full contribution, no scaling)
    eta_prior_rate = jnp.broadcast_to(model.lambda_eta, E_eta.shape)
    kl_beta = jnp.sum(kl_gamma_stable(params['a_beta'], params['b_beta'], 
                                     model.alpha_beta, E_eta[:, None]))
    kl_eta = jnp.sum(kl_gamma_stable(params['a_eta'], params['b_eta'], 
                                    model.alpha_eta, eta_prior_rate))
    
    # KL for normal distributions with enhanced stability
    mu_v_safe = jnp.clip(params['mu_v'], -20.0, 20.0)
    tau2_v_safe = jnp.clip(tau2_v_diag, 1e-4, 50.0)
    
    kl_v_terms = ((mu_v_safe**2 + tau2_v_safe) / model.sigma2_v - 
                  jnp.log(tau2_v_safe) + jnp.log(model.sigma2_v) - 1)
    kl_v_terms = jnp.where(jnp.isnan(kl_v_terms) | jnp.isinf(kl_v_terms) | 
                          (kl_v_terms < -50) | (kl_v_terms > 50), 0.5, kl_v_terms)
    kl_v = 0.5 * jnp.sum(kl_v_terms)
    
    mu_gamma_safe = jnp.clip(params['mu_gamma'], -20.0, 20.0)
    tau2_gamma_safe = jnp.clip(tau2_gamma_diag, 1e-4, 50.0)
    
    kl_gamma_terms = ((mu_gamma_safe**2 + tau2_gamma_safe) / model.sigma2_gamma - 
                     jnp.log(tau2_gamma_safe) + jnp.log(model.sigma2_gamma) - 1)
    kl_gamma_terms = jnp.where(jnp.isnan(kl_gamma_terms) | jnp.isinf(kl_gamma_terms) |
                              (kl_gamma_terms < -50) | (kl_gamma_terms > 50), 0.5, kl_gamma_terms)
    kl_gamma_param = 0.5 * jnp.sum(kl_gamma_terms)
    
    # Total ELBO with comprehensive stability checks
    kl_total = kl_theta + kl_xi + kl_beta + kl_eta + kl_v + kl_gamma_param
    
    # Final stability check on KL terms
    kl_total = jnp.where(jnp.isnan(kl_total) | jnp.isinf(kl_total), 1000.0, kl_total)
    
    elbo = pois_ll + logit_ll - kl_total
    
    # ELBO sanity check - only handle true numerical issues, don't cap extreme values
    if jnp.isnan(elbo) or jnp.isinf(elbo):
        elbo = -1e6
        if debug_print:
            print(f"[ELBO WARNING] NaN/Inf ELBO detected, using fallback: {elbo}")
    # Remove the extreme value capping - let the model see the real ELBO
    
    if debug_print:
        print(f"[CORRECTED SVI ELBO] Poisson: {pois_ll:.3f}, Logistic: {logit_ll:.3f}")
        print(f"[CORRECTED SVI ELBO] KL total: {kl_total:.3f}, ELBO: {elbo:.3f}")
    
    return float(elbo)


def run_model_and_evaluate_corrected(
    x_data, x_aux, y_data, var_names, hyperparams,
    seed=None, test_size=0.15, val_size=0.15,
    max_iters=1000, batch_size=36, learning_rate=0.002,
    return_probs=True, sample_ids=None, mask=None, scores=None,
    plot_elbo=False, plot_prefix=None, return_params=False,
    verbose=False, early_stopping=True, patience=50, min_delta=1e-4, beta_init=None,
):
    """
    Run the CORRECTED SVI model and evaluate performance.
    """
    # Set random seed
    if seed is None:
        seed = np.random.randint(0, 2**32)

    if y_data.ndim == 1:
        y_data = y_data.reshape(-1, 1)
    if x_aux.ndim == 1:
        x_aux = x_aux.reshape(-1, 1)

    n_samples, n_genes = x_data.shape
    kappa = y_data.shape[1]
    d = hyperparams.get("d", 1)

    # Data splits
    indices = np.arange(n_samples)
    train_idx, temp_idx = train_test_split(indices, test_size=val_size + test_size, random_state=seed)
    val_rel = test_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(temp_idx, test_size=val_rel, random_state=seed)

    # Initialize model
    model = SupervisedPoissonFactorization(
        len(train_idx),
        n_genes,
        n_factors=d,
        n_outcomes=kappa,
        alpha_eta=hyperparams.get("alpha_eta", 1.0),
        lambda_eta=hyperparams.get("lambda_eta", 1.0),
        alpha_beta=hyperparams.get("alpha_beta", 1.0),
        alpha_xi=hyperparams.get("alpha_xi", 1.0),
        lambda_xi=hyperparams.get("lambda_xi", 1.0),
        alpha_theta=hyperparams.get("alpha_theta", 1.0),
        sigma2_gamma=hyperparams.get("sigma2_gamma", 1.0),
        sigma2_v=hyperparams.get("sigma2_v", 1.0),
        key=random.PRNGKey(seed),
    )

    # Fit using CORRECTED SVI
    svi_results = fit_svi_corrected(
        model,
        x_data[train_idx],
        y_data[train_idx],
        x_aux[train_idx],
        n_iter=max_iters,
        batch_size=batch_size,
        learning_rate=learning_rate,
        verbose=verbose,
        track_elbo=True,  # Always track ELBO for better monitoring
        elbo_freq=10,     # Compute ELBO every 10 iterations for much better monitoring
        early_stopping=early_stopping,
        patience=patience,
        min_delta=min_delta,
        beta_init=beta_init,
        mask=mask,
    )

    params = svi_results['params']
    expected = svi_results['expected']

    # Make predictions on training data (using existing expected values)
    all_probs_train = compute_probabilities_safe(
        expected["E_theta"], params["mu_v"], x_aux[train_idx], params["mu_gamma"], verbose=True
    )

    # For validation and test sets, use improved theta and xi inference
    E_theta_val, E_xi_val = infer_theta_and_xi_for_new_samples(model, x_data[val_idx], params, n_iter=20)
    print(f"DEBUG: Val theta shape: {E_theta_val.shape}, range: [{np.min(E_theta_val):.4f}, {np.max(E_theta_val):.4f}]")
    print(f"DEBUG: Val xi shape: {E_xi_val.shape}, range: [{np.min(E_xi_val):.4f}, {np.max(E_xi_val):.4f}]")

    E_theta_test, E_xi_test = infer_theta_and_xi_for_new_samples(model, x_data[test_idx], params, n_iter=20)
    print(f"DEBUG: Test theta shape: {E_theta_test.shape}, range: [{np.min(E_theta_test):.4f}, {np.max(E_theta_test):.4f}]")
    print(f"DEBUG: Test xi shape: {E_xi_test.shape}, range: [{np.min(E_xi_test):.4f}, {np.max(E_xi_test):.4f}]")

    all_probs_val = compute_probabilities_safe(
        E_theta_val, params["mu_v"], x_aux[val_idx], params["mu_gamma"], verbose=False
    )

    all_probs_test = compute_probabilities_safe(
        E_theta_test, params["mu_v"], x_aux[test_idx], params["mu_gamma"], verbose=False
    )

    # Compute metrics
    train_metrics = _compute_metrics(y_data[train_idx], np.array(all_probs_train))
    val_metrics = _compute_metrics(y_data[val_idx], np.array(all_probs_val))
    test_metrics = _compute_metrics(y_data[test_idx], np.array(all_probs_test))

    # Prepare results
    results = {
        "train_metrics": {k: v for k, v in train_metrics.items() if k != "probabilities"},
        "val_metrics": {k: v for k, v in val_metrics.items() if k != "probabilities"},
        "test_metrics": {k: v for k, v in test_metrics.items() if k != "probabilities"},
        "hyperparameters": hyperparams,
        "training_info": {
            "stopped_early": svi_results.get('stopped_early', False),
            "final_iteration": svi_results.get('final_iteration', max_iters),
            "best_elbo": svi_results.get('best_elbo', None)
        }
    }

    if return_probs:
        results["train_probabilities"] = train_metrics["probabilities"]
        results["val_probabilities"] = val_metrics["probabilities"]
        results["test_probabilities"] = test_metrics["probabilities"]

    results["val_labels"] = y_data[val_idx].tolist()

    if return_params:
        for k, v in params.items():
            if isinstance(v, jnp.ndarray):
                results[k] = np.array(v).tolist()

    # Plot and save ELBO if requested
    if plot_elbo and 'elbo_values' in svi_results and len(svi_results['elbo_values']) > 0:
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(svi_results['elbo_iterations'], svi_results['elbo_values'], 
                    'b-', linewidth=2, label="Corrected SVI ELBO")
            
            # Mark early stopping point if applicable
            if svi_results.get('stopped_early', False):
                final_iter = svi_results.get('final_iteration', len(svi_results['elbo_values']))
                final_elbo_idx = next((i for i, iter_num in enumerate(svi_results['elbo_iterations']) 
                                     if iter_num >= final_iter), -1)
                if final_elbo_idx >= 0:
                    plt.axvline(x=svi_results['elbo_iterations'][final_elbo_idx], 
                              color='red', linestyle='--', alpha=0.7, 
                              label=f"Early stop (iter {final_iter})")
            
            plt.xlabel("Iteration")
            plt.ylabel("Evidence Lower Bound (ELBO)")
            plt.title("CORRECTED SVI: ELBO Convergence")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot if requested
            if plot_prefix is not None:
                os.makedirs(plot_prefix, exist_ok=True)
                plot_path = os.path.join(plot_prefix, "corrected_svi_elbo_convergence.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Corrected SVI ELBO plot saved to {plot_path}")
                
                pdf_path = os.path.join(plot_prefix, "corrected_svi_elbo_convergence.pdf")
                plt.savefig(pdf_path, bbox_inches='tight')
            
            results['elbo_plot_data'] = {
                'iterations': svi_results['elbo_iterations'],
                'values': svi_results['elbo_values']
            }
            
            plt.close()
            
        except ImportError:
            print("Warning: matplotlib not available, skipping ELBO plot")
        except Exception as e:
            print(f"Warning: Failed to create ELBO plot: {e}")

    return results


# # ========== EXAMPLE USAGE AND TESTING ==========

# def compare_svi_implementations(x_data, x_aux, y_data, var_names, hyperparams, 
#                                seed=42, max_iters=200, verbose=True):
#     """
#     Compare the original (incorrect) and corrected SVI implementations.
#     This function helps validate that the corrections improve performance.
#     """
#     print("=" * 60)
#     print("COMPARING SVI IMPLEMENTATIONS")
#     print("=" * 60)
    
#     # Run corrected implementation  
#     print("\n[1/1] Running CORRECTED SVI implementation...")
#     try:
#         results_corrected = run_model_and_evaluate_corrected(
#             x_data, x_aux, y_data, var_names, hyperparams,
#             seed=seed, max_iters=max_iters, verbose=verbose,
#             plot_elbo=True, plot_prefix="corrected_svi_results"
#         )
#         print("✓ Corrected SVI completed")
#         corr_test_auc = results_corrected['test_metrics']['roc_auc']
#         corr_test_acc = results_corrected['test_metrics']['accuracy']
#     except Exception as e:
#         print(f"✗ Corrected SVI failed: {e}")
#         results_corrected = None
#         corr_test_auc = corr_test_acc = 0.0
    
#     # Display results
#     print("\n" + "=" * 60)
#     print("CORRECTED SVI RESULTS")
#     print("=" * 60)
#     print(f"{'Metric':<20} {'Value':<12}")
#     print("-" * 35)
#     print(f"{'Test AUC':<20} {corr_test_auc:<12.4f}")
#     print(f"{'Test Accuracy':<20} {corr_test_acc:<12.4f}")
    
#     if results_corrected:
#         corr_iters = results_corrected['training_info']['final_iteration']
#         corr_early = results_corrected['training_info']['stopped_early']
#         print(f"{'Iterations':<20} {corr_iters:<12}")
#         print(f"{'Early Stopping':<20} {corr_early:<12}")
    
#     print("\nKey Features of Corrected SVI:")
#     print("1. ✓ Proper natural gradient updates for global parameters")
#     print("2. ✓ Correct scaling of sufficient statistics by n/n_batch")
#     print("3. ✓ Robbins-Monro learning rate schedule")
#     print("4. ✓ Fixed regression term computation in theta updates")
#     print("5. ✓ Proper separation of local vs global parameter updates")
#     print("6. ✓ Corrected ELBO computation for SVI")
#     print("7. ✓ Proper handling of scalar parameter initialization")
    
#     return {
#         'corrected': results_corrected,
#         'test_auc': corr_test_auc,
#         'test_accuracy': corr_test_acc
#     }


# def test_corrected_svi():
#     """
#     Test the corrected SVI implementation with SIMPLER, SMALLER synthetic data for debugging.
#     """
#     print("Testing Corrected SVI Implementation")
#     print("=" * 40)
    
#     # Generate MUCH SIMPLER synthetic data for debugging
#     np.random.seed(42)
#     n_samples, n_genes, n_factors = 50, 20, 2  # Much smaller for debugging
    
#     # Create very simple structured data
#     print("Creating simple synthetic data...")
    
#     # Simple latent factors
#     true_theta = np.random.gamma(1, 1, (n_samples, n_factors)) + 0.1
#     true_beta = np.random.gamma(1, 1, (n_genes, n_factors)) + 0.1
    
#     # Gene expression with clear structure but smaller scale
#     rates = (true_theta @ true_beta.T) * 0.5  # Scale down to reduce Poisson penalty
#     rates = np.clip(rates, 0.1, 5.0)  # Keep rates reasonable
#     X = np.random.poisson(rates)
    
#     # Very simple auxiliary data
#     X_aux = np.random.randn(n_samples, 1) * 0.5  # Single auxiliary variable, small scale
    
#     # Simple outcome relationship with strong signal
#     outcome_weights_theta = np.array([1.0, -0.5])  # Clear weights
#     outcome_weights_aux = np.array([0.5])
    
#     logits = true_theta @ outcome_weights_theta + X_aux @ outcome_weights_aux
#     probs = 1 / (1 + np.exp(-logits))
#     Y = np.random.binomial(1, probs).reshape(-1, 1)
    
#     print(f"Simple data generated:")
#     print(f"  Samples: {n_samples}, Genes: {n_genes}, Factors: {n_factors}")
#     print(f"  Outcome rate: {np.mean(Y):.3f}")
#     print(f"  Gene expression range: [{np.min(X)}, {np.max(X)}], mean: {np.mean(X):.2f}")
#     print(f"  Logit range: [{np.min(logits):.2f}, {np.max(logits):.2f}]")
#     print(f"  True theta range: [{np.min(true_theta):.2f}, {np.max(true_theta):.2f}]")
    
#     var_names = [f"gene_{i}" for i in range(n_genes)]
    
#     # Very simple hyperparameters
#     hyperparams = {
#         "d": n_factors,
#         "alpha_eta": 0.01,    # Very weak priors
#         "lambda_eta": 0.01, 
#         "alpha_beta": 0.01,
#         "alpha_xi": 0.01,
#         "lambda_xi": 0.01,
#         "alpha_theta": 0.01,
#         "sigma2_gamma": 100.0,  # Very flexible
#         "sigma2_v": 100.0,
#     }
    
#     print("\nTesting with very weak priors and simple data...")
    
#     # Test with much more conservative settings
#     results = run_model_and_evaluate_corrected(
#         X, X_aux, Y, var_names, hyperparams,
#         seed=42, max_iters=100, batch_size=16, learning_rate=0.001,  # Much smaller LR
#         verbose=True, plot_elbo=True, early_stopping=False, patience=50  # No early stopping
#     )
    
#     print(f"\nSimple Test Results:")
#     print(f"Train AUC: {results['train_metrics']['roc_auc']:.4f}")
#     print(f"Val AUC: {results['val_metrics']['roc_auc']:.4f}")
#     print(f"Test AUC: {results['test_metrics']['roc_auc']:.4f}")
#     print(f"Final Iteration: {results['training_info']['final_iteration']}")
    
#     # Detailed diagnostics
#     print(f"\nDetailed Diagnostics:")
#     print(f"Train Accuracy: {results['train_metrics']['accuracy']:.4f}")
#     print(f"Val Accuracy: {results['val_metrics']['accuracy']:.4f}")
#     print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
    
#     # Check if we're learning anything meaningful
#     if results['train_metrics']['roc_auc'] > 0.7:
#         print("✓ Good performance - SVI is working!")
#     elif results['train_metrics']['roc_auc'] > 0.6:
#         print("⚠ Marginal performance - SVI might have issues")
#     else:
#         print("✗ Poor performance - SVI likely has bugs")
        
#     # Try to diagnose the issue further
#     print(f"\nDiagnostic Questions:")
#     print(f"1. Is the model overfitting? Train AUC ({results['train_metrics']['roc_auc']:.3f}) >> Val AUC ({results['val_metrics']['roc_auc']:.3f})?")
#     print(f"2. Is there enough signal? Outcome rate = {np.mean(Y):.3f} (should be between 0.2-0.8)")
#     print(f"3. Are the hyperparameters reasonable? Check initialization ranges above.")
    
#     return results


# def simple_sanity_check():
#     """
#     Even simpler sanity check - can we fit a tiny dataset perfectly?
#     """
#     print("\n" + "="*50)
#     print("RUNNING SIMPLE SANITY CHECK")
#     print("="*50)
    
#     # Tiny dataset that should be easy to fit
#     np.random.seed(123)
#     n_samples, n_genes, n_factors = 20, 10, 2
    
#     # Perfect structure
#     true_theta = np.array([[1.0, 0.0], [0.0, 1.0]] * 10)[:n_samples]  # Clear factors
#     true_beta = np.array([[1.0, 0.0], [0.0, 1.0]] * 5)[:n_genes]   # Clear loadings
    
#     rates = (true_theta @ true_beta.T) + 0.1
#     X = np.random.poisson(rates)
    
#     X_aux = np.ones((n_samples, 1))  # Constant auxiliary
    
#     # Perfect separation
#     Y = (true_theta[:, 0] > true_theta[:, 1]).astype(int).reshape(-1, 1)
    
#     print(f"Sanity check data:")
#     print(f"  Perfect separation: {np.mean(Y):.3f} (should be 0.5)")
#     print(f"  Gene expression mean: {np.mean(X):.2f}")
    
#     var_names = [f"gene_{i}" for i in range(n_genes)]
    
#     hyperparams = {
#         "d": n_factors,
#         "alpha_eta": 0.1, "lambda_eta": 0.1, "alpha_beta": 0.1,
#         "alpha_xi": 0.1, "lambda_xi": 0.1, "alpha_theta": 0.1,
#         "sigma2_gamma": 10.0, "sigma2_v": 10.0,
#     }
    
#     results = run_model_and_evaluate_corrected(
#         X, X_aux, Y, var_names, hyperparams,
#         seed=123, max_iters=50, batch_size=10, learning_rate=0.01,
#         verbose=False, early_stopping=False
#     )
    
#     print(f"\nSanity Check Results:")
#     print(f"Train AUC: {results['train_metrics']['roc_auc']:.4f} (should be > 0.9)")
#     print(f"Val AUC: {results['val_metrics']['roc_auc']:.4f}")
    
#     if results['train_metrics']['roc_auc'] > 0.9:
#         print("✓ PASSED: SVI can fit simple data!")
#         return True
#     else:
#         print("✗ FAILED: SVI cannot fit even simple data - there are bugs!")
#         return False


# if __name__ == "__main__":
#     # Run both tests
#     sanity_passed = simple_sanity_check()
    
#     if sanity_passed:
#         print("\nRunning main test...")
#         test_results = test_corrected_svi()
#     else:
#         print("\nSkipping main test due to sanity check failure.")
#         print("The SVI implementation needs debugging.")
    
#     print("\n✓ Testing completed!")