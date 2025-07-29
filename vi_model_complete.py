# Force JAX to use CPU only - must be set before importing jax
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
import jax.random as random
from jax import vmap, jit
import jax.scipy as jsp
from typing import Dict, Tuple, Union, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# Helper functions
def lambda_jj(zeta):
    """Jaakola-Jordan lambda function with numerical stability"""
    # Prevent division by very small numbers
    zeta_safe = jnp.maximum(jnp.abs(zeta), 1e-6)
    result = jnp.where(
        jnp.abs(zeta) < 1e-6, 
        1/8, 
        (1/(4*zeta_safe)) * jnp.tanh(zeta_safe/2)
    )
    # Bound the result to prevent extreme values
    return jnp.clip(result, 1e-8, 100.0)

def logistic(x):
    """Logistic function"""
    return 1 / (1 + jnp.exp(-x))

class SupervisedPoissonFactorization:
    def __init__(self, n_samples, n_genes, n_factors, n_outcomes, 
                 alpha_eta=1.0, lambda_eta=1.0, alpha_beta=1.0, 
                 alpha_xi=1.0, lambda_xi=1.0, alpha_theta=1.0,
                 sigma2_gamma=1.0, sigma2_v=1.0, key=None):
        
        print(f"DEBUG: SupervisedPoissonFactorization.__init__ called with n_samples={n_samples}, n_genes={n_genes}, n_factors={n_factors}, n_outcomes={n_outcomes}")
        self.n, self.p, self.K, self.kappa = n_samples, n_genes, n_factors, n_outcomes
        self.alpha_eta, self.lambda_eta = alpha_eta, lambda_eta
        self.alpha_beta, self.alpha_xi = alpha_beta, alpha_xi
        self.lambda_xi, self.alpha_theta = lambda_xi, alpha_theta
        self.sigma2_gamma, self.sigma2_v = sigma2_gamma, sigma2_v
        
        if key is None:
            key = random.PRNGKey(0)
        self.key = key
    
    def initialize_parameters(self, X, Y, X_aux):
        """Initialize variational parameters using informed guesses"""
        print("DEBUG: Starting parameter initialization")
        
        keys = random.split(self.key, 10)
        print("DEBUG: Generated random keys")
        
        # Use data statistics for better initialization
        gene_means = jnp.mean(X, axis=0)
        sample_totals = jnp.sum(X, axis=1)
        print("DEBUG: Computed basic statistics")
        print(f"DEBUG: Data range - gene_means: {jnp.min(gene_means):.2f} to {jnp.max(gene_means):.2f}")
        print(f"DEBUG: Data range - sample_totals: {jnp.min(sample_totals):.2f} to {jnp.max(sample_totals):.2f}")
        
        # FIXED: Use more diverse initialization instead of overly conservative values
        # Initialize theta with data-driven values
        theta_init = jnp.outer(sample_totals / jnp.mean(sample_totals), 
                            jnp.ones(self.K)) + random.normal(keys[0], (self.n, self.K)) * 0.2
        theta_init = jnp.maximum(theta_init, 0.1)
        print("DEBUG: Initialized theta with data-driven values")
        
        # Initialize beta with data-driven values
        beta_init = jnp.outer(gene_means / jnp.mean(gene_means), 
                            jnp.ones(self.K)) + random.normal(keys[1], (self.p, self.K)) * 0.2
        beta_init = jnp.maximum(beta_init, 0.1)
        print("DEBUG: Initialized beta with data-driven values")
        
        # Convert to Gamma parameters with reasonable variance
        theta_var = theta_init * 0.3  # More reasonable variance
        a_theta = jnp.clip((theta_init**2) / theta_var, 0.5, 10.0)  # Wider bounds
        b_theta = jnp.clip(theta_init / theta_var, 0.5, 10.0)
        
        beta_var = beta_init * 0.3  # More reasonable variance
        a_beta = jnp.clip((beta_init**2) / beta_var, 0.5, 10.0)  # Wider bounds
        b_beta = jnp.clip(beta_init / beta_var, 0.5, 10.0)
        print("DEBUG: Converted to Gamma parameters")
        
        # Initialize eta and xi with reasonable values
        a_eta = jnp.full(self.p, self.alpha_eta + self.K * self.alpha_beta)
        b_eta = self.lambda_eta + jnp.mean(a_beta / b_beta, axis=1)
        
        a_xi = jnp.full(self.n, self.alpha_xi + self.K * self.alpha_theta)
        b_xi = self.lambda_xi + jnp.sum(a_theta / b_theta, axis=1)
        print("DEBUG: Initialized eta and xi")
        
        # FIXED: Initialize gamma with more reasonable values
        print("DEBUG: About to initialize gamma...")
        if Y.shape[0] > 1:
            try:
                print("DEBUG: Computing XTX and XTY...")
                XTX = X_aux.T @ X_aux + jnp.eye(X_aux.shape[1]) * 1e-6
                XTY = X_aux.T @ Y
                print("DEBUG: Solving linear system...")
                gamma_init = jnp.linalg.solve(XTX, XTY).T
                print("DEBUG: Linear solve completed")
                # Add some noise to avoid exact linear solution
                gamma_init = gamma_init + random.normal(keys[2], gamma_init.shape) * 0.5  # Increased noise
            except Exception as e:
                print(f"DEBUG: Linear solve failed: {e}")
                gamma_init = random.normal(keys[2], (self.kappa, X_aux.shape[1])) * 1.0  # Larger values
        else:
            gamma_init = random.normal(keys[2], (self.kappa, X_aux.shape[1])) * 1.0  # Larger values
        print("DEBUG: Gamma initialization completed")
        
        mu_gamma = gamma_init
        tau2_gamma = jnp.ones((self.kappa, X_aux.shape[1])) * self.sigma2_gamma
        
        # FIXED: Initialize v with more reasonable values to allow proper learning
        print("DEBUG: About to initialize v...")
        print("DEBUG: Using reasonable v initialization")
        
        # Use more reasonable initialization to allow proper learning
        v_init = random.normal(keys[3], (self.kappa, self.K)) * 0.5  # Larger scale for better exploration
        
        # Ensure diversity across gene programs for single outcome case
        if self.kappa == 1:
            # Add extra randomness to ensure std > 0
            extra_noise = random.normal(keys[4], (self.K,)) * 0.3  # Larger noise
            v_init = v_init.at[0].set(v_init[0] + extra_noise)
        
        print(f"DEBUG: v_init shape: {v_init.shape}")
        print(f"DEBUG: v_init stats: min={jnp.min(v_init):.6f}, max={jnp.max(v_init):.6f}, mean={jnp.mean(v_init):.6f}, std={jnp.std(v_init):.6f}")
        print(f"DEBUG: theta_init stats: min={jnp.min(theta_init):.6f}, max={jnp.max(theta_init):.6f}, mean={jnp.mean(theta_init):.6f}, std={jnp.std(theta_init):.6f}")
        print(f"DEBUG: beta_init stats: min={jnp.min(beta_init):.6f}, max={jnp.max(beta_init):.6f}, mean={jnp.mean(beta_init):.6f}, std={jnp.std(beta_init):.6f}")
        print(f"DEBUG: gamma_init shape: {gamma_init.shape}, stats: min={jnp.min(gamma_init):.6f}, max={jnp.max(gamma_init):.6f}, mean={jnp.mean(gamma_init):.6f}, std={jnp.std(gamma_init):.6f}")
             
        mu_v = v_init
        tau2_v = jnp.ones((self.kappa, self.K)) * self.sigma2_v
        
        # Initialize zeta based on expected activations
        theta_exp = a_theta / b_theta  # (n, K)
        expected_linear = theta_exp @ mu_v.T + X_aux @ mu_gamma.T
        
        if expected_linear.ndim == 1:
            expected_linear = expected_linear[:, None]
        
        # FIXED: Use more reasonable zeta initialization
        zeta = jnp.abs(expected_linear) + 0.5  # Larger offset for better stability
        
        return {
            'a_eta': a_eta, 'b_eta': b_eta,
            'a_xi': a_xi, 'b_xi': b_xi,
            'a_beta': a_beta, 'b_beta': b_beta,
            'a_theta': a_theta, 'b_theta': b_theta,
            'mu_gamma': mu_gamma, 'tau2_gamma': tau2_gamma,
            'mu_v': mu_v, 'tau2_v': tau2_v,
            'zeta': zeta
        }

    def expected_values(self, params):
        """Compute expected values from variational parameters"""
        E_eta = params['a_eta'] / params['b_eta']
        E_xi = params['a_xi'] / params['b_xi']
        E_beta = params['a_beta'] / params['b_beta']
        E_theta = params['a_theta'] / params['b_theta']
        E_gamma = params['mu_gamma']
        E_v = params['mu_v']
        
        # Second moments for theta (needed for regression)
        E_theta_sq = (params['a_theta'] / params['b_theta']**2) * (params['a_theta'] + 1)
        E_theta_theta_T = jnp.expand_dims(E_theta_sq, -1) * jnp.eye(self.K) + \
                         jnp.expand_dims(E_theta, -1) @ jnp.expand_dims(E_theta, -2)
        
        return {
            'E_eta': E_eta, 'E_xi': E_xi, 'E_beta': E_beta, 'E_theta': E_theta,
            'E_gamma': E_gamma, 'E_v': E_v, 'E_theta_theta_T': E_theta_theta_T
        }
    
    def update_z_latent(self, X, E_theta, E_beta):
        """Update latent variables z_ijl using multinomial probabilities"""
        # Compute rates for each factor
        rates = jnp.expand_dims(E_theta, 1) * jnp.expand_dims(E_beta, 0)  # (n, p, K)
        total_rates = jnp.sum(rates, axis=2, keepdims=True)  # (n, p, 1)
        
        # Multinomial probabilities
        probs = rates / (total_rates + 1e-8)  # (n, p, K)
        
        # Expected z_ijl = x_ij * prob_ijl
        z = jnp.expand_dims(X, 2) * probs  # (n, p, K)
        
        return z
    
    def update_eta(self, params, expected_vals):
        """Update eta parameters - equation (13)"""
        a_eta_new = self.alpha_eta + self.K * self.alpha_beta
        b_eta_new = self.lambda_eta + jnp.sum(expected_vals['E_beta'], axis=1)
        
        # Only ensure positivity for numerical stability
        b_eta_new = jnp.maximum(b_eta_new, 1e-8)
        
        return {'a_eta': jnp.full(self.p, a_eta_new), 'b_eta': b_eta_new}
    
    def update_xi(self, params, expected_vals):
        """Update xi parameters - equation (14)"""
        a_xi_new = self.alpha_xi + self.K * self.alpha_theta
        b_xi_new = self.lambda_xi + jnp.sum(expected_vals['E_theta'], axis=1)
        
        # Only ensure positivity for numerical stability
        b_xi_new = jnp.maximum(b_xi_new, 1e-8)
        
        return {'a_xi': jnp.full(self.n, a_xi_new), 'b_xi': b_xi_new}
    
    def update_beta(self, params, expected_vals, z, mask=None):
        """Update beta parameters - equation (20)"""
        a_beta_new = self.alpha_beta + jnp.sum(z, axis=0)  # (p, K)
        b_beta_new = jnp.expand_dims(expected_vals['E_eta'], 1) + \
                     jnp.sum(expected_vals['E_theta'], axis=0, keepdims=True)  # (p, K)
        
        # Apply mask constraint if provided
        if mask is not None:
            mask_jnp = jnp.array(mask)
            a_beta_new = jnp.where(mask_jnp > 0, a_beta_new, 1e-10)
            b_beta_new = jnp.where(mask_jnp > 0, b_beta_new, 1.0)
        
        # Only ensure positivity for numerical stability
        a_beta_new = jnp.maximum(a_beta_new, 1e-8)
        b_beta_new = jnp.maximum(b_beta_new, 1e-8)
        
        return {'a_beta': a_beta_new, 'b_beta': b_beta_new}
    
    # def update_v(self, params, expected_vals, Y, X_aux):
    #     """Update v parameters - equation (10)"""
    #     lambda_vals = lambda_jj(params['zeta'])  # (n, κ)
        
    #     # Compute precision matrix (diagonal + low-rank update)
    #     prec_diag = 1/self.sigma2_v + 2 * jnp.sum(
    #         jnp.expand_dims(lambda_vals, -1) * 
    #         jnp.expand_dims(jnp.diagonal(expected_vals['E_theta_theta_T'], axis1=1, axis2=2), 1), 
    #         axis=0
    #     )  # (kappa, K)
        
    #     # Compute mean
    #     y_centered = Y - 0.5  # (n, kappa)
    #     reg_term = 2 * lambda_vals * jnp.sum(
    #         jnp.expand_dims(expected_vals['E_theta'], 1) * 
    #         jnp.expand_dims(X_aux @ expected_vals['E_gamma'].T, -1), 
    #         axis=2
    #     )  # (n, kappa)
        
    #     linear_term = jnp.sum(
    #         jnp.expand_dims(y_centered - reg_term, -1) * 
    #         jnp.expand_dims(expected_vals['E_theta'], 1), 
    #         axis=0
    #     )  # (kappa, K)
        
    #     # Update (assuming diagonal approximation for simplicity)
    #     tau2_v_new = 1 / prec_diag
    #     mu_v_new = tau2_v_new * linear_term
        
    #     return {'mu_v': mu_v_new, 'tau2_v': tau2_v_new}
    
    # def update_gamma(self, params, expected_vals, Y, X_aux):
    #     """Update gamma parameters - equation (11)"""
    #     lambda_vals = lambda_jj(params['zeta'])  # (n, kappa)
        
    #     # Compute precision matrix
    #     prec_diag = 1/self.sigma2_gamma + 2 * jnp.einsum('ic,id->cd', lambda_vals, X_aux**2)
        
    #     # Compute mean
    #     y_centered = Y - 0.5  # (n, kappa)
    #     reg_term = 2 * lambda_vals * jnp.sum(
    #         jnp.expand_dims(expected_vals['E_theta'], 1) * 
    #         jnp.expand_dims(expected_vals['E_v'], 0), 
    #         axis=2
    #     )  # (n, kappa)
        
    #     linear_term = jnp.einsum('ic,id->cd', y_centered - reg_term, X_aux)
        
    #     # Update
    #     tau2_gamma_new = 1 / prec_diag
    #     mu_gamma_new = tau2_gamma_new * linear_term
        
    #     return {'mu_gamma': mu_gamma_new, 'tau2_gamma': tau2_gamma_new}

    def update_v_full(self, params, expected, Y, X_aux):
        """Exact JJ‑Gaussian update with full K×K covariance."""
        n, K, kappa = self.n, self.K, self.kappa
        lam   = lambda_jj(params["zeta"])             # (n, κ)
        theta = expected["E_theta"]                   # (n, K)
        # Broadcast theta and lambda to compatible shapes before multiplication
        # theta.T : (K, n) -> (K, n, 1)
        # lam     : (n, κ) -> (1, n, κ)
        ThetaT_Lam = theta.T[:, :, None] * lam[None, :, :]  # (K, n, κ)

        # Pre‑allocate outputs
        mu_v   = jnp.zeros((kappa, K))
        tau2_v = jnp.zeros((kappa, K, K))             # full Σ

        I = jnp.eye(K) / self.sigma2_v                # (K,K)

        for k in range(kappa):
            # Precision Σ^{-1}
            S = I + 2.0 * (ThetaT_Lam[:, :, k] @ theta)   # (K,K)

            # Cholesky solve for Σ and μ
            L = jnp.linalg.cholesky(S)
            Sigma_k = jsp.linalg.cho_solve((L, True), jnp.eye(K))  # Σ
            tau2_v = tau2_v.at[k].set(Sigma_k)

            # Right‑hand side for the mean
            rhs = ((Y[:, k] - 0.5)                     # (n,)
                    - 2.0 * lam[:, k] * (X_aux @ expected["E_gamma"][k]))  # (n,)
            rhs = rhs @ theta                          # (K,)
            mu_k = Sigma_k @ rhs
            mu_v = mu_v.at[k].set(mu_k)

        return {"mu_v": mu_v, "tau2_v": tau2_v}

    def update_gamma_full(self, params, expected, Y, X_aux):
        lam = lambda_jj(params["zeta"])               # (n,κ)
        theta = expected["E_theta"]                   # (n,K)
        v     = expected["E_v"]                       # (κ,K)

        kappa, d = self.kappa, X_aux.shape[1]
        mu_g   = jnp.zeros((kappa, d))
        tau2_g = jnp.zeros((kappa, d, d))

        I = jnp.eye(d) / self.sigma2_gamma

        for k in range(kappa):
            # Σ^{-1}_γ
            S = I + 2.0 * (X_aux.T * lam[:, k]) @ X_aux
            L = jnp.linalg.cholesky(S)
            Sigma_k = jsp.linalg.cho_solve((L, True), jnp.eye(d))
            tau2_g = tau2_g.at[k].set(Sigma_k)

            rhs = ((Y[:, k] - 0.5)
                - 2.0 * lam[:, k] * (theta @ v[k]))  # (n,)
            rhs = rhs @ X_aux                           # (d,)
            mu_k = Sigma_k @ rhs
            mu_g = mu_g.at[k].set(mu_k)

        return {"mu_gamma": mu_g, "tau2_gamma": tau2_g}

    
    def update_theta(self, params, expected_vals, z, Y, X_aux):
        """Update theta parameters with both Poisson and logistic regression terms"""
        # Shape parameter (this is definitely correct)
        a_theta_new = self.alpha_theta + jnp.sum(z, axis=1)  # (n, K)
        
        # Rate parameter - Poisson terms
        b_theta_poisson = jnp.expand_dims(expected_vals['E_xi'], 1) + \
                         jnp.sum(expected_vals['E_beta'], axis=0)  # (n, K)
        
        # Logistic regression terms from Jaakola-Jordan bound
        # According to Equation (27): regression rate terms = 
        # sum_{k=1 to kappa} [-(y_ik - 1/2) E[v_kl] - 2 lambda(zeta_ik) E[v_kl] x_i^aux E[gamma_k]^T - 2 lambda(zeta_ik) E[v_kl]^2 theta_il^current]
        
        # Get current theta values for the linearization term
        theta_current = expected_vals['E_theta']  # (n, K)
        
        # Compute lambda values from zeta
        lam = lambda_jj(params['zeta'])  # (n, kappa)
        
        # Initialize logistic terms
        b_theta_logistic = jnp.zeros_like(b_theta_poisson)  # (n, K)
        
        # Loop over outcomes k to compute logistic terms
        for k in range(self.kappa):
            # Term 1: -(y_ik - 1/2) E[v_kl]
            term1 = -(Y[:, k:k+1] - 0.5) * expected_vals['E_v'][k, :]  # (n, K)
            
            # Term 2: -2 lambda(zeta_ik) E[v_kl] x_i^aux E[gamma_k]^T
            # This term involves the interaction between auxiliary covariates and gamma
            term2 = -2.0 * jnp.expand_dims(lam[:, k], 1) * expected_vals['E_v'][k, :] * \
                    jnp.expand_dims(X_aux @ expected_vals['E_gamma'][k, :].T, 1)  # (n, K)
            
            # Term 3: -2 lambda(zeta_ik) E[v_kl]^2 theta_il^current
            # This is the linearization term using current theta values
            term3 = -2.0 * jnp.expand_dims(lam[:, k], 1) * \
                    jnp.expand_dims(expected_vals['E_v'][k, :]**2, 0) * theta_current  # (n, K)
            
            # Sum all terms for this outcome
            b_theta_logistic += term1 + term2 + term3
        
        # Combine Poisson and logistic terms
        b_theta_new = b_theta_poisson + b_theta_logistic
        
        # Ensure positivity
        b_theta_new = jnp.maximum(b_theta_new, 1e-8)

        return {'a_theta': a_theta_new, 'b_theta': b_theta_new}
    
    def update_zeta(self, params, expected_vals, Y, X_aux):
        """Update auxiliary parameters zeta for Jaakola-Jordan bound"""
        # Compute A = theta_i * v_k^T + x_aux_i * gamma_k^T
        A = jnp.sum(jnp.expand_dims(expected_vals['E_theta'], 1) * 
                   jnp.expand_dims(expected_vals['E_v'], 0), axis=2) + \
            X_aux @ expected_vals['E_gamma'].T  # (n, kappa)
        
        # CORRECTED: Jaakola-Jordan optimal setting is zeta = |A| with small offset
        # for numerical stability
        zeta_new = jnp.abs(A) + 0.01  # Small offset for numerical stability
        
        return {'zeta': zeta_new}
    def fit(self, X, Y, X_aux, n_iter=100, tol=1e-4, verbose=False, mask=None):
        """Main fitting loop.

        Parameters
        ----------
        X, Y, X_aux : jnp.ndarray
            Input data.
        n_iter : int, optional
            Maximum number of iterations.
        tol : float, optional
            Convergence tolerance based on the change in expected theta.
        verbose : bool, optional
            If True, print ELBO components at each iteration.
        """
        print(f"DEBUG: Starting model fit with n_iter={n_iter}, verbose={verbose}")
        print(f"DEBUG: Data shapes: X={X.shape}, Y={Y.shape}, X_aux={X_aux.shape}")
        
        print(f"DEBUG: About to initialize parameters...")
        params = self.initialize_parameters(X, Y, X_aux)
        print(f"DEBUG: Parameters initialized successfully")

        from typing import Union
        previous_elbo: Union[float, None] = None
        elbo_history = []  # Track ELBO values for plotting
        
        for iteration in range(n_iter):
            old_params = params.copy()
            
            # Compute expected values
            expected_vals = self.expected_values(params)

            # Update latent variables (uses current expectations)
            z = self.update_z_latent(X, expected_vals['E_theta'], expected_vals['E_beta'])

            # Sequentially update parameters; after each update refresh expectations
            params.update(self.update_eta(params, expected_vals))
            expected_vals = self.expected_values(params)

            params.update(self.update_xi(params, expected_vals))
            expected_vals = self.expected_values(params)

            params.update(self.update_beta(params, expected_vals, z, mask))
            expected_vals = self.expected_values(params)

            # CRITICAL FIX: Update z after beta update since beta affects z computation
            z = self.update_z_latent(X, expected_vals['E_theta'], expected_vals['E_beta'])

            params.update(self.update_v_full(params, expected_vals, Y, X_aux))
            expected_vals = self.expected_values(params)

            params.update(self.update_gamma_full(params, expected_vals, Y, X_aux))
            expected_vals = self.expected_values(params)

            params.update(self.update_theta(params, expected_vals, z, Y, X_aux))
            expected_vals = self.expected_values(params)

            # CRITICAL FIX: Update z again after theta update since theta affects z computation
            z = self.update_z_latent(X, expected_vals['E_theta'], expected_vals['E_beta'])

            params.update(self.update_zeta(params, expected_vals, Y, X_aux))

            # Emergency stability check (only when needed)
            max_param_val = max(
                jnp.max(params['b_theta']), 
                jnp.max(params['b_beta']),
                jnp.max(params.get('tau2_v', 0)),
                jnp.max(params.get('tau2_gamma', 0))
            )
            
            # DISABLED: Aggressive clipping that might interfere with coordinate ascent
            # Apply stabilization after EVERY iteration to prevent feedback loops
            # params['b_eta'] = jnp.clip(params['b_eta'], 1e-8, 1e3)
            # params['b_xi'] = jnp.clip(params['b_xi'], 1e-8, 1e3)
            # params['b_theta'] = jnp.clip(params['b_theta'], 1e-8, 1e4)
            # params['b_beta'] = jnp.clip(params['b_beta'], 1e-8, 1e4)
            
            # Only apply emergency clipping when truly needed
            if max_param_val > 1e8 or jnp.any(jnp.isnan(params['b_theta'])) or jnp.any(jnp.isinf(params['b_theta'])):
                print(f"Emergency bounds applied at iteration {iteration+1}")
                params['b_theta'] = jnp.clip(params['b_theta'], 1e-8, 1e6)
                params['b_beta'] = jnp.clip(params['b_beta'], 1e-8, 1e6)
                params['b_eta'] = jnp.clip(params['b_eta'], 1e-8, 1e4)
                params['b_xi'] = jnp.clip(params['b_xi'], 1e-8, 1e4)
                if 'tau2_v' in params:
                    params['tau2_v'] = jnp.clip(params['tau2_v'], 1e-8, 100.0)
                if 'tau2_gamma' in params:
                    params['tau2_gamma'] = jnp.clip(params['tau2_gamma'], 1e-8, 100.0)
                params['zeta'] = jnp.clip(params['zeta'], 1e-3, 100.0)

            # *** ALWAYS COMPUTE ELBO FOR MONOTONICITY CHECK ***
            current_elbo_val = self.compute_elbo(X, Y, X_aux, params, z, return_components=False, debug_print=False)
            # When return_components=False, compute_elbo returns a float directly
            elbo_history.append(current_elbo_val)  # Track for plotting
            
            # === ELBO MONOTONICITY CHECK ===
            if previous_elbo is not None and isinstance(current_elbo_val, float) and current_elbo_val < previous_elbo:
                elbo_decrease = previous_elbo - current_elbo_val
                print(f"ELBO decreased by {elbo_decrease:.3f}: {previous_elbo:.3f} → {current_elbo_val:.3f}")
                print("This indicates a coordinate ascent issue!")
                
                # Check if decrease is significant
                if elbo_decrease > abs(float(previous_elbo)) * 1e-6:  # More than 1e-6 relative decrease
                    print("Significant ELBO decrease detected!")
            
            # === ELBO VALIDATION ===
            # Check for NaN/Inf
            if isinstance(current_elbo_val, float) and (jnp.isnan(current_elbo_val) or jnp.isinf(current_elbo_val)):
                print(f"CRITICAL: ELBO is NaN/Inf at iteration {iteration+1}")
                print("Stopping training to prevent further instability")
                break
            
            # Check for unreasonably large values
            if isinstance(current_elbo_val, float) and abs(current_elbo_val) > 1e12:
                print(f"WARNING: ELBO magnitude too large: {current_elbo_val:.2e}")
                print("Applying emergency parameter stabilization...")
                
                # Emergency parameter clipping
                params['tau2_v'] = jnp.clip(params['tau2_v'], 1e-8, 10.0)
                params['tau2_gamma'] = jnp.clip(params['tau2_gamma'], 1e-8, 10.0)
                params['zeta'] = jnp.clip(params['zeta'], 1e-3, 10.0)
                params['b_theta'] = jnp.clip(params['b_theta'], 1e-6, 1e4)
                params['b_beta'] = jnp.clip(params['b_beta'], 1e-6, 1e4)

            if verbose:
                # Get detailed components for verbose output WITH debug printing
                comps = self.compute_elbo(
                    X, Y, X_aux, params, z, return_components=True, debug_print=True
                )  # Only debug print here when verbose=True
                
                # Check individual components for explosion
                for comp_name, comp_val in comps.items():
                    if comp_name != 'elbo' and abs(comp_val) > 1e10:
                        print(f"Component {comp_name} is large: {comp_val:.2e}")
                
                print(
                    f"Iter {iteration + 1}: ELBO={comps['elbo']:.3f}, "
                    f"Poisson={comps['pois_ll']:.3f}, Logistic={comps['logit_ll']:.3f}, "
                    f"KL_theta={comps['kl_theta']:.3f}, KL_beta={comps['kl_beta']:.3f}, "
                    f"KL_eta={comps['kl_eta']:.3f}, KL_xi={comps['kl_xi']:.3f}, "
                    f"KL_v={comps['kl_v']:.3f}, KL_gamma={comps['kl_gamma']:.3f}"
                )
            
            # *** UPDATE PREVIOUS ELBO FOR NEXT ITERATION ***
            previous_elbo = current_elbo_val

            # Check convergence (simple check on theta means)
            if iteration > 0:
                theta_diff = jnp.mean(jnp.abs(params['a_theta']/params['b_theta'] -
                                            old_params['a_theta']/old_params['b_theta']))
                if theta_diff < tol:
                    print(f"Converged after {iteration+1} iterations")
                    break
        
        # Store ELBO history in params for plotting
        params['elbo_history'] = elbo_history
        return params, expected_vals

    # ----------------- ELBO (unchanged, but with λ clip) ----------------
    def compute_elbo(self, X, Y, X_aux,
                    params: Dict[str, jnp.ndarray],
                    z: jnp.ndarray,
                    return_components: bool = False,
                    debug_print: bool = False,
                    batch_idx: Optional[jnp.ndarray] = None) -> Union[float, Dict[str, float]]:
        """Exact mean‑field ELBO (constants dropped).

        Parameters
        ----------
        X : jnp.ndarray
            Gene expression matrix.
        Y : jnp.ndarray
            Binary outcomes.
        X_aux : jnp.ndarray
            Auxiliary covariates.
        params : Dict[str, jnp.ndarray]
            Current variational parameters.
        z : jnp.ndarray
            Current latent counts.
        return_components : bool, optional
            If True, return a dictionary with individual ELBO components
            instead of just the total ELBO.
        debug_print : bool, optional
            If True, print debug information. Default False to avoid duplicates.
        """

        digamma   = jsp.special.digamma
        gammaln   = jsp.special.gammaln

        if batch_idx is not None:
            a_theta = params['a_theta'][batch_idx]
            b_theta = params['b_theta'][batch_idx]
            a_xi = params['a_xi'][batch_idx]
            b_xi = params['b_xi'][batch_idx]
            zeta = params['zeta'][batch_idx]
        else:
            a_theta = params['a_theta']
            b_theta = params['b_theta']
            a_xi = params['a_xi']
            b_xi = params['b_xi']
            zeta = params['zeta']

        # ---------- Expectations ----------
        E_theta     = a_theta / b_theta          # (n,K)
        E_theta_sq  = (a_theta * (a_theta + 1)) / b_theta**2
        E_log_theta = digamma(a_theta) - jnp.log(b_theta)

        E_beta      = params['a_beta'] / params['b_beta']           # (p,K)
        E_log_beta  = digamma(params['a_beta'])  - jnp.log(params['b_beta'])

        # ---------- Poisson likelihood (observed data form) ----------
        # For augmented Poisson factorization: X_ij = Σ_ℓ z_ijℓ
        # The ELBO should use the observed data X, not latent counts z
        # Compute expected rate for observed data: λ_ij = Σ_ℓ E[θ_iℓ] E[β_jℓ]
        expected_rate = jnp.sum(E_theta[:, None, :] * E_beta[None, :, :], axis=2)  # (n, p)
        
        # Poisson likelihood using observed data X
        # log p(X_ij | λ_ij) = X_ij * log(λ_ij) - λ_ij - log(X_ij!)
        pois_term1 = jnp.sum(X * jnp.log(expected_rate + 1e-8))  # X_ij * log(λ_ij)
        pois_term2 = -jnp.sum(expected_rate)  # -λ_ij
        pois_term3 = -jnp.sum(jsp.special.gammaln(X + 1))  # -log(X_ij!)
        pois_ll = pois_term1 + pois_term2 + pois_term3
        
        # REMOVED: Debug prints moved to return_components section only
        # Add debug output to catch issues - only when requested
        # if debug_print:
        #     print("--- POISSON DEBUG ---")
        #     print(f"  E_log_theta -> min: {jnp.min(E_log_theta):.2f}, max: {jnp.max(E_log_theta):.2f}, mean: {jnp.mean(E_log_theta):.2f}")
        #     print(f"  E_log_beta  -> min: {jnp.min(E_log_beta):.2f}, max: {jnp.max(E_log_beta):.2f}, mean: {jnp.mean(E_log_beta):.2f}")
        #     print(f"  params['b_theta'] -> min: {jnp.min(params['b_theta']):.4f}, max: {jnp.max(params['b_theta']):.2f}, mean: {jnp.mean(params['b_theta']):.2f}")
        #     print(f"  params['b_beta']  -> min: {jnp.min(params['b_beta']):.4f}, max: {jnp.max(params['b_beta']):.2f}, mean: {jnp.mean(params['b_beta']):.2f}")
        #     print(f"  pois_term1 (z*E[log(rate)]): {pois_term1:.2f}")
        #     print(f"  pois_term2 (-E[rate])    : {pois_term2:.2f}")
        #     print(f"  Resulting pois_ll        : {pois_ll:.2f}")
        #     print("---------------------")
        # ---------- Logistic JJ bound --------------------------------
        psi     = E_theta @ params['mu_v'].T + X_aux @ params['mu_gamma'].T  # (n,κ)
        #zeta    = params['zeta']
        lam     = lambda_jj(zeta)                                            # (n,κ)

        # If τ²_v is a full covariance (κ, K, K), take its diagonal; else assume it is already (κ, K)
        if params['tau2_v'].ndim == 3:
            tau2_v_diag = jnp.diagonal(params['tau2_v'], axis1=1, axis2=2)  # (κ, K)
        else:
            tau2_v_diag = params['tau2_v']  # (κ, K)

        # full variance of ψ: Var[θ v_k] + Var[x_aux γ_k]
        var_theta_v = jnp.einsum('ik,ck->ic', E_theta_sq, tau2_v_diag)

        # tau2_gamma may be full covariance (kappa, d, d); use diagonals only
        if params['tau2_gamma'].ndim == 3:
            tau2_gamma_diag = jnp.diagonal(params['tau2_gamma'], axis1=1, axis2=2)
        else:
            tau2_gamma_diag = params['tau2_gamma']

        # Use matrix multiplication for variance of x_aux gamma
        var_x_aux_gamma = (X_aux**2) @ tau2_gamma_diag.T
        var_psi = var_theta_v + var_x_aux_gamma

        logit_term1 = jnp.sum((Y - 0.5) * psi)
        # Add bounds to prevent explosion
        tau2_v_bounded = jnp.clip(tau2_v_diag, 1e-8, 100.0)
        tau2_gamma_bounded = jnp.clip(tau2_gamma_diag, 1e-8, 100.0)
        E_theta_sq_bounded = jnp.clip(E_theta_sq, 1e-8, 1000.0)

        var_psi = jnp.einsum('ik,ck->ic', E_theta_sq_bounded, tau2_v_bounded) + \
                  (X_aux**2) @ tau2_gamma_bounded.T

        # Bound the total variance term
        total_var = jnp.clip(psi**2 + var_psi, 1e-8, 1e8)
        logit_term2 = -jnp.sum(lam * total_var)
        logit_term3 = jnp.sum(lam * zeta**2 - jnp.log(1.0 + jnp.exp(-zeta)) - zeta / 2.0)
        logit_ll = logit_term1 + logit_term2 + logit_term3
        
        # ---------- KL divergences (Gamma / Gaussian) ---------------
        def kl_gamma(a_q, b_q, a0, b0):
            """
            Computes KL(q||p) for two Gamma distributions.
            q ~ Gamma(a_q, b_q)
            p ~ Gamma(a0, b0)
            """
            
            # Term 1: (a_q - a_p) * E_q[log x]
            # E_q[log x] is digamma(a_q) - log(b_q)
            term1 = (a_q - a0) * digamma(a_q)

            # Term 2: The log-normalizer part
            term2 = gammaln(a0) - gammaln(a_q)

            # Term 3: The log of the rate parameters
            term3 = a0 * (jnp.log(b_q) - jnp.log(b0))
            
            # Term 4: The linear term
            # E_q[x] is a_q / b_q
            term4 = (a_q / b_q) * (b0 - b_q)

            # The KL divergence is the sum of these terms
            # It must be summed over all elements if the inputs are arrays
            kl_div = term1 + term2 + term3 + term4
            
            return kl_div

        E_log_theta = digamma(a_theta) - jnp.log(b_theta)
        E_theta = a_theta / b_theta
        E_xi = a_xi / b_xi
        
        # Need these for KL computation
        E_log_beta = digamma(params['a_beta']) - jnp.log(params['b_beta'])
        E_beta = params['a_beta'] / params['b_beta']
        E_eta = params['a_eta'] / params['b_eta']
        
        # FIXED: Use proper KL computation for theta (with hyperprior xi)
        # For each sample i, theta_i ~ Gamma(alpha_theta, xi_i)
        # So prior parameters are (alpha_theta, E[xi_i]) for each i
        E_xi_broadcast = jnp.broadcast_to(E_xi[:, None], a_theta.shape)  # (n, K)
        kl_theta = jnp.sum(kl_gamma(
            jnp.clip(a_theta, 1e-8, 1e6),  # q params (n, K)
            jnp.clip(b_theta, 1e-8, 1e6),  # q params (n, K)
            self.alpha_theta,                        # prior shape (scalar)
            jnp.clip(E_xi_broadcast, 1e-8, 1e6)    # prior rate (n, K)
        ))

        # FIXED: Use proper KL computation for beta (with hyperprior eta)  
        # For each gene j, beta_j ~ Gamma(alpha_beta, eta_j)
        # So prior parameters are (alpha_beta, E[eta_j]) for each j
        E_eta_broadcast = jnp.broadcast_to(E_eta[:, None], params['a_beta'].shape)  # (p, K)
        kl_beta = jnp.sum(kl_gamma(
            jnp.clip(params['a_beta'], 1e-8, 1e6),  # q params (p, K)
            jnp.clip(params['b_beta'], 1e-8, 1e6),  # q params (p, K)
            self.alpha_beta,                        # prior shape (scalar)
            jnp.clip(E_eta_broadcast, 1e-8, 1e6)   # prior rate (p, K)
        ))

        # Apply safety bounds for KL computation only
        kl_eta = kl_gamma(
            jnp.clip(params['a_eta'], 1e-8, 1e6), 
            jnp.clip(params['b_eta'], 1e-8, 1e6), 
            self.alpha_eta, self.lambda_eta
        ).sum()

        kl_xi = kl_gamma(
            jnp.clip(a_xi, 1e-8, 1e6),
            jnp.clip(b_xi, 1e-8, 1e6),
            self.alpha_xi, self.lambda_xi
        ).sum()

        # Fixed KL divergences with improved numerical stability
        # KL for v uses only diagonal elements of Σ_v to match mu_v shape
        # If any entries of tau2_v_diag are non‑positive, log() will produce
        # NaNs which propagate to the ELBO.  Apply a small lower bound for
        # numerical stability before computing the KL term.
        tau2_v_safe = jnp.clip(tau2_v_diag, 1e-8, 1e8)
        
        # Check for numerical issues in mu_v and apply gentle stabilization
        mu_v_safe = jnp.nan_to_num(params['mu_v'], nan=0.0, posinf=50.0, neginf=-50.0)
        if jnp.any(jnp.isnan(params['mu_v'])) or jnp.any(jnp.isinf(params['mu_v'])):
            print("WARNING KL_V: mu_v contains NaN/Inf, using safe values")
        
        # Check for extreme values that could cause overflow
        max_mu_v = jnp.max(jnp.abs(mu_v_safe))
        if max_mu_v > 100.0:
            scale_factor = 100.0 / max_mu_v
            print(f"WARNING KL_V: Large mu_v values (max={max_mu_v:.2e}), scaling by {scale_factor:.3f}")
            mu_v_safe = mu_v_safe * scale_factor

        print(f"DEBUG KL_V: mu_v_safe range: {jnp.min(mu_v_safe):.2e} to {jnp.max(mu_v_safe):.2e}")
        print(f"DEBUG KL_V: tau2_v_safe range: {jnp.min(tau2_v_safe):.2e} to {jnp.max(tau2_v_safe):.2e}")
        print(f"DEBUG KL_V: sigma2_v={self.sigma2_v}")
        
        # Compute KL divergence terms with numerical safety
        # Term 1: (μ² + τ²) / σ²
        term1_raw = (mu_v_safe ** 2 + tau2_v_safe) / self.sigma2_v
        # Clip to prevent overflow in summation
        term1 = jnp.clip(term1_raw, 0, 1e6)
        if jnp.max(term1_raw) > 1e6:
            print(f"WARNING KL_V: Large term1 values (max={jnp.max(term1_raw):.2e}), clipping to prevent overflow")
        
        # Term 2: -log(τ²)
        log_tau2_v = jnp.log(tau2_v_safe)
        term2 = -log_tau2_v
        
        # Check for problematic log values
        if jnp.any(log_tau2_v < -20):
            print(f"WARNING KL_V: Very small tau2_v values, min log(tau2)={jnp.min(log_tau2_v):.2e}")
        if jnp.any(log_tau2_v > 20):
            print(f"WARNING KL_V: Very large tau2_v values, max log(tau2)={jnp.max(log_tau2_v):.2e}")
        
        # Term 3: log(σ²)
        term3 = jnp.log(self.sigma2_v)
        
        # Term 4: -1 (constant)
        term4 = -1
        
        print(f"DEBUG KL_V: term1 range: {jnp.min(term1):.2e} to {jnp.max(term1):.2e}")
        print(f"DEBUG KL_V: term2 range: {jnp.min(term2):.2e} to {jnp.max(term2):.2e}")
        print(f"DEBUG KL_V: term3: {term3:.2e}")
        print(f"DEBUG KL_V: term4: {term4:.2e}")
        
        # Compute final KL with overflow protection
        kl_components = term1 + term2 + term3 + term4
        
        # Check for numerical issues before summation
        if jnp.any(jnp.isnan(kl_components)):
            print("ERROR KL_V: KL components contain NaN!")
            kl_components = jnp.nan_to_num(kl_components, nan=0.0)
        if jnp.any(jnp.isinf(kl_components)):
            print("ERROR KL_V: KL components contain Inf!")
            kl_components = jnp.nan_to_num(kl_components, posinf=1e6, neginf=-1e6)
            
        kl_v = 0.5 * jnp.sum(kl_components)
        
        # Final safety check
        if jnp.isnan(kl_v) or jnp.isinf(kl_v):
            print(f"ERROR KL_V: Final kl_v is {kl_v}, using fallback value")
            kl_v = 1e6  # Large but finite penalty
        
        print(f"DEBUG KL_V: final kl_v = {kl_v:.2e}")
        
        if params['tau2_gamma'].ndim == 3:  # Full covariance (kappa, d, d)
            tau2_gamma_diag = jnp.diagonal(params['tau2_gamma'], axis1=1, axis2=2)  # (kappa, d)
        else:  # Already diagonal (kappa, d)
            tau2_gamma_diag = params['tau2_gamma']

        kl_gamma_coef = 0.5 * jnp.sum(
            (params['mu_gamma']**2 + tau2_gamma_diag) / self.sigma2_gamma - 
            jnp.log(tau2_gamma_diag) + jnp.log(self.sigma2_gamma) - 1
        )
        
        # Add numerical bounds check
        if jnp.any(jnp.isnan(kl_v)) or jnp.any(jnp.isinf(kl_v)):
            print("WARNING: KL_v contains NaN/Inf values!")
            print(f"  Original kl_v: {kl_v}")
            print(f"  mu_v has NaN: {jnp.any(jnp.isnan(params['mu_v']))}")
            print(f"  tau2_v_diag has NaN: {jnp.any(jnp.isnan(tau2_v_diag))}")
            print(f"  tau2_v_safe has NaN: {jnp.any(jnp.isnan(tau2_v_safe))}")
            kl_v = jnp.nan_to_num(kl_v, nan=1e6, posinf=1e6, neginf=-1e6)
        
        kl_total = kl_theta + kl_beta + kl_eta + kl_xi + kl_v + kl_gamma_coef

        elbo = pois_ll + logit_ll - kl_total
        
        # REMOVED: Debug prints moved to return_components section only
        # Add debug output for KL terms
        # if debug_print:
        #     print("DEBUG KL Terms:")
        #     print(f"  kl_theta: {kl_theta}")
        #     print(f"  kl_beta: {kl_beta}")
        #     print(f"  kl_eta: {kl_eta}")
        #     print(f"  kl_xi: {kl_xi}")
        #     print(f"  kl_v: {kl_v}")
        #     print(f"  kl_gamma_coef: {kl_gamma_coef}")
        #     print(f"  kl_total: {kl_total}")

        if return_components:
            # Debug prints (only when components are requested)
            if debug_print:
                print("--- POISSON DEBUG ---")
                print(f"  E_log_theta -> min: {E_log_theta.min():.2f}, max: {E_log_theta.max():.2f}, mean: {E_log_theta.mean():.2f}")
                print(f"  E_log_beta  -> min: {E_log_beta.min():.2f}, max: {E_log_beta.max():.2f}, mean: {E_log_beta.mean():.2f}")
                print(f"  params['b_theta'] -> min: {params['b_theta'].min():.4f}, max: {params['b_theta'].max():.2f}, mean: {params['b_theta'].mean():.2f}")
                print(f"  params['b_beta']  -> min: {params['b_beta'].min():.4f}, max: {params['b_beta'].max():.2f}, mean: {params['b_beta'].mean():.2f}")
                print(f"  pois_term1 (z*E[log(rate)]): {pois_term1:.2f}")
                print(f"  pois_term2 (-E[rate])    : {pois_term2:.2f}")
                print(f"  Resulting pois_ll        : {pois_ll:.2f}")
                print("---------------------")
            if debug_print:
                print(f"  var_psi range: {jnp.min(var_psi):.2e} to {jnp.max(var_psi):.2e}")
                print(f"  lam range: {jnp.min(lam):.2e} to {jnp.max(lam):.2e}")
                print(f"  psi^2 range: {jnp.min(psi**2):.2e} to {jnp.max(psi**2):.2e}")
            if debug_print:
                print("--- LOGISTIC DEBUG ---")
                print(f"  logit_term1 ((y-0.5)E[A])  : {logit_term1:.2f}")
                print(f"  logit_term2 (-lam*E[A^2])  : {logit_term2:.2f}")
                print(f"  logit_term3 (C(zeta) only) : {logit_term3:.2f}")
                print(f"  Resulting logit_ll       : {logit_ll:.2f}")
                print("------------------------")
            if debug_print:
                print("DEBUG KL Terms:")
                print(f"  kl_theta: {kl_theta}")
                print(f"  kl_beta: {kl_beta}")
                print(f"  kl_eta: {kl_eta}")
                print(f"  kl_xi: {kl_xi}")
                print(f"  kl_v: {kl_v}")
                print(f"  kl_gamma_coef: {kl_gamma_coef}")
                print(f"  kl_total: {kl_total}")
            
            components = {
                "pois_ll": float(pois_ll),
                "logit_ll": float(logit_ll),
                "kl_theta": float(kl_theta),
                "kl_beta": float(kl_beta),
                "kl_eta": float(kl_eta),
                "kl_xi": float(kl_xi),
                "kl_v": float(kl_v),
                "kl_gamma": float(kl_gamma_coef),
                "elbo": float(elbo),
            }

            print("=== ELBO Summary ===")
            for key, value in components.items():
                print(f"  {key:<12}: {value:,.2f}")
            print("====================")
            return components

        return float(elbo)

    def infer_theta_for_new_samples(self, X_new, x_aux_new, global_params, n_iter=20):
        """
        Infer E_theta for new samples given trained global parameters.
        Only theta (and optionally xi) are updated; all other parameters are fixed.
        SIMPLIFIED VERSION WITHOUT COMPLEX LOGISTIC TERMS.
        """
        import jax.numpy as jnp
        n_new = X_new.shape[0]
        K = self.K
        kappa = self.kappa
        
        # FIXED: Use better initialization for theta
        sample_totals = jnp.sum(X_new, axis=1)
        theta_init = jnp.outer(sample_totals / jnp.mean(sample_totals), jnp.ones(K)) * 3.0  # Increased scale
        theta_init = jnp.maximum(theta_init, 1.0)  # Higher minimum
        theta_var = theta_init * 0.8  # Increased variance
        a_theta = (theta_init**2) / theta_var
        b_theta = theta_init / theta_var
        
        # FIXED: Use better initialization for xi
        a_xi = jnp.full(n_new, self.alpha_xi + K * self.alpha_theta)
        b_xi = self.lambda_xi + jnp.sum(a_theta / b_theta, axis=1)
        b_xi = jnp.maximum(b_xi, 1e-8)
        
        # Use global parameters for beta, eta, v, gamma, etc.
        a_beta = jnp.array(global_params['a_beta'])
        b_beta = jnp.array(global_params['b_beta'])
        mu_v = jnp.array(global_params['mu_v'])
        mu_gamma = jnp.array(global_params['mu_gamma'])
        
        # FIXED: Use better zeta initialization
        E_theta = a_theta / b_theta
        A_init = jnp.sum(jnp.expand_dims(E_theta, 1) * jnp.expand_dims(mu_v, 0), axis=2) + \
                x_aux_new @ mu_gamma.T
        zeta = jnp.abs(A_init) + 0.5  # Larger offset
        
        for it in range(n_iter):
            # E_beta, E_v, E_gamma
            E_beta = a_beta / b_beta
            E_theta = a_theta / b_theta
            
            # Update z for new samples
            rates = jnp.expand_dims(E_theta, 1) * jnp.expand_dims(E_beta, 0)
            total_rates = jnp.sum(rates, axis=2, keepdims=True)
            probs = rates / (total_rates + 1e-8)
            z = jnp.expand_dims(X_new, 2) * probs
            
            # Update xi for new samples
            a_xi_new = self.alpha_xi + K * self.alpha_theta
            b_xi_new = self.lambda_xi + jnp.sum(E_theta, axis=1)
            b_xi_new = jnp.maximum(b_xi_new, 1e-8)
            a_xi = jnp.full(n_new, a_xi_new)
            b_xi = b_xi_new
            
            # FIXED: Simplified theta update without complex logistic terms
            a_theta_new = self.alpha_theta + jnp.sum(z, axis=1)
            b_theta_new = jnp.expand_dims(b_xi, 1) + jnp.sum(E_beta, axis=0)
            b_theta_new = jnp.maximum(b_theta_new, 1e-8)
            
            a_theta = a_theta_new
            b_theta = b_theta_new
            
            # FIXED: Update zeta with better values
            E_theta = a_theta / b_theta
            A = jnp.sum(jnp.expand_dims(E_theta, 1) * jnp.expand_dims(mu_v, 0), axis=2) + \
                x_aux_new @ mu_gamma.T
            zeta = jnp.abs(A) + 0.5  # Larger offset
            
        return a_theta / b_theta  # E_theta for new samples


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
        metrics["per_class_metrics"] = per_class  # type: ignore

    metrics["probabilities"] = probs.tolist()
    return metrics


def run_model_and_evaluate(
    x_data,
    x_aux,
    y_data,
    var_names,
    hyperparams,
    seed=None,
    test_size=0.15,
    val_size=0.15,
    max_iters=100,
    return_probs=True,
    sample_ids=None,
    mask=None,
    scores=None,
    plot_elbo=False,
    plot_prefix=None,
    return_params=False,
    verbose=False,
):
    """Fit the model on training data and evaluate on splits (no data leakage)."""

    # If seed is None, use a valid random integer seed for sklearn
    if seed is None:
        seed = np.random.randint(0, 2**32)

    if y_data.ndim == 1:
        y_data = y_data.reshape(-1, 1)
    if x_aux.ndim == 1:
        x_aux = x_aux.reshape(-1, 1)

    n_samples, n_genes = x_data.shape
    kappa = y_data.shape[1]
    d = hyperparams.get("d", 1)

    # Split indices BEFORE fitting
    indices = np.arange(n_samples)
    train_idx, temp_idx = train_test_split(
        indices, test_size=val_size + test_size, random_state=seed
    )
    val_rel = test_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=val_rel, random_state=seed
    )

    # Fit model ONLY on training data
    print(f"DEBUG: Creating SupervisedPoissonFactorization model with n_samples={len(train_idx)}, n_genes={n_genes}, d={d}, kappa={kappa}")
    model = SupervisedPoissonFactorization(
        len(train_idx),
        n_genes,
        n_factors=d,  # Fix parameter name
        n_outcomes=kappa,  # Fix parameter name
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
    print(f"DEBUG: Model created successfully, about to call model.fit with max_iters={max_iters}, verbose={verbose}")

    params, expected = model.fit(x_data[train_idx], y_data[train_idx], x_aux[train_idx], n_iter=max_iters, verbose=verbose, mask=mask)
    print(f"DEBUG: Model fitting completed successfully")

    # Compute probabilities for training set
    all_probs_train = logistic(
        expected["E_theta"] @ params["mu_v"].T + x_aux[train_idx] @ params["mu_gamma"].T
    )

    # Infer E_theta for val and test splits using the trained global parameters
    E_theta_val = model.infer_theta_for_new_samples(x_data[val_idx], x_aux[val_idx], params, n_iter=20)
    all_probs_val = logistic(
        E_theta_val @ params["mu_v"].T + x_aux[val_idx] @ params["mu_gamma"].T
    )
    E_theta_test = model.infer_theta_for_new_samples(x_data[test_idx], x_aux[test_idx], params, n_iter=20)
    all_probs_test = logistic(
        E_theta_test @ params["mu_v"].T + x_aux[test_idx] @ params["mu_gamma"].T
    )

    train_metrics = _compute_metrics(y_data[train_idx], np.array(all_probs_train))
    val_metrics = _compute_metrics(y_data[val_idx], np.array(all_probs_val))
    test_metrics = _compute_metrics(y_data[test_idx], np.array(all_probs_test))

    results = {
        "train_metrics": {k: v for k, v in train_metrics.items() if k != "probabilities"},
        "val_metrics": {k: v for k, v in val_metrics.items() if k != "probabilities"},
        "test_metrics": {k: v for k, v in test_metrics.items() if k != "probabilities"},
        "hyperparameters": hyperparams,
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
    if plot_elbo and 'elbo_history' in params and len(params['elbo_history']) > 0:
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            elbo_iterations = list(range(1, len(params['elbo_history']) + 1))
            plt.plot(elbo_iterations, params['elbo_history'], 
                    'b-', linewidth=2, label="ELBO")
            plt.xlabel("Iteration")
            plt.ylabel("Evidence Lower Bound (ELBO)")
            plt.title("ELBO Convergence During VI Training")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot to file if plot_prefix is provided
            if plot_prefix is not None:
                import os
                os.makedirs(plot_prefix, exist_ok=True)
                plot_path = os.path.join(plot_prefix, "vi_elbo_convergence.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"ELBO plot saved to {plot_path}")
                
                # Also save as PDF for publications
                pdf_path = os.path.join(plot_prefix, "vi_elbo_convergence.pdf")
                plt.savefig(pdf_path, bbox_inches='tight')
            
            # Store plot data in results
            results["elbo_iterations"] = elbo_iterations
            results["elbo_values"] = params['elbo_history']
            
            plt.close()  # Close to free memory
            
        except Exception as e:
            print(f"Warning: Could not create ELBO plot: {e}")

    return results