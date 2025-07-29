import os
import json
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Force JAX to use CPU only - must be set before importing jax
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp

import gseapy
from gseapy import read_gmt
import mygene  
import argparse
import gc
import psutil  
import pickle  # Add pickle for caching results
import random  # Import for random sampling of pathways
import gzip # Import for gzipping files

from memory_tracking import get_memory_usage, log_memory, log_array_sizes, clear_memory

# Log initial memory
print(f"Initial memory usage: {get_memory_usage():.2f} MB")

from vi_model_complete import SupervisedPoissonFactorization, run_model_and_evaluate
from svi import run_model_and_evaluate_corrected

# Alias for the VI function
run_vi = run_model_and_evaluate
from data import *
from data import prepare_and_load_emtab_smote

def run_svi(x_data, x_aux, y_data, var_names, hyperparams, seed=None, test_size=0.15, val_size=0.15, max_iters=1000, return_probs=True, sample_ids=None, mask=None, scores=None, plot_elbo=False, plot_prefix=None, return_params=False, verbose=False, early_stopping=True, patience=50, min_delta=1e-4, beta_init=None):
    """
    Wrapper function to run the corrected SVI implementation.
    """
    return run_model_and_evaluate_corrected(
        x_data=x_data,
        x_aux=x_aux,
        y_data=y_data,
        var_names=var_names,
        hyperparams=hyperparams,
        seed=seed,
        test_size=test_size,
        val_size=val_size,
        max_iters=max_iters,
        return_probs=return_probs,
        sample_ids=sample_ids,
        mask=mask,
        scores=scores,
        plot_elbo=plot_elbo,
        plot_prefix=plot_prefix,
        return_params=return_params,
        verbose=verbose,
        early_stopping=early_stopping,
        patience=patience,
        min_delta=min_delta,
        beta_init=beta_init
    )

def scale_to_reasonable_counts(X, target_max=20):
    """Scale data to reasonable count range while preserving structure"""
    # Much more aggressive scaling for stability
    max_val = np.max(X)
    print(f"Original data: max={max_val:.1f}, median non-zero={np.median(X[X > 0]):.1f}")
    
    if max_val > target_max:
        # Scale so that max value becomes target_max
        scale_factor = target_max / max_val
        X_scaled = np.round(X * scale_factor).astype(int)
        # Ensure at least some non-zero values
        X_scaled = np.maximum(X_scaled, (X > 0).astype(int))
        print(f"Scaled data: max={np.max(X_scaled):.1f}, median non-zero={np.median(X_scaled[X_scaled > 0]):.1f}, scale_factor={scale_factor:.6f}")
        return X_scaled
    else:
        return X.astype(int)

from sklearn.model_selection import train_test_split

def custom_train_test_split(*arrays, test_size=0.15, val_szie=0.15, random_state=None):
    n_samples = len(arrays[0])
    indices = np.arange(n_samples)

    remaining_size = test_size + val_szie
    
    arrays_and_indices = list(arrays) + [indices]
    split1_results = train_test_split(*arrays_and_indices, test_size=remaining_size, random_state=random_state)
    
    train_parts = split1_results[0::2]
    temp_parts = split1_results[1::2]

    relative_test_size = test_size / remaining_size

    split2_results = train_test_split(*temp_parts, test_size=relative_test_size, random_state=random_state)
    val_parts = split2_results[0::2]
    test_parts = split2_results[1::2]

    final_result = []
    num_arrays = len(arrays)
    for i in range(num_arrays):
        final_result.append(train_parts[i])
        final_result.append(val_parts[i])
        final_result.append(test_parts[i])
    
    return final_result

def save_split_results(results, y_data, sample_ids, split_indices, split_name, output_dir, prefix):
    # y_data: (n_samples, n_labels)
    # sample_ids: list of sample ids
    # split_indices: indices for this split
    # split_name: 'train', 'val', 'test'
    # results: dict from run_model_and_evaluate
    # prefix: file prefix for output
    
    probs = np.array(results[f"{split_name}_probabilities"])
    y_true = y_data[split_indices]
    sample_id_list = [sample_ids[i] for i in split_indices]
    preds = (probs >= 0.5).astype(int)
    n_labels = y_true.shape[1]
    data = []
    for i, idx in enumerate(split_indices):
        row = {
            "sample_id": sample_id_list[i],
        }
        # True labels, predicted probs, predicted labels
        for k in range(n_labels):
            row[f"true_label_{k+1}"] = y_true[i, k]
            row[f"pred_prob_{k+1}"] = probs[i, k]
            row[f"pred_label_{k+1}"] = preds[i, k]
        data.append(row)
    df = pd.DataFrame(data)
    out_path = os.path.join(output_dir, f"{prefix}_{split_name}_results.csv.gz")
    df.to_csv(out_path, index=False, compression='gzip')


def save_beta_matrix_csv(E_beta, gene_names, row_names, mu_v, out_path):
    """
    Save the E_beta matrix as a CSV (gzip) with pathways/programs as rows and genes as columns.
    Add v columns (one per label) for each program/pathway.
    """
    n_programs = E_beta.shape[1]
    n_labels = mu_v.shape[0] if mu_v is not None else 1
    data = []
    for k in range(n_programs):
        row = {"name": row_names[k]}
        # Add v columns per label
        if mu_v is not None:
            for l in range(n_labels):
                row[f"v_{l+1}"] = mu_v[l, k]
        # Add gene contributions
        for g, gene in enumerate(gene_names):
            row[gene] = E_beta[g, k]
        data.append(row)
    df = pd.DataFrame(data)
    df.to_csv(out_path, index=False, compression="gzip")
# def run_all_experiments(datasets, hyperparams_map, output_dir="/labs/Aguiar/SSPA_BRAY/BRay/SVIResults/unmasked", seed=None, mask=None, max_iter=100, pathway_names=None, run_fn=run_vi):
def run_all_experiments(datasets, hyperparams_map, output_dir="/mnt/research/aguiarlab/proj/SSPA-BRAY/BRay/SVIResults/unmasked", seed=None, mask=None, max_iter=100, pathway_names=None, run_fn=run_vi):
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for dataset_name, (adata, label_col) in datasets.items():
        print(f"\nRunning experiment on dataset {dataset_name}, label={label_col}")
        
        if dataset_name == "emtab":
            # For EMTAB, now using binary disease classification
            label_names = [label_col]  # label_col should be "disease_binary"
            Y = adata.obs[label_col].values.astype(float).reshape(-1, 1)
            X = adata.X
            var_names = list(adata.var_names)
            
            # For EMTAB, x_aux includes both age and sex_female columns
            x_aux = adata.obs[['age', 'sex_female']].values.astype(float)
            sample_ids = adata.obs.index.tolist()
        elif dataset_name == "thyroid":
            # For Thyroid Cancer dataset, Y contains Clinical_History column
            label_names = [label_col]  # label_col should be "Clinical_History"
            Y = adata.obs[label_col].values.astype(float).reshape(-1, 1)
            X = adata.X
            var_names = list(adata.var_names)
            
            # For Thyroid, x_aux includes both Age and sex_female columns
            x_aux = adata.obs[['Age', 'sex_female']].values.astype(float)
            sample_ids = adata.obs.index.tolist()
        elif dataset_name == "sim":
            # For Simulated dataset, Y contains disease column (real responses only)
            label_names = [label_col]  # label_col should be "disease"
            Y = adata.obs[label_col].values.astype(float).reshape(-1, 1)
            X = adata.X
            var_names = list(adata.var_names)
            
            # For simulation, x_aux includes aux data (like gender, population, etc.)
            # Use all available numeric columns except the label column as auxiliary features
            aux_cols = [col for col in adata.obs.columns if col != label_col and adata.obs[col].dtype in ['int64', 'float64']]
            if aux_cols:
                x_aux = adata.obs[aux_cols].values.astype(float)
            else:
                x_aux = np.ones((X.shape[0], 1))  # fallback to intercept only
            sample_ids = adata.obs.index.tolist()
        else:
            # For AJM datasets (ajm_cyto, ajm_ap), use original logic
            label_names = [label_col]
            Y = adata.obs[label_col].values.astype(float).reshape(-1,1) 
            X = adata.X  
            var_names = list(adata.var_names)

            x_aux = np.ones((X.shape[0],1)) 
            sample_ids = adata.obs.index.tolist()
        
        scores = None
        if 'cyto_seed_score' in adata.obs:
            scores = adata.obs['cyto_seed_score'].values
            print(f"Found cyto_seed_score in dataset with mean value: {np.mean(scores):.4f}")

        hyperparams = hyperparams_map[dataset_name].copy()  
        d_values = hyperparams.pop("d")
        # Convert to list if it's a single value
        if not isinstance(d_values, list):
            d_values = [d_values]
        for d in d_values:
            print(f"Running with d={d}")
            
            hyperparams["d"] = d
            
            if mask is not None:
                print(f"Using mask with shape: {mask.shape}")
                log_array_sizes({'mask': mask})
            
            clear_memory()

            test_split_size = 0.15
            val_split_size = 0.15
            
            try:
                print(f"DEBUG: About to call run_model_and_evaluate with X.shape={X.shape}, Y.shape={Y.shape}")
                print(f"DEBUG: x_aux.shape={x_aux.shape}, hyperparams['d']={hyperparams['d']}")
                print(f"DEBUG: max_iter={max_iter}")

                if run_fn == run_svi:
                    hyperparams['enable_param_logging'] = True
                    hyperparams['log_every_n_iters'] = 5
                    # Add SVI-specific convergence parameters
                    hyperparams['convergence_window'] = 5  # Smaller window for faster convergence detection
                    hyperparams['min_delta'] = 5e-3        # Less strict convergence criterion
                    hyperparams['patience'] = 20           # Reduced patience for faster stopping
                    hyperparams['learning_rate'] = 0.005
                    hyperparams['batch_size'] = 48          # More conservative learning rate
                results = run_fn(
                    x_data=X,
                    x_aux=x_aux,
                    y_data=Y,
                    var_names=var_names,
                    hyperparams=hyperparams,
                    seed=seed,
                    test_size=test_split_size,
                    val_size=val_split_size,
                    max_iters=max_iter,
                    return_probs=True,
                    sample_ids=sample_ids,
                    mask=mask,
                    scores=scores,
                    return_params=True,
                    verbose=True,
                    plot_elbo=True,
                    plot_prefix=os.path.join(output_dir, "elbo_trace")
                )

                # Attach label names to results for summary printing
                results["label_names"] = label_names
                print(f"DEBUG: run_model_and_evaluate completed successfully")

                if "error" in results:
                    print(f"Skipping post-processing for d={d} due to training error.")
                    all_results[f"{dataset_name}_{label_col}_d_{d}"] = results
                    continue 
            
                # Reconstruct split indices
                n_samples = X.shape[0]
                indices = np.arange(n_samples)
                train_idx, temp_idx = train_test_split(indices, test_size=val_split_size + test_split_size, random_state=seed if seed is not None else 0)
                val_rel = test_split_size / (val_split_size + test_split_size)
                val_idx, test_idx = train_test_split(temp_idx, test_size=val_rel, random_state=seed if seed is not None else 0)

                # Save split results
                save_split_results(results, Y, sample_ids, train_idx, "train", output_dir, f"{dataset_name}_{label_col}_d_{d}")
                save_split_results(results, Y, sample_ids, val_idx, "val", output_dir, f"{dataset_name}_{label_col}_d_{d}")
                save_split_results(results, Y, sample_ids, test_idx, "test", output_dir, f"{dataset_name}_{label_col}_d_{d}")

                main_results = results.copy()
                if "alpha_beta" in main_results:
                    del main_results["alpha_beta"]
                if "omega_beta" in main_results:
                    del main_results["omega_beta"]
                if "top_genes" in main_results:
                    del main_results["top_genes"]
                
                results_json_path = os.path.join(output_dir, f"{dataset_name}_{label_col}_d_{d}_results_with_scores.json.gz")
                with gzip.open(results_json_path, "wt", encoding="utf-8") as f:
                    json.dump(main_results, f, indent=2)
                    
                all_results[f"{dataset_name}_{label_col}_d_{d}"] = results

                # Save E_beta matrix as CSV.gz with v columns
                a_beta = np.array(results["a_beta"])
                b_beta = np.array(results["b_beta"])
                E_beta = a_beta / np.maximum(b_beta, 1e-10)
                mu_v = np.array(results["mu_v"]) if "mu_v" in results else None
                if mask is not None:
                    row_names = pathway_names if pathway_names is not None else [f"pathway_{i+1}" for i in range(E_beta.shape[1])]
                    out_path = os.path.join(output_dir, f"pathway_contributions.csv.gz")
                else:
                    row_names = [f"DRGP{i+1}" for i in range(E_beta.shape[1])]
                    out_path = os.path.join(output_dir, f"gene_programs.csv.gz")
                save_beta_matrix_csv(E_beta, var_names, row_names, mu_v, out_path)

            except Exception as e:
                print(f"--- UNHANDLED EXCEPTION for d={d} ---")
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                all_results[f"{dataset_name}_{label_col}_d_{d}"] = {"error": str(e), "status": "crashed"}
            clear_memory()

    return all_results

def run_combined_gp_and_pathway_experiment(dataset_name, adata, label_col, mask, pathway_names, n_gp=500,
                                  output_dir="/mnt/research/aguiarlab/proj/SSPA-BRAY/BRay/SVIResults/combined",
                                  seed=None, max_iter=100, run_fn=run_vi):
    print(f"\nRunning combined pathway+GP experiment on {dataset_name}, label={label_col}, with {n_gp} additional gene programs")
    
    # Prepare label matrix (Y) and auxiliary matrix (x_aux) depending on the
    # dataset and requested label column.  The EMTAB dataset now uses binary
    # disease classification (0 = no disease, 1 = any disease).
    if dataset_name == "emtab":
        Y = adata.obs[label_col].values.astype(float).reshape(-1, 1)
        x_aux = adata.obs[["age", "sex_female"]].values.astype(float)
    elif dataset_name == "thyroid":
        Y = adata.obs[label_col].values.astype(float).reshape(-1, 1)
        x_aux = adata.obs[["Age", "sex_female"]].values.astype(float)
    elif dataset_name == "sim":
        Y = adata.obs[label_col].values.astype(float).reshape(-1, 1)
        # For simulation, use all available numeric columns except the label column as auxiliary features
        aux_cols = [col for col in adata.obs.columns if col != label_col and adata.obs[col].dtype in ['int64', 'float64']]
        if aux_cols:
            x_aux = adata.obs[aux_cols].values.astype(float)
        else:
            x_aux = np.ones((adata.shape[0], 1))  # fallback to intercept only
    else:
        Y = adata.obs[label_col].values.astype(float).reshape(-1, 1)
        x_aux = np.ones((adata.shape[0], 1))

    X = adata.X
    var_names = list(adata.var_names)
    sample_ids = adata.obs.index.tolist()
    
    log_array_sizes({
        'X': X,
        'Y': Y,
        'x_aux': x_aux,
        'mask': mask
    })
    
    scores = None
    if 'cyto_seed_score' in adata.obs:
        scores = adata.obs['cyto_seed_score'].values
        print(f"Found cyto_seed_score in dataset with mean value: {np.mean(scores):.4f}")

    hyperparams = {
        "alpha_eta": 2.0, "lambda_eta": 3.0,
        "alpha_beta": 0.6,
        "alpha_xi": 2.0, "lambda_xi": 3.0,
        "alpha_theta": 0.6,
        "sigma2_v": 1.0, "sigma2_gamma": 1.0
    }
    
    n_pathways = mask.shape[1]
    total_d = n_pathways + n_gp
    hyperparams["d"] = total_d
    
    print(f"Total dimensions: {total_d} = {n_pathways} pathways + {n_gp} gene programs")
    
    n_genes = mask.shape[0]
    extended_mask = np.zeros((n_genes, total_d))
    extended_mask[:, :n_pathways] = mask
    extended_mask[:, n_pathways:] = 1
    
    print(f"Extended mask shape: {extended_mask.shape}")
    print(f"Original mask columns: {n_pathways}, Additional unmasked columns: {n_gp}")
    
    # Use the provided output_dir (timestamped directory) directly
    exp_output_dir = output_dir 
    os.makedirs(exp_output_dir, exist_ok=True) # Ensure it exists, harmless if already there
    
    # Create plot prefix for ELBO plots, ensuring filename uniqueness
    plot_prefix_basename = f"combined_{dataset_name}_pw{n_pathways}_gp{n_gp}"
    plot_prefix = os.path.join(exp_output_dir, plot_prefix_basename)
    
    clear_memory()
    try:
        results = run_fn(
            x_data=X,
            x_aux=x_aux,
            y_data=Y,
            var_names=var_names,
            hyperparams=hyperparams,
            seed=seed,
            test_size=0.15,
            val_size=0.15,
            max_iters=max_iter,
            return_probs=True,
            sample_ids=sample_ids,
            mask=extended_mask,
            scores=scores,
            return_params=True,
            plot_elbo=True,
            plot_prefix=os.path.join(exp_output_dir, "elbo_trace")
        )
        
        # Reconstruct split indices
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        train_idx, temp_idx = train_test_split(indices, test_size=0.15 + 0.15, random_state=seed if seed is not None else 0)
        val_rel = 0.15 / (0.15 + 0.15)
        val_idx, test_idx = train_test_split(temp_idx, test_size=val_rel, random_state=seed if seed is not None else 0)

        # Save split results
        save_split_results(results, Y, sample_ids, train_idx, "train", exp_output_dir, f"{dataset_name}_{label_col}_combined_pw{n_pathways}_gp{n_gp}")
        save_split_results(results, Y, sample_ids, val_idx, "val", exp_output_dir, f"{dataset_name}_{label_col}_combined_pw{n_pathways}_gp{n_gp}")
        save_split_results(results, Y, sample_ids, test_idx, "test", exp_output_dir, f"{dataset_name}_{label_col}_combined_pw{n_pathways}_gp{n_gp}")

        main_results = results.copy()
        for large_field in ["alpha_beta", "omega_beta", "E_beta"]:
            if large_field in main_results:
                del main_results[large_field]
        
        results_json_filename = f"{dataset_name}_{label_col}_combined_pw{n_pathways}_gp{n_gp}_results.json.gz"
        out_path = os.path.join(exp_output_dir, results_json_filename)
        with gzip.open(out_path, "wt", encoding="utf-8") as f:
            json.dump(main_results, f, indent=2)
        
        # Save E_beta matrix as CSV.gz with v columns
        a_beta = np.array(results["a_beta"])
        b_beta = np.array(results["b_beta"])
        E_beta = a_beta / np.maximum(b_beta, 1e-10)
        combined_row_names = pathway_names.copy() if pathway_names else [f"pathway_{i+1}" for i in range(n_pathways)]
        combined_row_names.extend([f"DRGP{i+1}" for i in range(n_gp)])
        out_path = os.path.join(exp_output_dir, "pathway_DRGP.csv.gz")
        mu_v = np.array(results["mu_v"]) if "mu_v" in results else None
        save_beta_matrix_csv(E_beta, var_names, combined_row_names, mu_v, out_path)

    except Exception as e:
        print(f"--- UNHANDLED EXCEPTION in combined experiment ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        results = {"error": str(e), "status": "crashed"}
        
        return results

def run_pathway_initialized_experiment(dataset_name, adata, label_col, mask, pathway_names,
                                output_dir="/mnt/research/aguiarlab/proj/SSPA-BRAY/BRay/SVIResults/pathway_initiated",
                                seed=None, max_iter=100, run_fn=run_vi):
    """
    Run pathway-initialized experiment.
    Results will be saved directly into the provided output_dir (timestamped directory).
    """
    print(f"\nRunning pathway-initialized experiment on {dataset_name}, label={label_col}")
    print(f"This will initialize gene programs with pathway information, then let them evolve freely")
    
    # Prepare Y and x_aux similarly to the combined experiment.  The EMTAB dataset
    # now uses binary disease classification (0 = no disease, 1 = any disease).
    if dataset_name == "emtab":
        Y = adata.obs[label_col].values.astype(float).reshape(-1, 1)
        x_aux = adata.obs[["age", "sex_female"]].values.astype(float)
    elif dataset_name == "thyroid":
        Y = adata.obs[label_col].values.astype(float).reshape(-1, 1)
        x_aux = adata.obs[["Age", "sex_female"]].values.astype(float)
    elif dataset_name == "sim":
        Y = adata.obs[label_col].values.astype(float).reshape(-1, 1)
        # For simulation, use all available numeric columns except the label column as auxiliary features
        aux_cols = [col for col in adata.obs.columns if col != label_col and adata.obs[col].dtype in ['int64', 'float64']]
        if aux_cols:
            x_aux = adata.obs[aux_cols].values.astype(float)
        else:
            x_aux = np.ones((adata.shape[0], 1))  # fallback to intercept only
    else:
        Y = adata.obs[label_col].values.astype(float).reshape(-1, 1)
        x_aux = np.ones((adata.shape[0], 1))

    X = adata.X
    var_names = list(adata.var_names)
    sample_ids = adata.obs.index.tolist()
    
    log_array_sizes({
        'X': X,
        'Y': Y,
        'x_aux': x_aux,
        'mask': mask
    })
    
    scores = None
    if 'cyto_seed_score' in adata.obs:
        scores = adata.obs['cyto_seed_score'].values
        print(f"Found cyto_seed_score in dataset with mean value: {np.mean(scores):.4f}")

    hyperparams = {
        "alpha_eta": 0.5, "lambda_eta": 1.0,  # More reasonable priors for SVI
        "alpha_beta": 0.1,                      # Less restrictive
        "alpha_xi": 0.5, "lambda_xi": 1.0,     # More reasonable
        "alpha_theta": 0.1,                     # Less restrictive
        "sigma2_v": 1.0, "sigma2_gamma": 1.0   # Keep reasonable scales
    }
    
    n_pathways = mask.shape[1]
    hyperparams["d"] = n_pathways
    
    n_genes = mask.shape[0]
    beta_init = np.zeros((n_genes, n_pathways))
    
    for i in range(n_pathways):
        pathway_genes = np.where(mask[:, i] > 0)[0]
        if len(pathway_genes) > 0:
            # Use a more gradual initialization instead of 1.0
            base_value = 0.3  # Start with a more moderate value
            beta_init[pathway_genes, i] = base_value
            if seed is not None:
                np.random.seed(seed + i)
            # Add smaller random noise
            beta_init[pathway_genes, i] += np.random.uniform(0, 0.1, size=len(pathway_genes))
        else:
            # If no genes in pathway, set a small baseline value to avoid numerical issues
            beta_init[:, i] = 0.1
    
    # Use a higher minimum value to reduce the gap between pathway and non-pathway genes
    beta_init = np.maximum(beta_init, 0.15)  # Increased from 0.1
    print(f"Beta init shape: {beta_init.shape}, range: [{beta_init.min():.4f}, {beta_init.max():.4f}]")
    print(f"Non-zero entries: {np.count_nonzero(beta_init)} out of {beta_init.size}")
    
    # Use the provided output_dir (timestamped directory) directly
    exp_output_dir = output_dir
    os.makedirs(exp_output_dir, exist_ok=True) # Ensure it exists

    # Create plot prefix for ELBO plots, ensuring filename uniqueness
    plot_prefix_basename = f"initialized_{dataset_name}_pw{n_pathways}"
    plot_prefix = os.path.join(exp_output_dir, plot_prefix_basename)
    
    clear_memory()
    
    if run_fn.__name__ == 'run_model_and_evaluate_corrected':
    # Extract SVI-specific parameters from hyperparams
        svi_params = {
            'batch_size': hyperparams.pop('batch_size', 48),
            'learning_rate': hyperparams.pop('learning_rate', 0.005),
            'early_stopping': True,
            'patience': hyperparams.pop('patience', 20),
            'min_delta': hyperparams.pop('min_delta', 5e-3),
            'verbose': True
        }
        print(f"Added SVI-specific parameters for pathway-initialized experiment")
    else:
        svi_params = {}
        
    results = run_fn(
        x_data=X,
        x_aux=x_aux,
        y_data=Y,
        var_names=var_names,
        hyperparams=hyperparams,
        seed=seed,
        test_size=0.15,
        val_size=0.15,
        max_iters=max_iter,
        return_probs=True,
        sample_ids=sample_ids,
        mask=None,
        scores=scores,
        return_params=True,
        plot_elbo=True,
        plot_prefix=os.path.join(exp_output_dir, "elbo_trace"),
        beta_init=beta_init,
        verbose=True,
        **svi_params
    )
    
    # Reconstruct split indices
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    train_idx, temp_idx = train_test_split(indices, test_size=0.15 + 0.15, random_state=seed if seed is not None else 0)
    val_rel = 0.15 / (0.15 + 0.15)
    val_idx, test_idx = train_test_split(temp_idx, test_size=val_rel, random_state=seed if seed is not None else 0)

    # Save split results
    save_split_results(results, Y, sample_ids, train_idx, "train", exp_output_dir, f"{dataset_name}_{label_col}_initialized_pw{n_pathways}")
    save_split_results(results, Y, sample_ids, val_idx, "val", exp_output_dir, f"{dataset_name}_{label_col}_initialized_pw{n_pathways}")
    save_split_results(results, Y, sample_ids, test_idx, "test", exp_output_dir, f"{dataset_name}_{label_col}_initialized_pw{n_pathways}")

    main_results = results.copy()
    for large_field in ["alpha_beta", "omega_beta", "E_beta"]:
        if large_field in main_results:
            del main_results[large_field]
    
    results_json_filename = f"{dataset_name}_{label_col}_initialized_pw{n_pathways}_results.json.gz"
    out_path = os.path.join(exp_output_dir, results_json_filename)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(main_results, f, indent=2)
    
    # Save E_beta matrix as CSV.gz with v columns
    a_beta = np.array(results["a_beta"])
    b_beta = np.array(results["b_beta"])
    E_beta = a_beta / np.maximum(b_beta, 1e-10)
    row_names = [f"DRGP{i+1}" for i in range(E_beta.shape[1])]
    out_path = os.path.join(exp_output_dir, "gene_programs.csv.gz")
    mu_v = np.array(results["mu_v"]) if "mu_v" in results else None
    save_beta_matrix_csv(E_beta, var_names, row_names, mu_v, out_path)

    return results

def main():
    parser = argparse.ArgumentParser(description="Run experiments with optional mask, custom d, and VI iterations.")
    parser.add_argument("--mask", action="store_true", help="Use mask derived from pathways matrix")
    parser.add_argument("--d", type=int, help="Value of d when mask is not provided")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum iterations for variational inference")
    parser.add_argument("--reduced_pathways", type=int, help="Use only this many pathways from the full set (for testing with mask)")
    parser.add_argument("--combined", action="store_true", help="Run combined pathway+gene program configuration")
    parser.add_argument("--n_gp", type=int, default=500, help="Number of gene programs to learn in combined mode")
    parser.add_argument("--initialized", action="store_true", help="Run pathway-initialized unmasked configuration")
    parser.add_argument("--dataset", type=str, default="cyto", choices=["cyto", "ap", "emtab", "thyroid", "sim"], help="Which dataset to use: cyto, ap, emtab, thyroid, or sim")
    parser.add_argument("--label", type=str, help="Label column to use (for EMTAB and SIM datasets)")
    parser.add_argument("--method", choices=["vi", "svi"], default="vi", help="Inference method to use")
    parser.add_argument("--smote", action="store_true", help="Apply SMOTE upsampling to balance minority classes (for EMTAB dataset)")
    parser.add_argument("--smote_strategy", type=str, default="auto", choices=["auto", "minority"], help="SMOTE sampling strategy")
    parser.add_argument("--profile", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    run_fn = run_svi if args.method == "svi" else run_vi

    if not args.mask and not args.initialized and args.d is None and not args.combined and args.dataset not in ["emtab", "thyroid", "sim"]:
        parser.error("When --mask, --combined, and --initialized flags are not used, --d must be specified (except for emtab, thyroid, and sim).")

    # Determine the base output directory (e.g., .../masked, .../unmasked)
    if args.combined:
        base_output_dir_name = "combined"
    elif args.initialized:
        base_output_dir_name = "pathway_initiated"
    elif args.mask:
        base_output_dir_name = "masked"
    else:
        base_output_dir_name = "unmasked"
    
    # Define the root results directory
    root_results_dir = "/mnt/research/aguiarlab/proj/SSPA-BRAY/BRay/SVIResults"
    base_output_dir = os.path.join(root_results_dir, base_output_dir_name)
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create a timestamp-based subdirectory for this specific run
    date_time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_output_dir, date_time_stamp) 
    os.makedirs(run_dir, exist_ok=True)

    # --- DATASET LOADING LOGIC ---
    datasets = {}
    hyperparams_map = {}
    mask_array = None
    pathway_names_list = None

    if args.dataset == "emtab":
        # Load EMTAB data from emtab_data.py (already imported above)
        if args.smote:
            print("Loading EMTAB data with binary disease classification...")
            random_seed = np.random.randint(0, 2**32 - 1)
            emtab_data = prepare_and_load_emtab_smote(
                random_state=random_seed, 
                sampling_strategy=args.smote_strategy,
                label_type='binary'  # Use binary disease classification
            )
        else:
            emtab_data = prepare_and_load_emtab()
            # Create binary label for non-SMOTE data too
            emtab_data = create_emtab_binary_label(emtab_data)
        # For EMTAB, use binary disease label by default
        label_col = args.label if args.label else "disease_binary"
        print(f"Loaded EMTAB data with shape {emtab_data.shape} and label column '{label_col}'")
        datasets["emtab"] = (emtab_data, label_col)
        
        # Pathway mask logic for EMTAB (same as AJM datasets)
        if args.mask or args.combined or args.initialized:
            gene_names = list(emtab_data.var_names)
            if args.reduced_pathways and args.reduced_pathways > 0:
                if args.reduced_pathways >= len(pathways):
                    print(f"Warning: Requested {args.reduced_pathways} pathways but only {len(pathways)} are available. Using all pathways.")
                    pathway_names_list = list(pathways.keys())
                else:
                    print(f"Using a reduced set of {args.reduced_pathways} pathways out of {len(pathways)} total pathways")
                    random.seed()
                    pathway_names_list = random.sample(list(pathways.keys()), args.reduced_pathways)
                    print(f"Selected {len(pathway_names_list)} pathways randomly")
            else:
                pathway_names_list = list(pathways.keys())
            print(f"Number of genes: {len(gene_names)}")
            print(f"Number of pathways: {len(pathway_names_list)}")
            M = pd.DataFrame(0, index=gene_names, columns=pathway_names_list)
            print("Filling matrix M...")
            chunk_size = 100
            total_chunks = (len(pathway_names_list) + chunk_size - 1) // chunk_size
            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, len(pathway_names_list))
                current_pathways = pathway_names_list[start_idx:end_idx]
                print(f"Processing pathway chunk {chunk_idx+1}/{total_chunks}, pathways {start_idx} to {end_idx}")
                for pathway in current_pathways:
                    gene_list = pathways[pathway]
                    for gene in gene_list:
                        if gene in M.index:
                            M.loc[gene, pathway] = 1
            print(f"Matrix M created with shape {M.shape}")
            log_array_sizes({'M': M.values, 'emtab_data.X': emtab_data.X})
            mask_array = M.values
            print(f"Mask shape: {mask_array.shape}, dtype: {mask_array.dtype}")
            non_zero_entries = np.count_nonzero(mask_array)
            total_entries = mask_array.size
            sparsity = 100 * (1 - non_zero_entries / total_entries)
            print(f"Mask sparsity: {sparsity:.2f}% ({non_zero_entries} non-zero entries out of {total_entries})")
            del M
            clear_memory()
        
        # Set up hyperparams for EMTAB
        if args.method == "svi":
            hyperparams_emtab = {
                "alpha_eta": 0.5, "lambda_eta": 1.0,     # More reasonable priors
                "alpha_beta": 0.1,                        # Less restrictive
                "alpha_xi": 0.5, "lambda_xi": 1.0,       # More reasonable
                "alpha_theta": 0.1,                       # Less restrictive
                "sigma2_v": 1.0, "sigma2_gamma": 1.0     # Keep reasonable scales
            }
        else:
            hyperparams_emtab = {
                "alpha_eta": 0.04, "lambda_eta": 0.11,
                "alpha_beta": 0.11,
                "alpha_xi": 0.02, "lambda_xi": 1.82,
                "alpha_theta": 8.7,
                "sigma2_v": 0.57, "sigma2_gamma": 0.034
            }
        if args.mask:
            hyperparams_emtab["d"] = mask_array.shape[1]
            print(f"Using mask-based d value: {mask_array.shape[1]}")
        elif args.combined:
            print(f"Combined config will use total d = {mask_array.shape[1]} + {args.n_gp}")
        elif args.initialized:
            print(f"Initialized config will use d = {mask_array.shape[1]}")
        else:
            if args.d is not None:
                hyperparams_emtab["d"] = args.d
                print(f"Using specified d value: {args.d}")
            else:
                hyperparams_emtab["d"] = 320
                print(f"Using default d value: 320")
        hyperparams_map = {"emtab": hyperparams_emtab}

    elif args.dataset == "thyroid":
        # Load Thyroid Cancer data
        thyroid_data = prepare_and_load_thyroid()
        
        # # Scale data to reasonable Poisson range
        # print(f"Original data range: [{thyroid_data.X.min()}, {thyroid_data.X.max()}]")
        # X_processed = scale_to_reasonable_counts(thyroid_data.X, target_max=20)
        # print(f"Scaled data range: [{X_processed.min()}, {X_processed.max()}]")
        # thyroid_data.X = X_processed
        
        label_col = "Clinical_History"  # Single label column for thyroid dataset
        print(f"Loaded Thyroid Cancer data with shape {thyroid_data.shape} and label column '{label_col}'")
        datasets["thyroid"] = (thyroid_data, label_col)
        
        # Pathway mask logic for Thyroid (same as other datasets)
        if args.mask or args.combined or args.initialized:
            gene_names = list(thyroid_data.var_names)
            if args.reduced_pathways and args.reduced_pathways > 0:
                if args.reduced_pathways >= len(pathways):
                    print(f"Warning: Requested {args.reduced_pathways} pathways but only {len(pathways)} are available. Using all pathways.")
                    pathway_names_list = list(pathways.keys())
                else:
                    print(f"Using a reduced set of {args.reduced_pathways} pathways out of {len(pathways)} total pathways")
                    pathway_names_list = random.sample(list(pathways.keys()), args.reduced_pathways)
                    print(f"Selected {len(pathway_names_list)} pathways randomly")
            else:
                pathway_names_list = list(pathways.keys())
            print(f"Number of genes: {len(gene_names)}")
            print(f"Number of pathways: {len(pathway_names_list)}")
            M = pd.DataFrame(0, index=gene_names, columns=pathway_names_list)
            print("Filling matrix M...")
            chunk_size = 100
            total_chunks = (len(pathway_names_list) + chunk_size - 1) // chunk_size
            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, len(pathway_names_list))
                current_pathways = pathway_names_list[start_idx:end_idx]
                print(f"Processing pathway chunk {chunk_idx+1}/{total_chunks}, pathways {start_idx} to {end_idx}")
                for pathway in current_pathways:
                    gene_list = pathways[pathway]
                    for gene in gene_list:
                        if gene in M.index:
                            M.loc[gene, pathway] = 1
            print(f"Matrix M created with shape {M.shape}")
            log_array_sizes({'M': M.values, 'thyroid_data.X': thyroid_data.X})
            mask_array = M.values
            print(f"Mask shape: {mask_array.shape}, dtype: {mask_array.dtype}")
            non_zero_entries = np.count_nonzero(mask_array)
            total_entries = mask_array.size
            sparsity = 100 * (1 - non_zero_entries / total_entries)
            print(f"Mask sparsity: {sparsity:.2f}% ({non_zero_entries} non-zero entries out of {total_entries})")
            del M
            clear_memory()
        
        # Set up hyperparams for Thyroid (using similar values to EMTAB since both are human datasets)
        if args.method == "svi":
            hyperparams_thyroid = {
                "alpha_eta": 0.5, "lambda_eta": 1.0,     # More conservative priors
                "alpha_beta": 0.1,                        # Very weak prior
                "alpha_xi": 0.5, "lambda_xi": 1.0,       # More conservative
                "alpha_theta": 0.1,                       # Very weak prior
                "sigma2_v": 1.0, "sigma2_gamma": 1.0     # More reasonable scales
            }
        else:
            # Keep original hyperparameters for VI
            hyperparams_thyroid = {
                "alpha_eta": 0.04, "lambda_eta": 0.11,
                "alpha_beta": 0.11,
                "alpha_xi": 0.02, "lambda_xi": 1.82,
                "alpha_theta": 8.7,
                "sigma2_v": 0.57, "sigma2_gamma": 0.034
            }
            
        if args.mask:
            hyperparams_thyroid["d"] = mask_array.shape[1]
            print(f"Using mask-based d value: {mask_array.shape[1]}")
        elif args.combined:
            print(f"Combined config will use total d = {mask_array.shape[1]} + {args.n_gp}")
        elif args.initialized:
            print(f"Initialized config will use d = {mask_array.shape[1]}")
        else:
            if args.d is not None:
                hyperparams_thyroid["d"] = args.d
                print(f"Using specified d value: {args.d}")
            else:
                hyperparams_thyroid["d"] = 320
                print(f"Using default d value: 320")
        hyperparams_map = {"thyroid": hyperparams_thyroid}

    elif args.dataset == "sim":
        # Load Simulated data
        sim_data = load_data_simulation()
        
        # Scale data to reasonable Poisson range
        print(f"Original data range: [{sim_data.X.min()}, {sim_data.X.max()}]")
        X_processed = scale_to_reasonable_counts(sim_data.X, target_max=20)
        print(f"Scaled data range: [{X_processed.min()}, {X_processed.max()}]")
        sim_data.X = X_processed
        
        # For simulated data, we have the real disease column
        if args.label:
            label_col = args.label
        else:
            # Default to using disease as the response (from real responses)
            if 'disease' in sim_data.obs.columns:
                label_col = 'disease'
                print(f"Using default label column: {label_col}")
            else:
                # Fallback to any numerical column that could be a response
                numeric_cols = sim_data.obs.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    label_col = numeric_cols[0]
                    print(f"Using first numeric column as label: {label_col}")
                else:
                    raise ValueError("No suitable response columns found in simulated data")
        
        print(f"Loaded Simulated data with shape {sim_data.shape} and label column '{label_col}'")
        datasets["sim"] = (sim_data, label_col)
        
        # Pathway mask logic for Simulated data (same as other datasets)
        if args.mask or args.combined or args.initialized:
            gene_names = list(sim_data.var_names)
            if args.reduced_pathways and args.reduced_pathways > 0:
                if args.reduced_pathways >= len(pathways):
                    print(f"Warning: Requested {args.reduced_pathways} pathways but only {len(pathways)} are available. Using all pathways.")
                    pathway_names_list = list(pathways.keys())
                else:
                    print(f"Using a reduced set of {args.reduced_pathways} pathways out of {len(pathways)} total pathways")
                    random.seed()
                    pathway_names_list = random.sample(list(pathways.keys()), args.reduced_pathways)
                    print(f"Selected {len(pathway_names_list)} pathways randomly")
            else:
                pathway_names_list = list(pathways.keys())
            print(f"Number of genes: {len(gene_names)}")
            print(f"Number of pathways: {len(pathway_names_list)}")
            M = pd.DataFrame(0, index=gene_names, columns=pathway_names_list)
            print("Filling matrix M...")
            chunk_size = 100
            total_chunks = (len(pathway_names_list) + chunk_size - 1) // chunk_size
            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, len(pathway_names_list))
                current_pathways = pathway_names_list[start_idx:end_idx]
                print(f"Processing pathway chunk {chunk_idx+1}/{total_chunks}, pathways {start_idx} to {end_idx}")
                for pathway in current_pathways:
                    gene_list = pathways[pathway]
                    for gene in gene_list:
                        if gene in M.index:
                            M.loc[gene, pathway] = 1
            print(f"Matrix M created with shape {M.shape}")
            log_array_sizes({'M': M.values, 'sim_data.X': sim_data.X})
            mask_array = M.values
            print(f"Mask shape: {mask_array.shape}, dtype: {mask_array.dtype}")
            non_zero_entries = np.count_nonzero(mask_array)
            total_entries = mask_array.size
            sparsity = 100 * (1 - non_zero_entries / total_entries)
            print(f"Mask sparsity: {sparsity:.2f}% ({non_zero_entries} non-zero entries out of {total_entries})")
            del M
            clear_memory()
        
        # Set up hyperparams for Simulated data (using default values suitable for simulation)
        if args.method == "svi":
            hyperparams_sim = {
                "alpha_eta": 0.5, "lambda_eta": 1.0,     # Conservative priors
                "alpha_beta": 0.1,                        # Weak prior
                "alpha_xi": 0.5, "lambda_xi": 1.0,       # Conservative
                "alpha_theta": 0.1,                       # Weak prior
                "sigma2_v": 1.0, "sigma2_gamma": 1.0     # Reasonable scales
            }
        else:
            # Default hyperparameters for VI with simulated data
            hyperparams_sim = {
                "alpha_eta": 0.1, "lambda_eta": 1.0,
                "alpha_beta": 0.1,
                "alpha_xi": 0.1, "lambda_xi": 1.0,
                "alpha_theta": 0.1,
                "sigma2_v": 1.0, "sigma2_gamma": 1.0
            }
            
        if args.mask:
            hyperparams_sim["d"] = mask_array.shape[1]
            print(f"Using mask-based d value: {mask_array.shape[1]}")
        elif args.combined:
            print(f"Combined config will use total d = {mask_array.shape[1]} + {args.n_gp}")
        elif args.initialized:
            print(f"Initialized config will use d = {mask_array.shape[1]}")
        else:
            if args.d is not None:
                hyperparams_sim["d"] = args.d
                print(f"Using specified d value: {args.d}")
            else:
                hyperparams_sim["d"] = 50  # Default d value for simulation
                print(f"Using default d value: 50")
        hyperparams_map = {"sim": hyperparams_sim}

    else:
        # Load AJM data (cyto or ap)
        ajm_ap_samples, ajm_cyto_samples = prepare_ajm_dataset()
        if args.dataset == "cyto":
            ajm_cyto_filtered = filter_protein_coding_genes(ajm_cyto_samples, gene_annotation)
            adata = ajm_cyto_filtered
            label_col = "cyto"
            del ajm_ap_samples
            del ajm_cyto_samples
        elif args.dataset == "ap":
            ajm_ap_filtered = filter_protein_coding_genes(ajm_ap_samples, gene_annotation)
            adata = ajm_ap_filtered
            label_col = "ap"
            del ajm_ap_samples
            del ajm_cyto_samples
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        clear_memory()

        # For cyto, add cyto_seed_score
        if args.dataset == "cyto":
            common_genes = np.intersect1d(adata.var_names, CYTOSEED_ensembl)
            print(f"Found {len(common_genes)} common genes between dataset and CYTOSEED_ensembl")
            cyto_seed_mask = np.array([gene in CYTOSEED_ensembl for gene in adata.var_names])
            cyto_seed_scores = adata.X[:, cyto_seed_mask].sum(axis=1)
            adata.obs['cyto_seed_score'] = cyto_seed_scores

        # Pathway mask logic (for cyto only)
        if args.mask or args.combined or args.initialized:
            gene_names = list(adata.var_names)
            if args.reduced_pathways and args.reduced_pathways > 0:
                if args.reduced_pathways >= len(pathways):
                    print(f"Warning: Requested {args.reduced_pathways} pathways but only {len(pathways)} are available. Using all pathways.")
                    pathway_names_list = list(pathways.keys())
                else:
                    print(f"Using a reduced set of {args.reduced_pathways} pathways out of {len(pathways)} total pathways")
                    random.seed()
                    pathway_names_list = random.sample(list(pathways.keys()), args.reduced_pathways)
                    print(f"Selected {len(pathway_names_list)} pathways randomly")
            else:
                pathway_names_list = list(pathways.keys())
            print(f"Number of genes: {len(gene_names)}")
            print(f"Number of pathways: {len(pathway_names_list)}")
            M = pd.DataFrame(0, index=gene_names, columns=pathway_names_list)
            print("Filling matrix M...")
            chunk_size = 100
            total_chunks = (len(pathway_names_list) + chunk_size - 1) // chunk_size
            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, len(pathway_names_list))
                current_pathways = pathway_names_list[start_idx:end_idx]
                print(f"Processing pathway chunk {chunk_idx+1}/{total_chunks}, pathways {start_idx} to {end_idx}")
                for pathway in current_pathways:
                    gene_list = pathways[pathway]
                    for gene in gene_list:
                        if gene in M.index:
                            M.loc[gene, pathway] = 1
            print(f"Matrix M created with shape {M.shape}")
            log_array_sizes({'M': M.values, 'adata.X': adata.X})
            mask_array = M.values
            print(f"Mask shape: {mask_array.shape}, dtype: {mask_array.dtype}")
            non_zero_entries = np.count_nonzero(mask_array)
            total_entries = mask_array.size
            sparsity = 100 * (1 - non_zero_entries / total_entries)
            print(f"Mask sparsity: {sparsity:.2f}% ({non_zero_entries} non-zero entries out of {total_entries})")
            del M
            clear_memory()
        # Set up hyperparams for cyto/ap
        hyperparams_cyto = {
            "alpha_eta": 2.0,  "lambda_eta": 3.0,
            "alpha_beta": 0.6,
            "alpha_xi": 2.0,   "lambda_xi": 3.0,
            "alpha_theta": 0.6,
            "sigma2_v": 1.0,   "sigma2_gamma":   1.0,
        }
        if args.mask:
            hyperparams_cyto["d"] = mask_array.shape[1]
            print(f"Using mask-based d value: {mask_array.shape[1]}")
        elif args.combined:
            print(f"Combined config will use total d = {mask_array.shape[1]} + {args.n_gp}")
        elif args.initialized:
            print(f"Initialized config will use d = {mask_array.shape[1]}")
        else:
            if args.d is not None:
                hyperparams_cyto["d"] = args.d
                print(f"Using specified d value: {args.d}")
            else:
                hyperparams_cyto["d"] = 50
                print(f"Using default d value: 50")
        dataset_key = "ajm_cyto" if args.dataset == "cyto" else "ajm_ap"
        hyperparams_map = {dataset_key: hyperparams_cyto}
        datasets = {dataset_key: (adata, label_col)}

    all_results = {}

    # --- EXPERIMENT RUNNING LOGIC ---
    if args.combined:
        print("\nRunning COMBINED PATHWAY + GENE PROGRAM configuration:")
        print(f"This will use {mask_array.shape[1]} pathway dimensions plus {args.n_gp} freely learned gene program dimensions")
        dataset_name = list(datasets.keys())[0]
        adata, label_col = datasets[dataset_name]
        combined_results = run_combined_gp_and_pathway_experiment(
            dataset_name, adata, label_col, mask_array, pathway_names_list,
            n_gp=args.n_gp, output_dir=run_dir,
            seed=None, max_iter=args.max_iter, run_fn=run_fn
        )
        all_results["combined_config"] = combined_results
    
    elif args.initialized:
        print("\nRunning PATHWAY-INITIALIZED configuration:")
        print(f"This will initialize {mask_array.shape[1]} gene programs using pathway information, then let them evolve freely")
        dataset_name = list(datasets.keys())[0]
        adata, label_col = datasets[dataset_name]
        initialized_results = run_pathway_initialized_experiment(
            dataset_name, adata, label_col, mask_array, pathway_names_list,
            output_dir=run_dir, seed=None, max_iter=args.max_iter, run_fn=run_fn
        )
        all_results["initialized_config"] = initialized_results
    
    else: # Standard masked or unmasked configuration
        print("\nRunning standard configuration:")
        current_mask_for_run = mask_array if args.mask else None
        
        if args.mask:
            print(f"Using MASKED configuration with {mask_array.shape[1]} pathways")
        else:
            dval = hyperparams_map[list(datasets.keys())[0]]["d"]
            if isinstance(dval, list):
                dval = dval[0]
            print(f"Using UNMASKED configuration with {dval} gene programs")

        std_results = run_all_experiments(
            datasets, hyperparams_map, output_dir=run_dir,
            seed=None, mask=current_mask_for_run, max_iter=args.max_iter,
            pathway_names=pathway_names_list, run_fn=run_fn
        )
        all_results.update(std_results)

    print("\nAll experiments completed!")
    print(f"Results saved to: {run_dir}")

    print("\nSummary of results:")
    print("-" * 80)
    # Check if any result has per_class_metrics
    any_per_class = any(
        res and 'train_metrics' in res and isinstance(res['train_metrics'].get('per_class_metrics', None), list)
        for res in all_results.values()
    )
    if any_per_class:
        # Print a table for each label
        # Find the first result with per_class_metrics to get number of labels
        for exp_name, res in all_results.items():
            if res and 'train_metrics' in res and isinstance(res['train_metrics'].get('per_class_metrics', None), list):
                n_labels = len(res['train_metrics']['per_class_metrics'])
                label_names = res.get('label_names', [f"Label {i+1}" for i in range(n_labels)])
                break
        for label_idx in range(n_labels):
            label_name = label_names[label_idx] if label_idx < len(label_names) else f"Label {label_idx+1}"
            print(f"\nSummary of results for label: {label_name}")
            print("-" * 80)
            print(f"{'Experiment':<30} {'Train Acc':<10} {'Val Acc':<10} {'Test Acc':<10} {'Train F1':<10} {'Val F1':<10} {'Test F1':<10}")
            print("-" * 60)
            for exp_name, res in all_results.items():
                if res and 'train_metrics' in res and 'test_metrics' in res and 'val_metrics' in res and \
                   isinstance(res['train_metrics'].get('per_class_metrics', None), list):
                    train_m = res['train_metrics']['per_class_metrics'][label_idx]
                    val_m = res['val_metrics']['per_class_metrics'][label_idx]
                    test_m = res['test_metrics']['per_class_metrics'][label_idx]
                    print(f"{exp_name:<30} {train_m['accuracy']:<10.4f} {val_m['accuracy']:<10.4f} {test_m['accuracy']:<10.4f} "
                          f"{train_m['f1']:<10.4f} {val_m['f1']:<10.4f} {test_m['f1']:<10.4f}")
                else:
                    print(f"{exp_name:<30} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
    else:
        # Fallback: single-label summary as before
        print(f"{'Experiment':<30} {'Train Acc':<10} {'Val Acc':<10} {'Test Acc':<10} {'Train F1':<10} {'Val F1':<10} {'Test F1':<10}")
        print("-" * 60)
        for exp_name, res in all_results.items():
            if res and 'train_metrics' in res and 'test_metrics' in res and 'val_metrics' in res: 
                train_acc = res['train_metrics']['accuracy']
                val_acc   = res['val_metrics']['accuracy']
                test_acc  = res['test_metrics']['accuracy']
                train_f1  = res['train_metrics']['f1']
                val_f1    = res['val_metrics']['f1']
                test_f1   = res['test_metrics']['f1']
                print(f"{exp_name:<30} {train_acc:<10.4f} {val_acc:<10.4f} {test_acc:<10.4f} {train_f1:<10.4f} {val_f1:<10} {test_f1:<10.4f}")
            else:
                print(f"{exp_name:<30} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")

if __name__ == "__main__":
    import sys
    import cProfile
    import pstats
    import argparse as _argparse

    parser = _argparse.ArgumentParser()
    parser.add_argument('--profile', action='store_true', help='Run with cProfile and save output to profile_output.prof')
    args, unknown = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + unknown 

    if args.profile:
        print("Profiling run_experiments.py with cProfile...")
        profile_output = "profile_output.prof"
        cProfile.run('main()', profile_output)
        print(f"Profile data saved to {profile_output}. You can analyze it with 'snakeviz {profile_output}' or 'python -m pstats {profile_output}'")
    else:
        main()