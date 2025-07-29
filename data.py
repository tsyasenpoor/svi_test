import pickle  
import mygene
from gseapy import read_gmt
import anndata as ad
import numpy as np
import pandas as pd
import os
from memory_tracking import get_memory_usage, log_memory, log_array_sizes, clear_memory

# Import SMOTE for upsampling
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline
    SMOTE_AVAILABLE = True
except ImportError:
    print("Warning: imbalanced-learn not available. SMOTE functionality will be disabled.")
    SMOTE_AVAILABLE = False

# Force JAX to use CPU only - must be set before importing jax
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

# Log initial memory
print(f"Initial memory usage: {get_memory_usage():.2f} MB")

from vi_model_complete import run_model_and_evaluate

log_memory("Before loading data files")

cytoseeds_csv_path = "/mnt/research/aguiarlab/proj/SSPA-BRAY/BRay/BRAY_FileTransfer/Seed genes/CYTOBEAM_Cytokines_KEGGPATHWAY_addedMif.csv"
# cytoseeds_csv_path = "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_FileTransfer/Seed genes/CYTOBEAM_Cytokines_KEGGPATHWAY_addedMif.csv"
CYTOSEEDS_df = pd.read_csv(cytoseeds_csv_path)
CYTOSEEDS = CYTOSEEDS_df['V4'].tolist() #173
log_memory("After loading CYTOSEEDS")

mg = mygene.MyGeneInfo()


def save_cache(data, cache_file):
    """Save data to a cache file using pickle."""
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved cached data to {cache_file}")

def load_cache(cache_file):
    """Load data from a cache file if it exists."""
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None


# def convert_pathways_to_ensembl(pathways, cache_file="/labs/Aguiar/SSPA_BRAY/BRay/pathways_ensembl_cache.pkl"):
#     log_memory("Before convert_pathways_to_ensembl")
    
#     # Try to load from cache first
#     cached_pathways = load_cache(cache_file)
#     if cached_pathways is not None:
#         log_memory("After loading pathways from cache")

#         filtered_pathways = {}
#         excluded_keywords = ["ADME", "DRUG", "MISCELLANEOUS", "EMT"]

#         total_pathways = len(cached_pathways)
#         reactome_count = sum(1 for k in cached_pathways if k.startswith('REACTOME'))
        
#         for k, v in cached_pathways.items():
#             if k.startswith('REACTOME'):
#                 if not any(keyword in k.upper() for keyword in excluded_keywords):
#                     filtered_pathways[k] = v
        
#         print(f"Original pathways count: {total_pathways}")
#         print(f"Reactome pathways count: {reactome_count}")
#         print(f"Filtered pathways count (REACTOME only, excluding keywords): {len(filtered_pathways)}")
        
#         # Define specific pathways to exclude
#         specific_exclude = [
#             "REACTOME_GENERIC_TRANSCRIPTION_PATHWAY",
#             "REACTOME_ADAPTIVE_IMMUNE_SYSTEM",
#             "REACTOME_INNATE_IMMUNE_SYSTEM",
#             "REACTOME_IMMUNE_SYSTEM",
#             "REACTOME_METABOLISM"
#         ]
        
#         # Remove specific pathways
#         for pathway in specific_exclude:
#             if pathway in filtered_pathways:
#                 filtered_pathways.pop(pathway)
            
#         excluded_count = reactome_count - len(filtered_pathways)
#         print(f"Reactome pathways excluded due to keywords: {excluded_count}")
#         cached_pathways = filtered_pathways
#         return cached_pathways
    
#     print("Cache not found. Converting pathways to Ensembl IDs...")
#     mg = mygene.MyGeneInfo()
#     unique_genes = set()
#     for genes in pathways.values():
#         unique_genes.update(genes)
#     gene_list = list(unique_genes)
#     print(f"Number of unique genes for conversion: {len(gene_list)}")
    
#     mapping = {}
#     batch_size = 100  # processing in batches for memory efficiency
#     for i in range(0, len(gene_list), batch_size):
#         batch = gene_list[i:i+batch_size]
#         query_results = mg.querymany(batch, scopes='symbol', fields='ensembl.gene', species='mouse', returnall=False)
#         for hit in query_results:
#             if 'ensembl' in hit:
#                 if isinstance(hit['ensembl'], list):
#                     mapping[hit['query']] = hit['ensembl'][0]['gene']
#                 else:
#                     mapping[hit['query']] = hit['ensembl']['gene']
#             else:
#                 mapping[hit['query']] = hit['query']  # keep original if no conversion found
    
#     new_pathways = {}
#     for pathway, genes in pathways.items():
#         new_pathways[pathway] = [mapping.get(g, g) for g in genes]
    
#     # Save to cache for future use
#     save_cache(new_pathways, cache_file)
    
#     log_memory("After convert_pathways_to_ensembl")
#     return new_pathways

# Load pre-filtered pathways directly from pickle
# import pickle
# pathways_pkl_path = "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_FileTransfer/human pathways/filtered_pathways.pkl"
# print(f"Loading pre-filtered pathways from {pathways_pkl_path}")
# with open(pathways_pkl_path, 'rb') as f:
#     pathways = pickle.load(f)

# print("DEBUG: type(pathways) =", type(pathways))

# # If pathways is a DataFrame, convert to dict of lists
# if isinstance(pathways, pd.DataFrame):
#     print("Converting pathways DataFrame to dict of lists...")
#     pathways_dict = {}
#     for col in pathways.columns:
#         gene_list = pathways.index[pathways[col] == 1].tolist()
#         pathways_dict[col] = gene_list
#     pathways = pathways_dict

# Robustly flatten any nested lists in pathways
# import collections.abc

# def flatten(l):
#     for el in l:
#         if isinstance(el, collections.abc.Iterable) and not isinstance(el, str):
#             yield from flatten(el)
#         else:
#             yield el

# for k, v in pathways.items():
#     pathways[k] = list(flatten(v))

# print(f"Loaded {len(pathways)} pre-filtered pathways")


def batch_query(genes, batch_size=100):
    results = []
    for i in range(0, len(genes), batch_size):
        batch = genes[i:i+batch_size]
        results.extend(mg.querymany(batch, scopes='symbol', fields='ensembl.gene', species='mouse'))
    return results

# Define a cache file for CYTOSEED conversions
cytoseed_cache_file = "/mnt/research/aguiarlab/proj/SSPA-BRAY/BRay/cytoseed_ensembl_cache.pkl"
# cytoseed_cache_file = "/labs/Aguiar/SSPA_BRAY/BRay/cytoseed_ensembl_cache.pkl"
# Try to load CYTOSEED mappings from cache
symbol_to_ensembl_asg = load_cache(cytoseed_cache_file)

if symbol_to_ensembl_asg is None:
    log_memory("Before batch query")
    query_results = batch_query(CYTOSEEDS, batch_size=100)
    log_memory("After batch query")
    
    symbol_to_ensembl_asg = {}
    for entry in query_results:
        if 'ensembl' in entry and 'gene' in entry['ensembl']:
            if isinstance(entry['ensembl'], list):
                symbol_to_ensembl_asg[entry['query']] = entry['ensembl'][0]['gene']
            else:
                symbol_to_ensembl_asg[entry['query']] = entry['ensembl']['gene']
        else:
            symbol_to_ensembl_asg[entry['query']] = None 
    
    # Save to cache for future use
    save_cache(symbol_to_ensembl_asg, cytoseed_cache_file)

CYTOSEED_ensembl = [symbol_to_ensembl_asg.get(gene) for gene in CYTOSEEDS if symbol_to_ensembl_asg.get(gene)]
print(f"CYTOSEED_ensembl length: {len(CYTOSEED_ensembl)}")

# log_memory("Before reading pathways")
pathways_path = "/mnt/research/aguiarlab/proj/SSPA-BRAY/BRay/BRAY_FileTransfer/human_pathways/ReactomePathways.gmt"
# pathways_path = "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_FileTransfer/human_pathways/ReactomePathways.gmt"

pathways = read_gmt(pathways_path)  # 1730 pathways
print(f"Number of pathways: {len(pathways)}")
log_memory("After reading pathways")

def filter_pathways(pathways_dict, min_genes=10, max_genes=500, exclude_disease=True, exclude_drug_metabolism=True):
    """
    Filter pathways based on various criteria
    
    Args:
        pathways_dict: Dictionary with pathway names as keys and gene lists as values
        min_genes: Minimum number of genes required in pathway
        max_genes: Maximum number of genes allowed in pathway  
        exclude_disease: Whether to exclude disease-related pathways
        exclude_drug_metabolism: Whether to exclude drug metabolism pathways
    
    Returns:
        Filtered pathways dictionary
    """
    filtered_pathways = {}
    
    for pathway_name, genes in pathways_dict.items():
        gene_count = len(genes)
        
        # Size filter
        if gene_count < min_genes or gene_count > max_genes:
            continue
            
        # Disease/defect filter
        if exclude_disease:
            disease_keywords = ['defective', 'disease', 'cancer', 'tumor', 'carcinoma', 'syndrome', 'infection',
                              'deficiency', 'disorder', 'mutation', 'variant', 'resistant']
            if any(keyword in pathway_name.lower() for keyword in disease_keywords):
                continue
        
        # Drug metabolism filter
        if exclude_drug_metabolism:
            drug_keywords = ['drug', 'response to', 'aspirin', 'adme', 'metabolism', 'biotransformation', 
                           'glucuronidation', 'cytochrome', 'cyp', 'sulfation', 'antiviral']
            if any(keyword in pathway_name.lower() for keyword in drug_keywords):
                continue
        
        # If pathway passes all filters, include it
        filtered_pathways[pathway_name] = genes
    
    print(f"Original pathways: {len(pathways_dict)}")
    print(f"After filtering: {len(filtered_pathways)}")
    
    # Show size distribution of filtered pathways
    sizes = [len(genes) for genes in filtered_pathways.values()]
    if sizes:
        print(f"Gene count range: {min(sizes)} - {max(sizes)}")
        print(f"Mean gene count: {sum(sizes)/len(sizes):.1f}")
    
    return filtered_pathways

# Define a cache file for filtered pathways
filtered_human_pathways_cache_file = "/mnt/research/aguiarlab/proj/SSPA-BRAY/BRay/filtered_human_pathways_cache.pkl"
# filtered_human_pathways_cache_file = "/labs/Aguiar/SSPA_BRAY/BRay/filtered_human_pathways_cache.pkl"

# Try to load filtered pathways from cache
filtered_pathways = load_cache(filtered_human_pathways_cache_file)

if filtered_pathways is None:
    # Apply pathway filtering to the loaded pathways
    filtered_pathways = filter_pathways(pathways, min_genes=10, max_genes=500, 
                                      exclude_disease=True, exclude_drug_metabolism=True)
    
    # Save filtered pathways to cache for future use
    save_cache(filtered_pathways, filtered_human_pathways_cache_file)
    print(f"Filtered pathways saved to cache: {filtered_human_pathways_cache_file}")
else:
    print(f"Loaded filtered pathways from cache: {filtered_human_pathways_cache_file}")

# Use the filtered pathways
pathways = filtered_pathways
print(f"Using {len(pathways)} filtered pathways for analysis")

# pathways = convert_pathways_to_ensembl(pathways)  
# log_memory("After converting pathways to ensembl")

# nap_file_path_raw = "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_AJM2/2_Data/2_SingleCellData/1_GSE139565_NaiveAndPlasma/GEX_NAP_filt_raw_modelingonly_2024-02-05.csv"
# nap_metadata_path = "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_AJM2/2_Data/2_SingleCellData/1_GSE139565_NaiveAndPlasma/meta_NAP_unfilt_fullData_2024-02-05.csv"

# Updated to use RDS file instead of CSV
ajm_file_path = "/mnt/research/aguiarlab/proj/SSPA-BRAY/BRay/BRAY_AJM2/2_Data/2_SingleCellData/2_AJM_Parse_Timecourse/GEX_TC_LPSonly_Bcellonly_filt_raw_2024-02-05.rds"
ajm_metadata_path = "/mnt/research/aguiarlab/proj/SSPA-BRAY/BRay/BRAY_AJM2/2_Data/2_SingleCellData/2_AJM_Parse_Timecourse/meta_TC_LPSonly_Bcellonly_filt_raw_2024-02-05.csv"
# ajm_file_path = "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_AJM2/2_Data/2_SingleCellData/2_AJM_Parse_Timecourse/GEX_TC_LPSonly_Bcellonly_filt_raw_2024-02-05.rds"
# ajm_metadata_path = "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_AJM2/2_Data/2_SingleCellData/2_AJM_Parse_Timecourse/meta_TC_LPSonly_Bcellonly_filt_raw_2024-02-05.csv"

def prepare_ajm_dataset(cache_file="/mnt/research/aguiarlab/proj/SSPA-BRAY/BRay/ajm_dataset_cache.h5ad"):
# def prepare_ajm_dataset(cache_file="/labs/Aguiar/SSPA_BRAY/BRay/ajm_dataset_cache.h5ad"):
    print("Loading AJM dataset...")
    log_memory("Before loading AJM dataset")
    
    # Import required modules at the function's top level
    import os
    import subprocess
    import scipy.sparse as sp
    
    # Try to load from cache first
    if os.path.exists(cache_file):
        print(f"Loading AnnData object from cache file: {cache_file}")
        try:
            adata = ad.read_h5ad(cache_file)
            print(f"Successfully loaded cached AnnData object with shape: {adata.shape}")
            
            # Extract the dataset splits
            ajm_ap_samples = adata[adata.obs['dataset'] == 'ap']
            ajm_cyto_samples = adata[adata.obs['dataset'] == 'cyto']

            # Normalize and log-transform
            QCscRNAsizeFactorNormOnly(ajm_ap_samples)
            QCscRNAsizeFactorNormOnly(ajm_cyto_samples)
            
            print("AJM AP Samples distribution:")
            print(ajm_ap_samples.obs['ap'].value_counts())

            print("AJM CYTO Samples distribution:")
            print(ajm_cyto_samples.obs['cyto'].value_counts())
            
            log_memory("After loading AnnData from cache")
            return ajm_ap_samples, ajm_cyto_samples
        except Exception as e:
            print(f"Error loading cached AnnData: {e}")
            print("Proceeding with RDS conversion...")
    else:
        print(f"Cache file {cache_file} not found, converting RDS file...")
    
    # Run the R script to convert the RDS file to CSV files
    print("Converting RDS to anndata format using R...")
    r_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rds_to_anndata.R")
    
    try:
        result = subprocess.run(["Rscript", r_script_path], 
                               capture_output=True, 
                               text=True, 
                               check=True)
        print("R conversion output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("R conversion error:")
        print(e.stdout)
        print(e.stderr)
        raise RuntimeError("Failed to convert RDS file to CSV")
    
    log_memory("After running R script for conversion")
    
    # Load the sparse matrix data
    print("Loading sparse matrix data...")
    sparse_data = pd.read_csv("matrix_sparse.csv")
    
    # Load row and column names
    row_names = pd.read_csv("matrix_rownames.csv")["row_names"].tolist()  # These are gene names
    col_names = pd.read_csv("matrix_colnames.csv")["col_names"].tolist()  # These are cell names
    
    # Load matrix dimensions (if available)
    matrix_dims = None
    if os.path.exists("matrix_dims.csv"):
        matrix_dims = pd.read_csv("matrix_dims.csv")
        nrows = matrix_dims["rows"].iloc[0]
        ncols = matrix_dims["cols"].iloc[0]
        print(f"Matrix dimensions from file: {nrows} x {ncols}")
        # Verify that dimensions match the length of row and column names
        if nrows != len(row_names) or ncols != len(col_names):
            print(f"WARNING: Dimension mismatch! Row names: {len(row_names)}, Column names: {len(col_names)}")
    else:
        print("Matrix dimensions file not found, using length of row and column names")
        nrows = len(row_names)
        ncols = len(col_names)
    
    print(f"Sparse data shape: {sparse_data.shape}")
    print(f"Number of genes (rows in original matrix): {len(row_names)}")
    print(f"Number of cells (columns in original matrix): {len(col_names)}")
    
    # Create sparse matrix from the CSV data
    # The sparse data has format: row (gene), col (cell), value
    # Indices should already be 0-based from R script
    row_indices = sparse_data["row"].values  # Gene indices
    col_indices = sparse_data["col"].values  # Cell indices
    values = sparse_data["value"].values
    
    # Debug information
    print(f"Row indices range: {row_indices.min()} to {row_indices.max()}")
    print(f"Column indices range: {col_indices.min()} to {col_indices.max()}")
    
    # In AnnData, rows are cells (observations) and columns are genes (variables)
    # So we need to transpose the matrix from our R output
    print(f"Creating sparse matrix with shape: ({nrows}, {ncols}) and transposing to match AnnData format")
    
    # Create a sparse COO matrix with original shape
    sparse_matrix = sp.coo_matrix((values, (row_indices, col_indices)), 
                                 shape=(nrows, ncols))
    
    # Transpose the matrix to have cells as rows and genes as columns
    sparse_matrix = sparse_matrix.transpose().tocsr()
    
    # Now the shape is (ncols, nrows) - (cells, genes)
    print(f"Transposed matrix shape: {sparse_matrix.shape}")
    
    log_memory("After loading sparse matrix")
    
    # Create AnnData object with transposed matrix where:
    # - Rows (observations) are cells
    # - Columns (variables) are genes
    ajm_adata = ad.AnnData(X=sparse_matrix)
    
    # In AnnData:
    # - obs_names (rows) should be cell names
    # - var_names (columns) should be gene names
    ajm_adata.obs_names = col_names  # Cell names as observation names
    ajm_adata.var_names = row_names  # Gene names as variable names
    
    log_memory("After creating AnnData object")
    
    print(f"AnnData object created with shape: {ajm_adata.shape}")
    
    # Load metadata separately
    ajm_features = pd.read_csv(ajm_metadata_path, index_col=0)
    log_memory("After loading AJM metadata")
    
    print(f"AJM features shape: {ajm_features.shape}")
    
    # Create label mappings
    ajm_label_mapping = {
        'TC-0hr':       {'ap':0,'cyto':0,'ig':-1},
        'TC-LPS-3hr':   {'ap':-1,'cyto':0,'ig':-1},
        'TC-LPS-6hr':   {'ap':1,'cyto':-1,'ig':-1},
        'TC-LPS-24hr':  {'ap':1,'cyto':1,'ig':-1},
        'TC-LPS-48hr':  {'ap':-1,'cyto':-1,'ig':-1},
        'TC-LPS-72hr':  {'ap':-1,'cyto':-1,'ig':-1},
    }
    
    # Initialize label columns in metadata
    for col in ['ap','cyto','ig']:
        ajm_features[col] = None
    
    # Apply mapping based on sample values
    for sample_value, labels in ajm_label_mapping.items():
        mask = ajm_features['sample'] == sample_value
        for col in labels:
            ajm_features.loc[mask,col] = labels[col]

    # Set cell_id as index if available
    if 'cell_id' in ajm_features.columns:
        ajm_features.set_index('cell_id', inplace=True)
    
    # Ensure cell IDs match between anndata and metadata
    common_cells_ajm = ajm_adata.obs_names.intersection(ajm_features.index)
    print(f"Number of common cells: {len(common_cells_ajm)}")
    
    # Subset to common cells
    ajm_adata = ajm_adata[common_cells_ajm]
    ajm_features = ajm_features.loc[common_cells_ajm]
    
    # Add metadata to AnnData object
    for col in ajm_features.columns:
        ajm_adata.obs[col] = ajm_features[col].values
    
    # Ensure gene symbols are available
    if 'gene_symbols' not in ajm_adata.var:
        ajm_adata.var['gene_symbols'] = ajm_adata.var_names
    
    # Create subset AnnData objects for different analyses
    ajm_ap_samples = ajm_adata[ajm_adata.obs['ap'].isin([0,1])]
    ajm_cyto_samples = ajm_adata[ajm_adata.obs['cyto'].isin([0,1])]

    # Normalize and log-transform
    QCscRNAsizeFactorNormOnly(ajm_ap_samples)
    QCscRNAsizeFactorNormOnly(ajm_cyto_samples)

    # Add dataset identifier to help with cache loading
    ajm_ap_samples.obs['dataset'] = 'ap'
    ajm_cyto_samples.obs['dataset'] = 'cyto'
    
    # Combine both datasets for caching
    combined_adata = ad.concat(
        [ajm_ap_samples, ajm_cyto_samples],
        join='outer',
        merge='same'
    )
    
    # Make sure all values in .obs are serializable (convert to string if needed)
    for col in combined_adata.obs.columns:
        if combined_adata.obs[col].dtype == 'object':
            combined_adata.obs[col] = combined_adata.obs[col].astype(str)
        
        # Convert None/NaN values to strings to avoid serialization issues
        combined_adata.obs[col] = combined_adata.obs[col].fillna('NA')
    
    # Similarly ensure .var values are serializable
    for col in combined_adata.var.columns:
        if combined_adata.var[col].dtype == 'object':
            combined_adata.var[col] = combined_adata.var[col].astype(str)
        
        # Convert None/NaN values to strings
        combined_adata.var[col] = combined_adata.var[col].fillna('NA')
    
    # Save to cache file
    print(f"Saving AnnData object to cache file: {cache_file}")
    try:
        combined_adata.write_h5ad(cache_file)
        print(f"Successfully saved AnnData to cache")
    except TypeError as e:
        print(f"Error saving AnnData to H5AD: {e}")
        print("Will proceed without caching.")
    
    print("AJM AP Samples distribution:")
    print(ajm_ap_samples.obs['ap'].value_counts())

    print("AJM CYTO Samples distribution:")
    print(ajm_cyto_samples.obs['cyto'].value_counts())
    
    # Log memory usage of created AnnData objects
    log_array_sizes({
        'ajm_adata.X': ajm_adata.X,
        'ajm_ap_samples.X': ajm_ap_samples.X,
        'ajm_cyto_samples.X': ajm_cyto_samples.X
    })
    
    # Try to clear some memory
    clear_memory()
    
    # Clean up the temporary CSV files
    temp_files = ["matrix_sparse.csv", "matrix_rownames.csv", "matrix_colnames.csv", "matrix_dims.csv"]
    for file in temp_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"Temporary file {file} removed")
            except Exception as e:
                print(f"Warning: Could not remove temporary file {file}: {e}")
    
    return ajm_ap_samples, ajm_cyto_samples

def prepare_and_load_thyroid():
    """
    Load and prepare Thyroid Cancer dataset from preprocessed files.
    
    Returns:
        adata: AnnData object containing gene expression data with gene symbols as var_names,
               and Clinical_History label and auxiliary variables in .obs
    """
    import pickle
    import mygene
    import numpy as np
    import scipy.sparse as sp
    data_path = "/mnt/research/aguiarlab/proj/SSPA-BRAY/SSPA/dataset/Thyroid_Cancer/preprocessed"
    gene_expression_file = "gene_expression.csv.gz"
    responses_file = "responses.csv.gz"
    aux_data_file = "aux_data.csv.gz"

    # Load data with proper handling of mixed types and set Sample_ID as index
    gene_expression = pd.read_csv(os.path.join(data_path, gene_expression_file), compression='gzip', low_memory=False, index_col='Sample_ID')
    responses = pd.read_csv(os.path.join(data_path, responses_file), compression='gzip', index_col='Sample_ID')
    aux_data = pd.read_csv(os.path.join(data_path, aux_data_file), compression='gzip', index_col='Sample_ID')

    # Concatenate all three dataframes into a single dataframe
    combined_data = pd.concat([gene_expression, responses, aux_data], axis=1)

    # Separate gene expression data (X) from labels and auxiliary variables
    gene_cols = [col for col in combined_data.columns if col not in ["Clinical_History", "Age", "sex_female"]]
    X = combined_data[gene_cols]
    labels = combined_data[["Clinical_History"]]
    aux_vars = combined_data[["Age", "sex_female"]]

    # Convert gene expression to numeric, handling any non-numeric values
    print("Converting gene expression data to numeric...")
    X = X.apply(pd.to_numeric, errors='coerce')
    
    # Fill any NaN values that resulted from conversion with 0
    X = X.fillna(0)
    
    # Ensure X is float type for numerical operations
    X = X.astype(np.float64)
    
    print(f"Gene expression matrix shape: {X.shape}")
    print(f"Gene expression data type: {X.dtypes.iloc[0]}")
    print(f"Gene expression min: {X.min().min()}, max: {X.max().max()}")

    adata = ad.AnnData(X=X)

    # Add labels as obs
    adata.obs = labels.copy()
    adata.obs_names = combined_data.index

    # Add auxiliary variables as obs
    adata.obs = pd.concat([adata.obs, aux_vars], axis=1)
    
    # Ensure all numeric columns are proper numeric types (not strings)
    for col in ['Clinical_History', 'Age', 'sex_female']:
        if col in adata.obs.columns:
            adata.obs[col] = pd.to_numeric(adata.obs[col], errors='coerce')
    
    # Check for any NaN values that might cause issues
    print(f"NaN values in obs columns:")
    for col in adata.obs.columns:
        nan_count = adata.obs[col].isna().sum()
        if nan_count > 0:
            print(f"  {col}: {nan_count} NaN values")
    
    # Fill any NaN values with appropriate defaults
    if 'Clinical_History' in adata.obs.columns:
        adata.obs['Clinical_History'] = adata.obs['Clinical_History'].fillna(0)
    if 'Age' in adata.obs.columns:
        adata.obs['Age'] = adata.obs['Age'].fillna(adata.obs['Age'].median())
    if 'sex_female' in adata.obs.columns:
        adata.obs['sex_female'] = adata.obs['sex_female'].fillna(0)

    # Make variable names unique to avoid warnings
    adata.var_names_make_unique()

    print(f"AnnData object created for Thyroid Cancer dataset:")
    print(f"  - Shape: {adata.shape}")
    print(f"  - Observations (samples): {adata.n_obs}")
    print(f"  - Variables (genes): {adata.n_vars}")
    print(f"  - Obs columns: {list(adata.obs.columns)}")
    print(f"  - Clinical_History label distribution:")
    print(adata.obs['Clinical_History'].value_counts())
    print(f"  - Clinical_History data type: {adata.obs['Clinical_History'].dtype}")
    print(f"  - First few obs values:")
    print(adata.obs.head())

    return adata

def prepare_and_load_emtab():
    """
    Load and prepare EMTAB dataset from preprocessed files, converting gene symbols to Ensembl IDs.
    Uses mygene to convert gene names, and caches the converted AnnData object as a pickle file.
    On subsequent runs, loads the converted AnnData from the pickle if it exists.
    
    Returns:
        adata: AnnData object containing gene expression data with Ensembl IDs as var_names,
               and labels and auxiliary variables in .obs
    """
    import pickle
    import mygene
    import numpy as np
    import scipy.sparse as sp
    data_path = "/mnt/research/aguiarlab/proj/SSPA-BRAY/SSPA/dataset/ArrayExpress/preprocessed"
    # data_path = "/labs/Aguiar/SSPA_BRAY/dataset/EMTAB11349/preprocessed/"
    cache_file = os.path.join(data_path, "raw_emtab_symbol.pkl")

    # If cached converted AnnData exists, load and return it
    if os.path.exists(cache_file):
        print(f"Loading cached Ensembl-converted AnnData from {cache_file}")
        with open(cache_file, "rb") as f:
            adata = pickle.load(f)
        print(f"Loaded AnnData with shape: {adata.shape}")
        return adata

    # Otherwise, load and process the data
    gene_expression_file = "tmm_tpm_gene_expression.csv.gz"
    responses_file = "responses.csv.gz"
    aux_data_file = "aux_data.csv.gz"

    gene_expression = pd.read_csv(os.path.join(data_path, gene_expression_file), index_col=0, compression='gzip')
    responses = pd.read_csv(os.path.join(data_path, responses_file), index_col=0, compression='gzip')
    aux_data = pd.read_csv(os.path.join(data_path, aux_data_file), index_col=0, compression='gzip')

    # Concatenate all three dataframes into a single dataframe
    combined_data = pd.concat([gene_expression, responses, aux_data], axis=1)

    # Separate gene expression data (X) from labels and auxiliary variables
    gene_cols = [col for col in combined_data.columns if col not in ["Crohn's disease", "ulcerative colitis", "age", "sex_female"]]
    X = combined_data[gene_cols]
    labels = combined_data[["Crohn's disease", "ulcerative colitis"]]
    aux_vars = combined_data[["age", "sex_female"]]

    # Create AnnData object using original gene expression data
    adata = ad.AnnData(X=X)

    # Add labels as obs
    adata.obs = labels.copy()
    adata.obs_names = combined_data.index

    # Add auxiliary variables as obs
    adata.obs = pd.concat([adata.obs, aux_vars], axis=1)

    # Add gene symbols as var_names (keeping original names)
    adata.var_names = X.columns.tolist()

    print(f"AnnData object created (gene symbols):")
    print(f"  - Shape: {adata.shape}")
    print(f"  - Observations (samples): {adata.n_obs}")
    print(f"  - Variables (genes): {adata.n_vars}")
    print(f"  - Obs columns: {list(adata.obs.columns)}")
    print(f"  - First few obs values:")
    print(adata.obs.head())

    # Save the AnnData to cache for future use
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(adata, f)
        print(f"Saved AnnData to {cache_file}")
    except Exception as e:
        print(f"Warning: Could not save cache file {cache_file}: {e}")

    return adata

def load_data_simulation():
    """
    Load the simulation dataset and return as an AnnData object.
    - Loads gene expression, aux data, and real responses only.
    - Returns: AnnData object (gene expression as X, aux/response as obs columns).
    """

    # main_path = '/labs/Aguiar/SSPA_BRAY/dataset/simulation_base'
    main_path = '/mnt/research/aguiarlab/proj/SSPA-BRAY/SSPA/dataset/simulation_large_scale'
    gene_expression_path = os.path.join(main_path, 'gene_expression.csv.gz')
    aux_data_path = os.path.join(main_path, 'aux_data.csv.gz')
    responses_sampled_path = os.path.join(main_path, 'responses_sampled.csv.gz')
    responses_real_path = os.path.join(main_path, 'responses_real.csv.gz')

    # Load gene expression
    gene_expression_df = pd.read_csv(gene_expression_path, compression='gzip')
    obs_names = gene_expression_df['Sample_ID'].astype(str).values  # Convert to strings
    gene_expression_df = gene_expression_df.drop(columns=['Sample_ID'])
    X = gene_expression_df.values
    var_names = gene_expression_df.columns.tolist()

    # Load aux data
    aux_data_df = pd.read_csv(aux_data_path, compression='gzip')
    aux_data_df = aux_data_df.drop(columns=['Sample_ID'])

    # Load only real responses (not sampled ones)
    responses_real_df = pd.read_csv(responses_real_path, compression='gzip')
    responses_real_df = responses_real_df.drop(columns=['Sample_ID'])
    # Keep the original column names since we're only using real responses

    # Combine obs columns
    obs = pd.concat([aux_data_df, responses_real_df], axis=1)
    obs.index = obs_names

    # Create AnnData object
    adata = ad.AnnData(X=X, obs=obs)
    adata.obs_names = obs_names
    adata.var_names = var_names

    return adata

log_memory("Before loading gene annotations")
gene_annotation_path = "/mnt/research/aguiarlab/proj/SSPA-BRAY/BRay/BRAY_FileTransfer/ENS_mouse_geneannotation.csv"
# gene_annotation_path = "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_FileTransfer/ENS_mouse_geneannotation.csv"
gene_annotation = pd.read_csv(gene_annotation_path)
gene_annotation = gene_annotation.set_index('GeneID')
log_memory("After loading gene annotations")

def filter_protein_coding_genes(adata, gene_annotation):
    log_memory("Before filtering protein coding genes")
    protein_coding_genes = gene_annotation[gene_annotation['Genetype'] == 'protein_coding'].index
    
    common_genes = np.intersect1d(adata.var_names, protein_coding_genes)
    
    print(f"Total genes: {adata.n_vars}")
    print(f"Protein-coding genes found: {len(common_genes)}")
    
    adata_filtered = adata[:, common_genes].copy()

    log_memory("After filtering protein coding genes")
    log_array_sizes({
        'adata.X': adata.X,
        'adata_filtered.X': adata_filtered.X
    })
    
    return adata_filtered


def QCscRNAsizeFactorNormOnly(adata):
    """Normalize counts in an AnnData object using a median-based size factor per cell (row-wise)."""
    import numpy as np
    import scipy.sparse as sp

    X = adata.X.astype(float)

    if sp.issparse(X):
        UMI_counts_per_cell = np.array(X.sum(axis=1)).flatten()  # Sum over columns → per row (cell)
    else:
        UMI_counts_per_cell = X.sum(axis=1)

    median_UMI = np.median(UMI_counts_per_cell)
    scaling_factors = median_UMI / UMI_counts_per_cell
    scaling_factors[np.isinf(scaling_factors)] = 0  # Avoid inf if dividing by zero

    if sp.issparse(X):
        scaling_matrix = sp.diags(scaling_factors)
        X = scaling_matrix @ X  # Multiply from the left: row-wise scaling
    else:
        X = X * scaling_factors[:, np.newaxis]  # Broadcast scaling per row

    adata.X = X
    return adata


def scale_to_reasonable_counts(X, target_max=100):
    """Scale data to reasonable count range while preserving structure"""
    percentile_99 = np.percentile(X, 99)
    scale_factor = target_max / percentile_99
    return np.round(X * scale_factor).astype(int)

def sample_adata(adata, n_cells=None, cell_fraction=None,
                 n_genes=None, gene_fraction=None, random_state=0):
    """Return a random subset of the AnnData object.

    Parameters
    ----------
    adata : AnnData
        Input dataset.
    n_cells : int, optional
        Number of cells to sample.  Mutually exclusive with ``cell_fraction``.
    cell_fraction : float, optional
        Fraction of cells to sample.
    n_genes : int, optional
        Number of genes to sample.  Mutually exclusive with ``gene_fraction``.
    gene_fraction : float, optional
        Fraction of genes to sample.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    AnnData
        Subsampled AnnData object.
    """

    rng = np.random.default_rng(random_state)

    if cell_fraction is not None:
        n_cells = max(1, int(adata.n_obs * cell_fraction))
    if n_cells is None or n_cells > adata.n_obs:
        n_cells = adata.n_obs
    cell_indices = rng.choice(adata.n_obs, size=n_cells, replace=False)

    if gene_fraction is not None:
        n_genes = max(1, int(adata.n_vars * gene_fraction))
    if n_genes is None or n_genes > adata.n_vars:
        n_genes = adata.n_vars
    gene_indices = rng.choice(adata.n_vars, size=n_genes, replace=False)

    return adata[cell_indices, :][:, gene_indices].copy()

def apply_mlsmote_to_emtab(adata, random_state=None, k_neighbors=5):
    """
    Apply Multi-Label SMOTE (MLSMOTE) to balance the minority classes in EMTAB dataset.
    This creates synthetic samples by interpolating between minority samples while respecting
    label co-occurrence patterns.
    
    Args:
        adata: AnnData object containing EMTAB data
        random_state: Random seed for reproducibility
        k_neighbors: Number of neighbors to consider for interpolation
    
    Returns:
        adata_balanced: AnnData object with balanced classes
    """
    print("Applying Multi-Label SMOTE to balance EMTAB dataset...")
    
    # Extract features and labels
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    y_crohn = adata.obs["Crohn's disease"].values
    y_ulcerative = adata.obs["ulcerative colitis"].values
    
    # Create multi-label array
    y_multilabel = np.column_stack([y_crohn, y_ulcerative])
    
    # Print original distribution
    print(f"Original: Crohn pos={np.sum(y_crohn)}, neg={np.sum(1-y_crohn)}")
    print(f"Original: Ulcerative pos={np.sum(y_ulcerative)}, neg={np.sum(1-y_ulcerative)}")
    
    # Calculate label frequencies
    label_counts = np.sum(y_multilabel, axis=0)
    print(f"Label counts: Crohn={label_counts[0]}, Ulcerative={label_counts[1]}")
    
    # Find minority samples (samples with at least one minority label)
    crohn_pos_indices = np.where(y_crohn == 1)[0]
    ulcerative_pos_indices = np.where(y_ulcerative == 1)[0]
    
    # Combine minority samples
    minority_indices = np.unique(np.concatenate([crohn_pos_indices, ulcerative_pos_indices]))
    print(f"Total minority samples: {len(minority_indices)}")
    
    # Calculate how many samples we need to add
    crohn_pos_count = np.sum(y_crohn)
    crohn_neg_count = np.sum(1 - y_crohn)
    ulcerative_pos_count = np.sum(y_ulcerative)
    ulcerative_neg_count = np.sum(1 - y_ulcerative)
    
    crohn_pos_needed = crohn_neg_count - crohn_pos_count
    ulcerative_pos_needed = ulcerative_neg_count - ulcerative_pos_count
    
    total_needed = max(crohn_pos_needed, ulcerative_pos_needed)
    print(f"Need to add {total_needed} synthetic samples")
    
    # Start with original data
    X_balanced = X.copy()
    y_crohn_balanced = y_crohn.copy()
    y_ulcerative_balanced = y_ulcerative.copy()
    
    if total_needed > 0:
        # Use KNN to find neighbors for interpolation
        from sklearn.neighbors import NearestNeighbors
        
        # Fit KNN on minority samples
        knn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(minority_indices)), 
                              algorithm='auto')
        knn.fit(X[minority_indices])
        
        # Generate synthetic samples
        np.random.seed(random_state)
        synthetic_samples = []
        synthetic_labels = []
        
        for i in range(total_needed):
            # Randomly select a minority sample
            minority_idx = np.random.choice(minority_indices)
            
            # Find its k nearest neighbors
            distances, indices = knn.kneighbors([X[minority_idx]])
            
            # Remove the sample itself from neighbors
            neighbor_indices = indices[0][1:]  # Skip first (self)
            
            if len(neighbor_indices) > 0:
                # Randomly select a neighbor
                neighbor_idx = np.random.choice(neighbor_indices)
                
                # Create synthetic sample by interpolation
                alpha = np.random.random()
                synthetic_sample = alpha * X[minority_idx] + (1 - alpha) * X[neighbor_idx]
                
                # Create synthetic labels (interpolate between the two samples)
                minority_labels = y_multilabel[minority_idx]
                neighbor_labels = y_multilabel[neighbor_idx]
                synthetic_label = (alpha * minority_labels + (1 - alpha) * neighbor_labels) > 0.5
                
                synthetic_samples.append(synthetic_sample)
                synthetic_labels.append(synthetic_label)
            else:
                # If no neighbors, just duplicate the sample
                synthetic_samples.append(X[minority_idx])
                synthetic_labels.append(y_multilabel[minority_idx])
        
        # Add synthetic samples
        X_synthetic = np.array(synthetic_samples)
        y_synthetic = np.array(synthetic_labels)
        
        X_balanced = np.vstack([X_balanced, X_synthetic])
        y_crohn_balanced = np.concatenate([y_crohn_balanced, y_synthetic[:, 0]])
        y_ulcerative_balanced = np.concatenate([y_ulcerative_balanced, y_synthetic[:, 1]])
    
    # Print final distribution
    print(f"Final: Crohn pos={np.sum(y_crohn_balanced)}, neg={np.sum(1-y_crohn_balanced)}")
    print(f"Final: Ulcerative pos={np.sum(y_ulcerative_balanced)}, neg={np.sum(1-y_ulcerative_balanced)}")
    
    # Create new AnnData object
    adata_balanced = ad.AnnData(X=X_balanced)
    adata_balanced.var_names = adata.var_names.copy()
    
    # Create balanced labels
    balanced_labels = pd.DataFrame({
        "Crohn's disease": y_crohn_balanced,
        "ulcerative colitis": y_ulcerative_balanced
    })
    
    # Handle auxiliary variables
    aux_vars = adata.obs[['age', 'sex_female']].copy()
    n_original = len(adata)
    n_new = len(X_balanced) - n_original
    
    if n_new > 0:
        age_mean = aux_vars['age'].mean()
        sex_distribution = aux_vars['sex_female'].value_counts(normalize=True)
        
        np.random.seed(random_state)
        new_sex_values = np.random.choice(
            sex_distribution.index, 
            size=n_new, 
            p=sex_distribution.values
        )
        
        new_aux_vars = pd.DataFrame({
            'age': [age_mean] * n_new,
            'sex_female': new_sex_values
        })
        
        balanced_aux_vars = pd.concat([aux_vars, new_aux_vars], ignore_index=True)
    else:
        balanced_aux_vars = aux_vars
    
    # Set observation names and data
    original_names = adata.obs_names.tolist()
    new_names = original_names + [f"synthetic_{i}" for i in range(n_new)]
    adata_balanced.obs_names = new_names[:len(X_balanced)]
    adata_balanced.obs = pd.concat([balanced_labels, balanced_aux_vars], axis=1)
    
    print(f"Balanced shape: {adata_balanced.shape} (original: {adata.shape})")
    
    return adata_balanced

def prepare_and_load_emtab_smote(random_state=None, sampling_strategy='auto', label_type='both_labels'):
    """
    Load and prepare EMTAB dataset with SMOTE upsampling applied.
    
    Args:
        random_state: Random seed for SMOTE
        sampling_strategy: SMOTE sampling strategy
        label_type: Type of label to create ('both_labels', 'binary', 'multiclass')
    
    Returns:
        adata: AnnData object with balanced classes
    """
    # First load the original EMTAB data
    adata_original = prepare_and_load_emtab()
    
    # Apply Multi-Label SMOTE to balance the classes
    adata_balanced = apply_mlsmote_to_emtab(
        adata_original, 
        random_state=random_state
    )
    
    # Create the appropriate label column based on label_type
    if label_type == 'binary':
        # For binary classification, the data is already fairly balanced
        # No disease: ~267, Any disease: ~323 - minimal imbalance
        print("Binary classification: Using original data (minimal imbalance)")
        adata_balanced = adata_original.copy()
        adata_balanced = create_emtab_binary_label(adata_balanced)
        print(f"Loaded EMTAB data with shape {adata_balanced.shape} and binary label column 'disease_binary'")
    elif label_type == 'multiclass':
        adata_balanced = create_emtab_multiclass_label(adata_balanced)
        print(f"Loaded EMTAB data with shape {adata_balanced.shape} and multi-class label column 'disease_multiclass'")
    else:  # default: both_labels
        # Add a 'both_labels' column for compatibility with downstream functions
        # This column is typically used for single-label experiments or for general data handling
        # For multi-label, the individual 'Crohn's disease' and 'ulcerative colitis' columns are used
        adata_balanced.obs['both_labels'] = (adata_balanced.obs["Crohn's disease"] | adata_balanced.obs["ulcerative colitis"]).astype(int)
        print(f"Loaded EMTAB data with shape {adata_balanced.shape} and label column 'both_labels'")
    
    return adata_balanced

def create_emtab_binary_label(adata):
    """
    Create a binary label for EMTAB dataset: 0 = no disease, 1 = any disease.
    
    Args:
        adata: AnnData object containing EMTAB data
        
    Returns:
        adata: AnnData object with new 'disease_binary' column
    """
    # Create binary label: 0 = no disease, 1 = any disease
    crohn = adata.obs["Crohn's disease"].values
    ulcerative = adata.obs["ulcerative colitis"].values
    
    # Any disease = Crohn's OR Ulcerative colitis
    disease_binary = (crohn | ulcerative).astype(int)
    
    # Add the new column
    adata.obs['disease_binary'] = disease_binary
    
    # Print distribution
    print(f"Binary disease distribution:")
    print(f"  0 (No disease): {np.sum(disease_binary == 0)}")
    print(f"  1 (Any disease): {np.sum(disease_binary == 1)}")
    
    return adata

def create_emtab_multiclass_label(adata):
    """
    Create a multi-class label for EMTAB dataset:
    0 = No disease, 1 = Crohn's only, 2 = Ulcerative only, 3 = Both diseases
    
    Args:
        adata: AnnData object containing EMTAB data
        
    Returns:
        adata: AnnData object with new 'disease_multiclass' column
    """
    # Create multi-class label
    crohn = adata.obs["Crohn's disease"].values
    ulcerative = adata.obs["ulcerative colitis"].values
    
    # Multi-class encoding
    disease_multiclass = np.zeros(len(crohn), dtype=int)
    disease_multiclass[crohn & ~ulcerative] = 1  # Crohn's only
    disease_multiclass[~crohn & ulcerative] = 2  # Ulcerative only
    disease_multiclass[crohn & ulcerative] = 3   # Both diseases
    
    # Add the new column
    adata.obs['disease_multiclass'] = disease_multiclass
    
    # Print distribution
    print(f"Multi-class disease distribution:")
    print(f"  0 (No disease): {np.sum(disease_multiclass == 0)}")
    print(f"  1 (Crohn's only): {np.sum(disease_multiclass == 1)}")
    print(f"  2 (Ulcerative only): {np.sum(disease_multiclass == 2)}")
    print(f"  3 (Both diseases): {np.sum(disease_multiclass == 3)}")
    
    return adata

def apply_binary_smote_to_emtab(adata, random_state=None, k_neighbors=5):
    """
    Apply SMOTE for binary disease classification: 0 = no disease, 1 = any disease.
    Only balances the overall disease vs no disease ratio.
    
    Args:
        adata: AnnData object containing EMTAB data
        random_state: Random seed for reproducibility
        k_neighbors: Number of neighbors to consider for interpolation
    
    Returns:
        adata_balanced: AnnData object with balanced classes
    """
    print("Applying Binary SMOTE to balance disease vs no disease...")
    
    # Extract features and create binary disease label
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    crohn = adata.obs["Crohn's disease"].values
    ulcerative = adata.obs["ulcerative colitis"].values
    
    # Create binary disease label: 0 = no disease, 1 = any disease
    disease_binary = (crohn | ulcerative).astype(int)
    
    # Print original distribution
    print(f"Original binary distribution:")
    print(f"  0 (No disease): {np.sum(disease_binary == 0)}")
    print(f"  1 (Any disease): {np.sum(disease_binary == 1)}")
    
    # Find minority class (disease samples)
    disease_indices = np.where(disease_binary == 1)[0]
    no_disease_indices = np.where(disease_binary == 0)[0]
    
    print(f"Disease samples: {len(disease_indices)}, No disease samples: {len(no_disease_indices)}")
    
    # Calculate how many disease samples we need to add
    disease_count = len(disease_indices)
    no_disease_count = len(no_disease_indices)
    disease_needed = no_disease_count - disease_count
    
    if disease_needed > 0:
        print(f"Need to add {disease_needed} disease samples to balance classes")
        
        # Use KNN to find nearest neighbors among disease samples
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(disease_indices)), algorithm='auto')
        knn.fit(X[disease_indices])
        
        synthetic_samples = []
        synthetic_crohn = []
        synthetic_ulcerative = []
        
        for i in range(disease_needed):
            # Randomly select a disease sample
            disease_idx = np.random.choice(disease_indices)
            
            # Find its k nearest neighbors
            distances, indices = knn.kneighbors([X[disease_idx]])
            
            # Remove the sample itself from neighbors
            neighbor_indices = indices[0][1:]  # Skip first (self)
            
            if len(neighbor_indices) > 0:
                # Randomly select a neighbor
                neighbor_idx = np.random.choice(neighbor_indices)
                neighbor_original_idx = disease_indices[neighbor_idx]
                
                # Create synthetic sample by interpolation
                alpha = np.random.random()
                synthetic_sample = alpha * X[disease_idx] + (1 - alpha) * X[neighbor_original_idx]
                
                # Create synthetic labels (interpolate between the two samples)
                original_crohn = crohn[disease_idx]
                original_ulcerative = ulcerative[disease_idx]
                neighbor_crohn = crohn[neighbor_original_idx]
                neighbor_ulcerative = ulcerative[neighbor_original_idx]
                
                # For binary classification, we want to preserve the "any disease" property
                # So we'll randomly choose between the two disease patterns
                if np.random.random() < 0.5:
                    synthetic_crohn_label = original_crohn
                    synthetic_ulcerative_label = original_ulcerative
                else:
                    synthetic_crohn_label = neighbor_crohn
                    synthetic_ulcerative_label = neighbor_ulcerative
                
                synthetic_samples.append(synthetic_sample)
                synthetic_crohn.append(synthetic_crohn_label)
                synthetic_ulcerative.append(synthetic_ulcerative_label)
            else:
                # If no neighbors, just duplicate the sample
                synthetic_samples.append(X[disease_idx])
                synthetic_crohn.append(crohn[disease_idx])
                synthetic_ulcerative.append(ulcerative[disease_idx])
        
        # Add synthetic samples
        X_synthetic = np.array(synthetic_samples)
        crohn_synthetic = np.array(synthetic_crohn)
        ulcerative_synthetic = np.array(synthetic_ulcerative)
        
        X_balanced = np.vstack([X, X_synthetic])
        crohn_balanced = np.concatenate([crohn, crohn_synthetic])
        ulcerative_balanced = np.concatenate([ulcerative, ulcerative_synthetic])
    else:
        print("Classes are already balanced or disease is majority class")
        X_balanced = X.copy()
        crohn_balanced = crohn.copy()
        ulcerative_balanced = ulcerative.copy()
    
    # Create new AnnData object
    adata_balanced = ad.AnnData(X=X_balanced)
    adata_balanced.var_names = adata.var_names.copy()
    
    # Create balanced labels
    balanced_labels = pd.DataFrame({
        "Crohn's disease": crohn_balanced,
        "ulcerative colitis": ulcerative_balanced
    })
    
    # Handle auxiliary variables
    aux_vars = adata.obs[['age', 'sex_female']].copy()
    n_original = len(adata)
    n_new = len(X_balanced) - n_original
    
    if n_new > 0:
        age_mean = aux_vars['age'].mean()
        sex_distribution = aux_vars['sex_female'].value_counts(normalize=True)
        
        np.random.seed(random_state)
        new_sex_values = np.random.choice(
            sex_distribution.index, 
            size=n_new, 
            p=sex_distribution.values
        )
        
        new_aux_vars = pd.DataFrame({
            'age': [age_mean] * n_new,
            'sex_female': new_sex_values
        })
        
        balanced_aux_vars = pd.concat([aux_vars, new_aux_vars], ignore_index=True)
    else:
        balanced_aux_vars = aux_vars
    
    # Set observation names and data
    original_names = adata.obs_names.tolist()
    new_names = original_names + [f"synthetic_{i}" for i in range(n_new)]
    adata_balanced.obs_names = new_names[:len(X_balanced)]
    adata_balanced.obs = pd.concat([balanced_labels, balanced_aux_vars], axis=1)
    
    # Create binary disease label for the balanced data
    disease_binary_balanced = (crohn_balanced | ulcerative_balanced).astype(int)
    adata_balanced.obs['disease_binary'] = disease_binary_balanced
    
    # Print final distribution
    print(f"Final binary distribution:")
    print(f"  0 (No disease): {np.sum(disease_binary_balanced == 0)}")
    print(f"  1 (Any disease): {np.sum(disease_binary_balanced == 1)}")
    print(f"Balanced shape: {adata_balanced.shape} (original: {adata.shape})")
    
    return adata_balanced