import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from pyclustering import KMedoids

# --- Configuration ---
ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / 'data' / '01_processed'
CLUSTER_DIR = ROOT_DIR / 'data' / '02_clusters'
CLUSTER_DIR.mkdir(exist_ok=True)

# Define strata for number of comorbidities
STRATA_BINS = [0, 2, 4, float('inf')]
STRATA_LABELS = ['Low_MM', 'Mid_MM', 'High_MM']

# Define number of clusters to find per stratum
# NOTE: This is a starting point. We will tune this later with stability analysis.
N_CLUSTERS_PER_STRATUM = 3

def run_mmsp():
    """
    Executes the Multimorbidity-Stratified Phenotyping (MMSP) pipeline.
    1. Loads the C-view (for num.co) and the scaled P-view.
    2. Stratifies patients based on their number of comorbidities.
    3. Within each stratum, performs PCA and then K-Medoids clustering.
    4. Saves the final cluster assignments.
    """
    print("--- Starting MMSP Clustering Pipeline (Path A) ---")
    
    try:
        c_view = pd.read_csv(PROCESSED_DIR / 'C_view.csv')
        p_view_scaled = pd.read_csv(PROCESSED_DIR / 'P_view_scaled.csv')
        print("Loaded C-view and scaled P-view successfully.")
    except FileNotFoundError as e:
        print(f"[ERROR] Could not find a required file: {e}. Please run '02_create_final_views.py'.")
        return

    # --- 1. Stratify Patients by Comorbidity Count ---
    c_view['stratum'] = pd.cut(
        c_view['num.co'], 
        bins=STRATA_BINS, 
        labels=STRATA_LABELS, 
        right=False
    )
    print("Stratified patients into:", STRATA_LABELS)
    
    # Prepare a DataFrame to store final cluster labels
    final_labels = pd.Series(index=p_view_scaled.index, dtype='object', name='mmsp_cluster')

    # --- 2. Run Clustering Within Each Stratum ---
    for stratum in STRATA_LABELS:
        print(f"\n--- Processing Stratum: {stratum} ---")
        
        # Get the indices of patients in the current stratum
        stratum_indices = c_view[c_view['stratum'] == stratum].index
        stratum_p_view = p_view_scaled.loc[stratum_indices]
        
        if len(stratum_p_view) < N_CLUSTERS_PER_STRATUM:
            print(f"Skipping stratum '{stratum}' due to insufficient data points.")
            continue
            
        # --- 3. Dimensionality Reduction with PCA ---
        pca = PCA(n_components=0.80, random_state=42) # Retain components explaining 80% of variance
        p_view_pca = pca.fit_transform(stratum_p_view)
        print(f"PCA complete. Reduced to {p_view_pca.shape[1]} components.")

        # --- 4. K-Medoids Clustering ---
        kmedoids = KMedoids(
            n_clusters=N_CLUSTERS_PER_STRATUM, 
            method='pam', 
            init='k-medoids++', 
            random_state=42
        )
        clusters = kmedoids.fit_predict(p_view_pca)
        print(f"K-Medoids complete. Found {N_CLUSTERS_PER_STRATUM} clusters.")

        # --- 5. Assign and Store Labels ---
        # Create unique labels like 'Low_MM_0', 'Mid_MM_1', etc.
        stratum_cluster_labels = [f"{stratum}_{c}" for c in clusters]
        final_labels.loc[stratum_indices] = stratum_cluster_labels

    # --- Save the final cluster assignments ---
    final_labels_df = final_labels.to_frame()
    output_path = CLUSTER_DIR / 'mmsp_clusters.csv'
    final_labels_df.to_csv(output_path)
    
    print("\n" + "="*50)
    print("✅ MMSP Clustering Complete!")
    print(f"Final cluster labels saved to: {output_path}")
    print("\nValue Counts of Final Clusters:")
    print(final_labels_df['mmsp_cluster'].value_counts())
    print("="*50)

if __name__ == "__main__":
    run_mmsp()
