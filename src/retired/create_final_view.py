import pandas as pd
import yaml
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
ROOT_DIR = Path(__file__).resolve().parents[1]
CLEANED_DATA_PATH = ROOT_DIR / 'data' / '01_processed' / 'support_preprocessed.csv'
SCHEMA_PATH = ROOT_DIR / 'support_schema.yml'
OUTPUT_DIR = ROOT_DIR / 'data' / '01_processed'

def create_final_views():
    """
    Loads the fully cleaned and preprocessed data, splits it into the C, P, and S
    views, scales the P-view, and saves the final analysis-ready files.
    """
    print("--- Starting Final View Creation Script ---")
    
    try:
        df = pd.read_csv(CLEANED_DATA_PATH)
        with open(SCHEMA_PATH, 'r') as f:
            schema = yaml.safe_load(f)
        print("Loaded cleaned data and schema successfully.")
    except FileNotFoundError as e:
        print(f"[ERROR] Could not find a required file: {e}. Please run previous scripts.")
        return

    # --- Define column sets from schema (CORRECTED SECTION) ---
    # We now access the 'feature_views' key first to match the YAML structure.
    p_view_cols = schema['feature_views']['P_view']
    s_view_cols = schema['feature_views']['S_view']
    c_view_cols_raw = schema['feature_views']['C_view']
    
    # Manually list the one-hot encoded columns based on our preprocessing script
    ohe_race_cols = [c for c in df.columns if 'race_' in c]
    ohe_dnr_cols = [c for c in df.columns if 'dnr_' in c]
    # We need to find the original columns in the schema to replace them
    s_view_cols_final = [c for c in s_view_cols if c not in ['race', 'dnr']] + ohe_race_cols + ohe_dnr_cols
    s_view_cols_final = [c for c in s_view_cols_final if c in df.columns] # Ensure all columns exist

    ohe_dz_cols = [c for c in df.columns if 'dzgroup_' in c or 'dzclass_' in c]
    c_view_cols_final = [c for c in c_view_cols_raw if c not in ['dzgroup', 'dzclass']] + ohe_dz_cols
    c_view_cols_final = [c for c in c_view_cols_final if c in df.columns]

    # --- Create the View DataFrames ---
    C_view = df[c_view_cols_final]
    P_view = df[p_view_cols]
    S_view = df[s_view_cols_final]
    print("Separated data into C, P, and S views.")

    # --- Scale the Physiology (P) View ---
    scaler = StandardScaler()
    P_view_scaled_array = scaler.fit_transform(P_view)
    P_view_scaled = pd.DataFrame(P_view_scaled_array, columns=p_view_cols)
    print("P-view has been scaled using StandardScaler.")

    # --- Save the Final Views ---
    OUTPUT_DIR.mkdir(exist_ok=True)
    C_view.to_csv(OUTPUT_DIR / 'C_view.csv', index=False)
    P_view_scaled.to_csv(OUTPUT_DIR / 'P_view_scaled.csv', index=False)
    S_view.to_csv(OUTPUT_DIR / 'S_view.csv', index=False)
    
    print(f"\n✅ Final analysis-ready views saved successfully to: {OUTPUT_DIR}")

if __name__ == "__main__":
    create_final_views()

