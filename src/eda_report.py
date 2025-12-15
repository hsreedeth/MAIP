import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

# --- Configuration ---
# Use pathlib to make file paths robust and independent of the OS.
# This script assumes it's in 'src/', so we navigate up one level to find the project root.
ROOT_DIR = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT_DIR / 'support_schema.yml'
RAW_DATA_PATH = ROOT_DIR / 'data' / '00_raw' / 'support2.csv'

# --- Core Functions ---

def load_schema(path=SCHEMA_PATH):
    """
    Loads and parses the YAML schema file.

    Args:
        path (Path, optional): The path to the schema file. Defaults to SCHEMA_PATH.

    Returns:
        dict: The parsed YAML content, or None if an error occurs.
    """
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Schema file not found at {path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None

def load_data(path=RAW_DATA_PATH):
    """
    Loads the raw dataset from the specified CSV file.

    Args:
        path (Path, optional): The path to the raw data CSV. Defaults to RAW_DATA_PATH.

    Returns:
        pd.DataFrame: The loaded DataFrame, or None if the file is not found.
    """
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {path}")
        print("Please ensure 'support2.csv' is in the 'data/00_raw/' directory.")
        return None

def missing_summary(data, plot_heatmap=False):
    """
    Summarizes missing data and optionally plots a heatmap.

    Args:
        data (pd.DataFrame): The dataset to analyze.
        plot_heatmap (bool): If True, displays a heatmap of missing values.

    Returns:
        pd.DataFrame: A table summarizing features with missing data.
    """
    missing_table = (
        data.isnull().sum()
        .to_frame(name='Missing Count')
        .assign(
            Missing_Percent=lambda x: 100 * x['Missing Count'] / len(data),
            Dtype=data.dtypes
        )
        .query("`Missing Count` > 0")
        .sort_values('Missing Count', ascending=False)
        .reset_index(names='Feature')
    )

    if plot_heatmap:
        missing_cols = missing_table['Feature'].tolist()
        if not missing_cols:
            print("No missing values to plot.")
            return missing_table
            
        plt.figure(figsize=(15, 8))
        sns.heatmap(data[missing_cols].isnull(), cbar=False, cmap='viridis', yticklabels=False)
        plt.title("Heatmap of Missing Values", fontsize=16)
        plt.xlabel("Features with Missing Data")
        plt.tight_layout()
        plt.show()

    return missing_table

def generate_full_report():
    """
    Generates and prints a full EDA report to the console.
    This function acts as the main orchestrator for the 'report' action.
    """
    print("=" * 80)
    print("SUPPORT-II Dataset: Exploratory Data Analysis Report")
    print("=" * 80)

    # Load necessary data and schema files
    schema = load_schema()
    df = load_data()
    if schema is None or df is None:
        return

    # 1. Dataset Overview
    print("\n--- 1. Dataset Overview ---\n")
    print(f"Shape of the dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 2. Variable Descriptions from Schema
    print("\n--- 2. Variable Descriptions ---\n")
    if 'variables' in schema:
        # Create a DataFrame for nice, aligned printing
        desc_df = pd.DataFrame.from_dict(schema.get('variables', {}), orient='index')
        print(desc_df[['type', 'description']].to_string())
    
    # 3. Missing Data Summary
    print("\n--- 3. Missing Data Summary ---\n")
    missing_df = missing_summary(df, plot_heatmap=False)
    if not missing_df.empty:
        # Use .to_string() to ensure all rows are printed
        print(missing_df.to_string(index=False))
    else:
        print("No missing values found in the dataset.")

    # 4. Summary Statistics for Numerical Columns
    print("\n--- 4. Numerical Data Summary ---\n")
    print(df.describe().to_string())
    
    # 5. Value Counts for Categorical Columns
    print("\n--- 5. Categorical Data Summary ---\n")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        print(f"\nDistribution for '{col}':")
        print(df[col].value_counts().to_frame().to_string())
        print("-" * 30)

    print("\n" + "=" * 80)
    print("End of Report")
    print("=" * 80)

def main():
    """
    Main function to handle command-line arguments using argparse.
    This allows the script to be called from the terminal with different options.
    """
    parser = argparse.ArgumentParser(
        description="EDA Tool for the SUPPORT-II Dataset. Provides reports and metadata.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'action', 
        choices=['report', 'get-dictionary-url'],
        help=(
            "The action to perform:\n"
            "  'report'             - Generates a full summary of the dataset.\n"
            "  'get-dictionary-url' - Provides the web link to the data dictionary."
        )
    )
    
    args = parser.parse_args()

    if args.action == 'report':
        generate_full_report()
    elif args.action == 'get-dictionary-url':
        schema = load_schema()
        if schema and 'data_dictionary_url' in schema:
            print("\nData Dictionary URL:")
            print(schema['data_dictionary_url'])
            print("\nPlease open this link in a web browser for detailed variable information.\n")
        else:
            print("Could not find the data dictionary URL in the schema file.")

# This standard Python construct ensures that main() is called only when the script
# is executed directly (not when it's imported as a module into another script).
if __name__ == "__main__":
    main()
