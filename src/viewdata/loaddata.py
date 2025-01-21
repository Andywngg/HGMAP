import pandas as pd
import os
import config.constants as constants

def load_excel_data(file_paths):
    """
    Load multiple Excel files and concatenate them.
    
    Args:
        file_paths (dict): Dictionary of file paths to load
    
    Returns:
        pd.DataFrame: Combined dataset
    """
    dataframes = []
    for name, path in file_paths.items():
        try:
            # Try reading Excel file
            df = pd.read_excel(path)
            df['source_file'] = name  # Add source file column for tracking
            dataframes.append(df)
        except Exception as e:
            print(f"Error loading {name}: {e}")
    
    return pd.concat(dataframes, ignore_index=True)

def inspect_data(data):
    """
    Perform comprehensive data inspection.
    
    Args:
        data (pd.DataFrame): Input dataframe
    """
    print("Dataset Overview:")
    print(f"Total Rows: {len(data)}")
    print(f"Total Columns: {len(data.columns)}")
    
    print("\nMissing Values:")
    print(data.isnull().sum())
    
    print("\nColumn Types:")
    print(data.dtypes)
    
    print("\nBasic Statistical Summary:")
    print(data.describe())

# Load and inspect data
taxonomy_data = load_excel_data(constants.TAXONOMY_FILES)
inspect_data(taxonomy_data)

# Optional: Save consolidated dataset
taxonomy_data.to_csv("consolidated_microbiome_data.csv", index=False)