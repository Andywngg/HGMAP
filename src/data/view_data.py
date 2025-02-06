import pandas as pd
from pathlib import Path

def view_tsv_file(file_path):
    """View the first few lines of a TSV file."""
    print(f"\nViewing contents of {file_path}:")
    print("-" * 80)
    try:
        df = pd.read_csv(file_path, sep='\t', nrows=5)
        print("\nFirst 5 rows:")
        print(df)
        print("\nColumns:")
        print(df.columns.tolist())
        print("\nData Info:")
        print(df.info())
    except Exception as e:
        print(f"Error reading file: {str(e)}")

def main():
    # View MGnify data
    for study_id in ["MGYS00001578", "MGYS00002673", "MGYS00004766"]:
        study_dir = Path(f"data/mgnify/{study_id}")
        
        abundance_file = study_dir / "abundance.tsv"
        if abundance_file.exists():
            view_tsv_file(abundance_file)
        
        samples_file = study_dir / "samples.tsv"
        if samples_file.exists():
            view_tsv_file(samples_file)

if __name__ == "__main__":
    main() 