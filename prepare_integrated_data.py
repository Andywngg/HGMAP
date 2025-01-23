import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def prepare_integrated_data():
    try:
        # Set up paths
        processed_dir = Path("data/processed")
        integrated_dir = Path("data/integrated")
        integrated_dir.mkdir(parents=True, exist_ok=True)
        
        # Process 2020 study data
        df_2020_taxonomy = pd.read_csv(processed_dir / "41467_2020_18476_MOESM4_ESM.csv")
        df_2020_health = pd.read_csv(processed_dir / "41467_2020_18476_MOESM5_ESM.csv")
        
        # Process 2024 study data
        df_2024_metadata = pd.read_csv(processed_dir / "41467_2024_51651_MOESM5_ESM.csv")
        
        # Create integrated metadata
        metadata = df_2024_metadata.copy()
        metadata.to_csv(integrated_dir / "integrated_metadata.csv", index=False)
        logging.info(f"Saved integrated metadata with {len(metadata)} samples")
        
        # Create integrated taxonomy reference
        taxonomy = df_2020_taxonomy.copy()
        taxonomy.to_csv(integrated_dir / "taxonomy_reference.csv", index=False)
        logging.info(f"Saved taxonomy reference with {len(taxonomy)} species")
        
        # Create integrated health associations
        health_assoc = df_2020_health.copy()
        health_assoc.to_csv(integrated_dir / "health_associations.csv", index=False)
        logging.info(f"Saved health associations with {len(health_assoc)} entries")
        
    except Exception as e:
        logging.error(f"Error preparing integrated data: {str(e)}")
        raise

if __name__ == "__main__":
    prepare_integrated_data() 