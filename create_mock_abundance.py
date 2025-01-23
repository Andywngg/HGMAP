import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_mock_abundance():
    try:
        # Set up paths
        integrated_dir = Path("data/integrated")
        
        # Load taxonomy reference to get species names
        taxonomy = pd.read_csv(integrated_dir / "taxonomy_reference.csv", skiprows=[0])  # Skip the long header
        species_names = taxonomy['Species'].values
        
        # Load metadata to get sample IDs
        metadata = pd.read_csv(integrated_dir / "integrated_metadata.csv")
        sample_ids = np.arange(len(metadata))  # Use numeric indices for samples
        
        # Generate mock abundance data
        n_samples = len(metadata)
        n_species = len(species_names)
        
        # Create sparse abundance matrix (most species will have zero abundance)
        abundance_matrix = np.zeros((n_samples, n_species))
        
        # For each sample, randomly select 20-30% of species to have non-zero abundance
        for i in range(n_samples):
            n_present = np.random.randint(int(0.2 * n_species), int(0.3 * n_species))
            present_species = np.random.choice(n_species, size=n_present, replace=False)
            abundance_matrix[i, present_species] = np.random.dirichlet(np.ones(n_present))
        
        # Create DataFrame
        abundance_df = pd.DataFrame(
            abundance_matrix,
            columns=species_names,
            index=sample_ids
        )
        
        # Save to file
        output_file = integrated_dir / "abundance_matrix.csv"
        abundance_df.to_csv(output_file)
        logging.info(f"Created mock abundance matrix with shape {abundance_df.shape}")
        
    except Exception as e:
        logging.error(f"Error creating mock abundance data: {str(e)}")
        raise

if __name__ == "__main__":
    create_mock_abundance() 