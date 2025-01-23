import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import entropy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class MicrobiomeProcessor:
    def __init__(self, n_pca_components=50):
        self.n_pca_components = n_pca_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_pca_components)
        
    def load_data(self, integrated_dir):
        """Load integrated data from CSV files."""
        try:
            metadata = pd.read_csv(Path(integrated_dir) / "integrated_metadata.csv")
            taxonomy = pd.read_csv(Path(integrated_dir) / "taxonomy_reference.csv")
            health_assoc = pd.read_csv(Path(integrated_dir) / "health_associations.csv")
            
            logging.info(f"Loaded metadata with {len(metadata)} samples")
            logging.info(f"Loaded taxonomy with {len(taxonomy)} species")
            logging.info(f"Loaded health associations with {len(health_assoc)} entries")
            
            return metadata, taxonomy, health_assoc
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise
    
    def calculate_diversity_metrics(self, abundance_data):
        """Calculate alpha diversity metrics."""
        shannon_diversity = abundance_data.apply(
            lambda x: entropy(x[x > 0]), axis=1
        )
        species_richness = (abundance_data > 0).sum(axis=1)
        evenness = shannon_diversity / np.log(species_richness)
        
        diversity_metrics = pd.DataFrame({
            'shannon_diversity': shannon_diversity,
            'species_richness': species_richness,
            'evenness': evenness
        })
        return diversity_metrics
    
    def calculate_pca_features(self, abundance_data):
        """Calculate PCA features from abundance data."""
        try:
            scaled_data = self.scaler.fit_transform(abundance_data)
            pca_features = self.pca.fit_transform(scaled_data)
            
            variance_explained = self.pca.explained_variance_ratio_
            logging.info(f"Total variance explained by {self.n_pca_components} components: {sum(variance_explained):.3f}")
            
            pca_df = pd.DataFrame(
                pca_features,
                columns=[f'PC{i+1}' for i in range(self.n_pca_components)],
                index=abundance_data.index
            )
            return pca_df
        except Exception as e:
            logging.error(f"Error in PCA calculation: {str(e)}")
            raise
    
    def process_data(self, input_dir, output_dir):
        """Main processing function."""
        try:
            # Create output directory if it doesn't exist
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load integrated data
            metadata, taxonomy, health_assoc = self.load_data(input_dir)
            
            # Load and process abundance data
            abundance_files = list(Path(input_dir).glob("abundance_*.csv"))
            abundance_data = pd.concat([pd.read_csv(f) for f in abundance_files])
            
            # Calculate features
            diversity_metrics = self.calculate_diversity_metrics(abundance_data)
            pca_features = self.calculate_pca_features(abundance_data)
            
            # Combine features
            combined_features = pd.concat([
                metadata,
                diversity_metrics,
                pca_features
            ], axis=1)
            
            # Save processed data
            combined_features.to_csv(output_dir / "processed_features.csv", index=False)
            logging.info(f"Saved processed features with {combined_features.shape[1]} features")
            
            # Save feature summary
            feature_summary = pd.DataFrame({
                'feature_type': ['Metadata', 'Diversity', 'PCA'],
                'count': [
                    len(metadata.columns),
                    len(diversity_metrics.columns),
                    len(pca_features.columns)
                ]
            })
            feature_summary.to_csv(output_dir / "feature_summary.csv", index=False)
            
        except Exception as e:
            logging.error(f"Error in data processing: {str(e)}")
            raise

def main():
    try:
        input_dir = Path("data/integrated")
        output_dir = Path("data/features")
        
        processor = MicrobiomeProcessor(n_pca_components=50)
        processor.process_data(input_dir, output_dir)
        
        logging.info("Data processing completed successfully")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 