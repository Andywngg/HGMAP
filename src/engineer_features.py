from pathlib import Path
import logging
import pandas as pd
from features.microbiome_features import MicrobiomeFeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the feature engineering process"""
    try:
        # Set up paths
        data_dir = Path("data/integrated")
        output_dir = Path("data/features")
        
        logger.info("Starting feature engineering process...")
        
        # Load integrated data
        metadata_df = pd.read_csv(data_dir / "integrated_metadata.csv")
        taxonomy_df = pd.read_csv(data_dir / "taxonomy_reference.csv")
        health_assoc_df = pd.read_csv(data_dir / "health_associations.csv")
        
        # Initialize feature engineer
        feature_engineer = MicrobiomeFeatureEngineer(
            n_pca_components=50,
            min_prevalence=0.1,
            min_abundance=0.001
        )
        
        # Load abundance data (assuming it's available)
        abundance_files = list(data_dir.glob("abundance_*.csv"))
        if not abundance_files:
            raise FileNotFoundError("No abundance data files found")
        
        abundance_df = pd.concat(
            [pd.read_csv(f) for f in abundance_files],
            axis=0,
            ignore_index=True
        )
        
        # Engineer features
        features_df = feature_engineer.engineer_features(
            abundance_df=abundance_df,
            taxonomy_ref=taxonomy_df,
            health_assoc=health_assoc_df,
            metadata_df=metadata_df
        )
        
        # Save features
        feature_engineer.save_features(
            features_df=features_df,
            output_dir=output_dir
        )
        
        # Log feature statistics
        logger.info("\nFeature Engineering Summary:")
        logger.info(f"Total samples: {len(features_df)}")
        logger.info(f"Total features: {len(features_df.columns)}")
        
        # Log feature types
        feature_types = {
            'Diversity': len([col for col in features_df.columns if 'diversity' in col.lower()]),
            'PCA': len([col for col in features_df.columns if col.startswith('PC')]),
            'Ratios': len([col for col in features_df.columns if 'ratio' in col.lower()]),
            'Functional': len([col for col in features_df.columns if 'abundance' in col.lower()]),
            'Metadata': len([col for col in features_df.columns if 'encoded' in col.lower()])
        }
        
        logger.info("\nFeature Types:")
        for feat_type, count in feature_types.items():
            logger.info(f"{feat_type}: {count}")
        
        logger.info("\nFeature engineering completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in feature engineering process: {e}")
        raise

if __name__ == "__main__":
    main() 