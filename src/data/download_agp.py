import requests
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import gzip
import json
from tqdm import tqdm
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AGPDownloader:
    """Download and process American Gut Project data"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.agp_dir = self.data_dir / "american_gut"
        self.agp_dir.mkdir(parents=True, exist_ok=True)
    
    def download_abundance_data(self):
        """Download abundance data"""
        logger.info("Downloading abundance data...")
        
        # Try multiple potential URLs
        urls = [
            "https://raw.githubusercontent.com/biocore/American-Gut-Project/master/data/AG/AG_100nt_even1k.biom",
            "https://raw.githubusercontent.com/biocore/American-Gut-Project/master/data/AG/100nt/AG_100nt_even1k.biom",
            "https://raw.githubusercontent.com/biocore/American-Gut-Project/master/data/abundance_tables/taxon_table.biom"
        ]
        
        success = False
        for url in urls:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    # Save raw BIOM file
                    with open(self.agp_dir / "abundance_table.biom", 'wb') as f:
                        f.write(response.content)
                    
                    # Parse BIOM format
                    data = json.loads(response.content)
                    
                    # Extract data from BIOM format
                    if data['matrix_type'] == 'sparse':
                        # Get dimensions
                        n_samples = len(data['columns'])
                        n_taxa = len(data['rows'])
                        
                        # Create empty matrix
                        abundance_matrix = np.zeros((n_samples, n_taxa))
                        
                        # Fill in sparse values
                        for i, j, value in data['data']:
                            abundance_matrix[j, i] = value
                        
                        # Create DataFrame
                        abundance_df = pd.DataFrame(
                            abundance_matrix,
                            columns=[row['id'] for row in data['rows']],
                            index=[col['id'] for col in data['columns']]
                        )
                    else:
                        # Dense matrix
                        abundance_df = pd.DataFrame(
                            data['data'],
                            columns=[row['id'] for row in data['rows']],
                            index=[col['id'] for col in data['columns']]
                        )
                    
                    # Save as TSV
                    abundance_df.to_csv(self.agp_dir / "abundance_table.tsv", sep='\t')
                    success = True
                    break
                    
            except Exception as e:
                logger.warning(f"Failed to download from {url}: {e}")
                continue
        
        if not success:
            raise ValueError("Failed to download abundance data from all sources")
    
    def download_metadata(self):
        """Download metadata"""
        logger.info("Downloading metadata...")
        
        try:
            # Try multiple potential URLs
            urls = [
                "https://raw.githubusercontent.com/biocore/American-Gut-Project/master/data/AG/AG_metadata.txt",
                "https://raw.githubusercontent.com/biocore/American-Gut-Project/master/data/AG/metadata.txt",
                "https://raw.githubusercontent.com/biocore/American-Gut-Project/master/data/metadata/AG_metadata.txt"
            ]
            
            success = False
            for url in urls:
                response = requests.get(url)
                if response.status_code == 200:
                    # Save raw metadata
                    with open(self.agp_dir / "metadata_raw.tsv", 'w') as f:
                        f.write(response.text)
                    success = True
                    break
            
            if not success:
                raise ValueError("Failed to download metadata from all sources")
            
            # Process metadata
            metadata_df = pd.read_csv(self.agp_dir / "metadata_raw.tsv", sep='\t', low_memory=False)
            
            # Try different column names for health status
            health_column_sets = [
                ['DIAGNOSED_DIABETES', 'DIAGNOSED_IBD', 'DIAGNOSED_IBS', 
                 'DIAGNOSED_AUTOIMMUNE', 'DIAGNOSED_CARDIOVASCULAR_DISEASE'],
                ['diagnosed_diabetes', 'diagnosed_ibd', 'diagnosed_ibs',
                 'diagnosed_autoimmune', 'diagnosed_cardiovascular_disease'],
                ['Disease_Status', 'Health_Status', 'Clinical_Condition']
            ]
            
            for columns in health_column_sets:
                if all(col in metadata_df.columns for col in columns):
                    metadata_df['health_status'] = metadata_df[columns].apply(
                        lambda row: 'Non-healthy' if any(
                            pd.notna(val) and str(val).lower() in ['yes', 'disease', 'sick', 'abnormal']
                            for val in row
                        ) else 'Healthy',
                        axis=1
                    )
                    break
            else:
                # If no health columns found, try text search in all columns
                health_status = []
                for _, row in metadata_df.iterrows():
                    is_healthy = True
                    for val in row.values:
                        if pd.notna(val) and isinstance(val, str):
                            if any(term in val.lower() for term in ['disease', 'disorder', 'syndrome', 'infection']):
                                is_healthy = False
                                break
                    health_status.append('Non-healthy' if not is_healthy else 'Healthy')
                metadata_df['health_status'] = health_status
            
            # Try different sample ID column names
            sample_id_columns = ['#SampleID', 'sample_name', 'sample_id', 'SampleID']
            for col in sample_id_columns:
                if col in metadata_df.columns:
                    metadata_df = metadata_df[[col, 'health_status']]
                    metadata_df = metadata_df.rename(columns={col: 'sample_id'})
                    break
            else:
                raise ValueError("Could not find sample ID column")
            
            # Save processed metadata
            metadata_df.to_csv(self.agp_dir / "metadata.csv", index=False)
            
            logger.info(f"Downloaded metadata for {len(metadata_df)} samples")
            return metadata_df
            
        except Exception as e:
            logger.error(f"Error downloading metadata: {e}")
            raise
    
    def process_abundance_data(self):
        """Process abundance table"""
        logger.info("Processing abundance data...")
        
        try:
            # Load abundance data
            abundance_df = pd.read_csv(
                self.agp_dir / "abundance_table.tsv",
                sep='\t',
                index_col=0
            )
            
            # Basic filtering
            # Remove rare taxa (present in less than 1% of samples)
            prevalence = (abundance_df > 0).mean()
            abundant_taxa = prevalence[prevalence >= 0.01].index
            abundance_df = abundance_df[abundant_taxa]
            
            # Log transform
            abundance_df = np.log1p(abundance_df)
            
            # Save processed data
            abundance_df.to_csv(self.agp_dir / "abundance_processed.csv")
            
            logger.info(f"Processed abundance data shape: {abundance_df.shape}")
            return abundance_df
            
        except Exception as e:
            logger.error(f"Error processing abundance data: {e}")
            raise
    
    def download_and_process(self):
        """Download and process all AGP data"""
        try:
            # Download data
            self.download_abundance_data()
            metadata_df = self.download_metadata()
            
            # Process abundance data
            abundance_df = self.process_abundance_data()
            
            # Ensure samples match between abundance and metadata
            common_samples = set(abundance_df.index) & set(metadata_df['sample_id'])
            
            if not common_samples:
                raise ValueError("No common samples found between abundance and metadata")
            
            abundance_df = abundance_df.loc[list(common_samples)]
            metadata_df = metadata_df[metadata_df['sample_id'].isin(common_samples)]
            
            logger.info(f"Final dataset has {len(common_samples)} samples")
            
            # Save final datasets
            abundance_df.to_csv(self.agp_dir / "abundance_final.csv")
            metadata_df.to_csv(self.agp_dir / "metadata_final.csv", index=False)
            
            return abundance_df, metadata_df
            
        except Exception as e:
            logger.error(f"Error downloading AGP data: {e}")
            raise

if __name__ == "__main__":
    downloader = AGPDownloader()
    abundance_df, metadata_df = downloader.download_and_process() 