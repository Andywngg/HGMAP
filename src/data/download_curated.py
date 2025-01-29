import requests
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import urllib3
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable SSL warnings
urllib3.disable_warnings()

class CuratedDownloader:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.curated_dir = self.data_dir / "curated"
        self.curated_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset URLs from curatedMetagenomicData
        self.datasets = {
            "MetaHIT_2012": {
                "abundance": "https://zenodo.org/record/840333/files/MetaHIT_2012.metaphlan_bugs_list.stool.tsv.gz",
                "metadata": "https://zenodo.org/record/840333/files/MetaHIT_2012.metadata.stool.tsv"
            },
            "HMP_2012": {
                "abundance": "https://zenodo.org/record/840333/files/HMP_2012.metaphlan_bugs_list.stool.tsv.gz",
                "metadata": "https://zenodo.org/record/840333/files/HMP_2012.metadata.stool.tsv"
            },
            "Qin_2012_obesity": {
                "abundance": "https://zenodo.org/record/840333/files/Qin_2012.metaphlan_bugs_list.stool.tsv.gz",
                "metadata": "https://zenodo.org/record/840333/files/Qin_2012.metadata.stool.tsv"
            },
            "Karlsson_2013_diabetes": {
                "abundance": "https://zenodo.org/record/840333/files/Karlsson_2013.metaphlan_bugs_list.stool.tsv.gz",
                "metadata": "https://zenodo.org/record/840333/files/Karlsson_2013.metadata.stool.tsv"
            }
        }
    
    def download_file(self, url, output_path, verify=False):
        """Download a file from URL with progress tracking."""
        try:
            response = requests.get(url, stream=True, verify=verify)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            
            with open(output_path, 'wb') as f:
                with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            size = f.write(chunk)
                            pbar.update(size)
            
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {str(e)}")
            if output_path.exists():
                output_path.unlink()  # Remove partial download
            return False
    
    def download_dataset(self, dataset_name, dataset_info):
        """Download data for a specific dataset."""
        try:
            dataset_dir = self.curated_dir / dataset_name
            dataset_dir.mkdir(exist_ok=True)
            
            # Download abundance data
            abundance_path = dataset_dir / "abundance.tsv.gz"
            if self.download_file(dataset_info['abundance'], abundance_path, verify=False):
                logger.info(f"Downloaded {dataset_name} abundance data")
            
            # Download metadata
            metadata_path = dataset_dir / "metadata.tsv"
            if self.download_file(dataset_info['metadata'], metadata_path, verify=False):
                logger.info(f"Downloaded {dataset_name} metadata")
            
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {dataset_name} data: {str(e)}")
            return False
    
    def download_all_data(self):
        """Download all available datasets."""
        success = False
        
        for dataset_name, dataset_info in self.datasets.items():
            logger.info(f"Downloading {dataset_name} data")
            if self.download_dataset(dataset_name, dataset_info):
                success = True
        
        if not success:
            logger.error("No data was successfully downloaded")
            logger.info("Please manually download the data from:")
            for dataset_name, dataset_info in self.datasets.items():
                logger.info(f"\n{dataset_name}:")
                logger.info(f"Abundance data: {dataset_info['abundance']}")
                logger.info(f"Metadata: {dataset_info['metadata']}")
                logger.info(f"Save to: {self.curated_dir / dataset_name}")

def main():
    try:
        downloader = CuratedDownloader()
        downloader.download_all_data()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main() 