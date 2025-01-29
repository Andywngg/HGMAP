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

class HMPDownloader:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.hmp_dir = self.data_dir / "hmp"
        self.hmp_dir.mkdir(parents=True, exist_ok=True)
        
        # HMP Data URLs
        self.hmp1_urls = {
            "abundance": "https://downloads.hmpdacc.org/dacc/HMQCP/otu_table_psn_v35.txt.gz",
            "metadata": "https://downloads.hmpdacc.org/dacc/HMQCP/metadata.csv"
        }
        
        self.hmp2_urls = {
            "abundance": "https://downloads.hmpdacc.org/dacc/HMCP2/otu_table_v2.txt.gz",
            "metadata": "https://downloads.hmpdacc.org/dacc/HMCP2/metadata.csv"
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
    
    def download_hmp_data(self, version=1):
        """Download data for a specific HMP version."""
        try:
            version_dir = self.hmp_dir / f"hmp{version}"
            version_dir.mkdir(exist_ok=True)
            
            urls = self.hmp1_urls if version == 1 else self.hmp2_urls
            
            # Download abundance data
            abundance_path = version_dir / "abundance.txt.gz"
            if self.download_file(urls['abundance'], abundance_path, verify=False):
                logger.info(f"Downloaded HMP{version} abundance data")
            
            # Download metadata
            metadata_path = version_dir / "metadata.csv"
            if self.download_file(urls['metadata'], metadata_path, verify=False):
                logger.info(f"Downloaded HMP{version} metadata")
            
            return True
            
        except Exception as e:
            logger.error(f"Error downloading HMP{version} data: {str(e)}")
            return False
    
    def download_all_data(self):
        """Download data from both HMP1 and HMP2."""
        success = False
        
        for version in [1, 2]:
            logger.info(f"Downloading HMP{version} data")
            if self.download_hmp_data(version):
                success = True
        
        if not success:
            logger.error("No data was successfully downloaded")
            logger.info("Please manually download the data from:")
            for version in [1, 2]:
                urls = self.hmp1_urls if version == 1 else self.hmp2_urls
                logger.info(f"\nHMP{version} Data:")
                logger.info(f"Abundance data: {urls['abundance']}")
                logger.info(f"Metadata: {urls['metadata']}")
                logger.info(f"Save to: {self.hmp_dir / f'hmp{version}'}")

def main():
    try:
        downloader = HMPDownloader()
        downloader.download_all_data()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main() 