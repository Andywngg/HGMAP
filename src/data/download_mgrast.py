import requests
import pandas as pd
import numpy as np
from pathlib import Path
import time
import logging
import urllib3
import ftplib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable SSL warnings
urllib3.disable_warnings()

class MGRASTDownloader:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.mgrast_dir = self.data_dir / "mgrast"
        self.mgrast_dir.mkdir(parents=True, exist_ok=True)
        
        # MG-RAST FTP details
        self.ftp_host = "ftp.mg-rast.org"
        
        # List of gut microbiome studies to download
        self.studies = {
            "hmp": {
                "abundance_url": "https://www.mg-rast.org/download.cgi?project=mgp93&file=abundance_profiles.zip",
                "metadata_url": "https://www.mg-rast.org/download.cgi?project=mgp93&file=metadata.zip"
            },
            "agp": {
                "abundance_url": "https://www.mg-rast.org/download.cgi?project=mgp401&file=abundance_profiles.zip",
                "metadata_url": "https://www.mg-rast.org/download.cgi?project=mgp401&file=metadata.zip"
            },
            "disease": {
                "abundance_url": "https://www.mg-rast.org/download.cgi?project=mgp17766&file=abundance_profiles.zip",
                "metadata_url": "https://www.mg-rast.org/download.cgi?project=mgp17766&file=metadata.zip"
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
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
            
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {str(e)}")
            return False
    
    def download_study_data(self, study_name, study_info):
        """Download data for a specific study."""
        try:
            study_dir = self.mgrast_dir / study_name
            study_dir.mkdir(exist_ok=True)
            
            # Download abundance data
            abundance_path = study_dir / "abundance.zip"
            if self.download_file(study_info['abundance_url'], abundance_path, verify=False):
                logger.info(f"Downloaded abundance data for {study_name}")
            
            # Download metadata
            metadata_path = study_dir / "metadata.zip"
            if self.download_file(study_info['metadata_url'], metadata_path, verify=False):
                logger.info(f"Downloaded metadata for {study_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error downloading data for {study_name}: {str(e)}")
            return False
    
    def download_all_data(self):
        """Download data from all specified studies."""
        success = False
        
        for study_name, study_info in self.studies.items():
            logger.info(f"Downloading data for study {study_name}")
            if self.download_study_data(study_name, study_info):
                success = True
        
        if not success:
            logger.error("No data was successfully downloaded")
            logger.info("Please manually download the data from:")
            for study_name, study_info in self.studies.items():
                logger.info(f"\n{study_name.upper()} Study:")
                logger.info(f"Abundance data: {study_info['abundance_url']}")
                logger.info(f"Metadata: {study_info['metadata_url']}")
                logger.info(f"Save to: {self.mgrast_dir / study_name}")

def main():
    try:
        downloader = MGRASTDownloader()
        downloader.download_all_data()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main() 