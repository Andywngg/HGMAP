import requests
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import urllib3
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable SSL warnings
urllib3.disable_warnings()

class MicrobiomeDownloader:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Current available datasets (2024)
        self.datasets = {
            "mgxdb": {
                "name": "MGnify/EBI Metagenomics",
                "base_url": "https://www.ebi.ac.uk/metagenomics/api/v1",
                "studies": [
                    # Original studies
                    "MGYS00005745",  # Human gut microbiome in health and disease
                    "MGYS00001608",  # MetaHIT project
                    
                    # Additional gut microbiome studies
                    "MGYS00001248",  # Human gut microbiome of type 2 diabetes patients
                    "MGYS00002185",  # Gut microbiota in obesity and metabolic disorders
                    "MGYS00004773",  # Colorectal cancer microbiome study
                    "MGYS00003188",  # Inflammatory markers and gut microbiota
                    "MGYS00002673",  # Diet influence on gut microbiome
                    "MGYS00005102"   # Longitudinal gut microbiome study
                ]
            },
            "gmrepo": {
                "name": "GMrepo (Gut Microbiome Database)",
                "base_url": "https://gmrepo.humangut.info/downloads",
                "files": {
                    "healthy": "healthy.tsv",
                    "disease": "disease.tsv"
                }
            }
        }
    
    def download_file(self, url, output_path, headers=None, verify=False):
        """Download a file from URL with progress tracking."""
        try:
            response = requests.get(url, stream=True, headers=headers, verify=verify)
            response.raise_for_status()
            
            # Get file size if available
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                with tqdm(total=total_size, unit='iB', unit_scale=True, desc=output_path.name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            size = f.write(chunk)
                            pbar.update(size)
            
            logger.info(f"Successfully downloaded: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {str(e)}")
            if output_path.exists():
                output_path.unlink()
            return False
    
    def download_mgnify_study(self, study_id):
        """Download data from MGnify/EBI Metagenomics."""
        try:
            study_dir = self.data_dir / "mgnify" / study_id
            study_dir.mkdir(parents=True, exist_ok=True)
            
            # Get study metadata
            metadata_url = f"{self.datasets['mgxdb']['base_url']}/studies/{study_id}"
            response = requests.get(metadata_url)
            response.raise_for_status()
            study_info = response.json()
            
            # Save study metadata
            with open(study_dir / "study_info.json", 'w') as f:
                json.dump(study_info, f, indent=2)
            
            # Get analysis ID for the latest pipeline
            analyses_url = f"{self.datasets['mgxdb']['base_url']}/studies/{study_id}/analyses"
            response = requests.get(analyses_url)
            response.raise_for_status()
            analyses = response.json()
            
            if 'data' in analyses and analyses['data']:
                latest_analysis = analyses['data'][0]  # Get the most recent analysis
                analysis_id = latest_analysis['id']
                pipeline_version = latest_analysis.get('pipeline_version', '4.1')
                
                # Download taxonomic abundance data
                abundance_url = f"{self.datasets['mgxdb']['base_url']}/analyses/{analysis_id}/file/taxonomy-summary"
                abundance_path = study_dir / "abundance.tsv.gz"
                self.download_file(abundance_url, abundance_path)
                
                # Download sample metadata
                samples_url = f"{self.datasets['mgxdb']['base_url']}/studies/{study_id}/samples"
                samples_path = study_dir / "samples.tsv"
                
                # Get all samples (handling pagination)
                all_samples = []
                page = 1
                while True:
                    response = requests.get(f"{samples_url}?page={page}")
                    if response.status_code != 200:
                        break
                    data = response.json()
                    if 'data' in data:
                        all_samples.extend(data['data'])
                    if not data.get('links', {}).get('next'):
                        break
                    page += 1
                
                # Save samples metadata
                if all_samples:
                    samples_df = pd.DataFrame(all_samples)
                    samples_df.to_csv(samples_path, sep='\t', index=False)
                    logger.info(f"Saved metadata for {len(all_samples)} samples")
                
                logger.info(f"Successfully downloaded data for study {study_id}")
                return True
            else:
                logger.error(f"No analyses found for study {study_id}")
                return False
            
        except Exception as e:
            logger.error(f"Error downloading MGnify study {study_id}: {str(e)}")
            return False
    
    def download_gmrepo_data(self):
        """Download data from GMrepo."""
        try:
            gmrepo_dir = self.data_dir / "gmrepo"
            gmrepo_dir.mkdir(parents=True, exist_ok=True)
            
            for data_type, filename in self.datasets['gmrepo']['files'].items():
                url = f"{self.datasets['gmrepo']['base_url']}/{filename}"
                output_path = gmrepo_dir / f"{data_type}.tsv"
                
                logger.info(f"Downloading GMrepo {data_type} data...")
                if self.download_file(url, output_path):
                    logger.info(f"Successfully downloaded GMrepo {data_type} data")
            
            return True
            
        except Exception as e:
            logger.error(f"Error downloading GMrepo data: {str(e)}")
            return False
    
    def download_all_data(self):
        """Download all available datasets."""
        try:
            # Download MGnify studies
            for study_id in self.datasets['mgxdb']['studies']:
                logger.info(f"Downloading MGnify study {study_id}")
                self.download_mgnify_study(study_id)
            
            # Download GMrepo data
            logger.info("Downloading GMrepo data")
            self.download_gmrepo_data()
            
            logger.info("Finished downloading all available datasets")
            
        except Exception as e:
            logger.error(f"Error in download_all_data: {str(e)}")
            
        finally:
            # Print manual download instructions
            logger.info("\nIf any downloads failed, you can manually download the data from:")
            logger.info("\nMGnify/EBI Metagenomics:")
            for study_id in self.datasets['mgxdb']['studies']:
                logger.info(f"- Study {study_id}: https://www.ebi.ac.uk/metagenomics/studies/{study_id}")
            logger.info("\nGMrepo:")
            logger.info("- Visit: https://gmrepo.humangut.info/downloads")

def main():
    try:
        downloader = MicrobiomeDownloader()
        downloader.download_all_data()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main() 