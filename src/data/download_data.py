import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import tarfile
import gzip
import shutil
from typing import Optional
from biom.table import Table
from biom import save_table, load_table
import urllib.request
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataDownloader:
    """Download and prepare microbiome data"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.agp_dir = self.data_dir / "american_gut"
        self.hmp_dir = self.data_dir / "hmp"
        self.meta_dir = self.data_dir / "metahit"
        self.img_dir = self.data_dir / "img"
        self.gmrepo_dir = self.data_dir / "gmrepo"
        
        # Create directories
        for dir_path in [self.agp_dir, self.hmp_dir, self.meta_dir, 
                        self.img_dir, self.gmrepo_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def download_all_datasets(self):
        """Download all available datasets"""
        datasets = {
            "American Gut Project": self.download_agp_data,
            "Human Microbiome Project": self.download_hmp_data,
            "MetaHIT": self.download_metahit_data,
            "IMG/M": self.download_img_data,
            "GMrepo": self.download_gmrepo_data
        }
        
        for name, download_func in datasets.items():
            logger.info(f"\nDownloading {name} dataset...")
            try:
                download_func()
                logger.info(f"✓ {name} download completed")
            except Exception as e:
                logger.error(f"✗ Error downloading {name}: {e}")
                logger.info(f"Continuing with other datasets...")
    
    def download_agp_data(self):
        """Download American Gut Project data from EBI/ENA"""
        logger.info("Downloading American Gut Project data...")
        
        try:
            # Download abundance table from EBI MGnify
            abundance_url = "https://www.ebi.ac.uk/metagenomics/api/v1/studies/MGYS00001935/pipelines/4.1/file/abundance_table.tsv.gz"
            self._download_file_with_progress(
                abundance_url,
                self.agp_dir / "abundance_table.tsv.gz"
            )
            
            # Extract and convert to BIOM format
            self._convert_to_biom(
                self.agp_dir / "abundance_table.tsv.gz",
                self.agp_dir / "abundance_table.biom"
            )
            
            # Download metadata
            metadata_url = "https://www.ebi.ac.uk/metagenomics/api/v1/studies/MGYS00001935/samples"
            self._download_file_with_progress(
                metadata_url,
                self.agp_dir / "metadata.json"
            )
            
            # Convert metadata to TSV
            self._convert_metadata_to_tsv(
                self.agp_dir / "metadata.json",
                self.agp_dir / "metadata.tsv"
            )
        except Exception as e:
            logger.error(f"Error with AGP data: {e}")
            logger.info("You can manually download AGP data from: https://www.ebi.ac.uk/metagenomics/studies/MGYS00001935")
            raise
    
    def download_hmp_data(self):
        """Download Human Microbiome Project data"""
        logger.info("Downloading HMP data...")
        
        try:
            # Download both HMP1 and HMP2 data
            hmp_versions = {
                "hmp1": "https://downloads.hmpdacc.org/dacc/HMQCP/otu_table_psn_v35.txt.gz",
                "hmp2": "https://downloads.hmpdacc.org/dacc/HMCP2/otu_table_v2.txt.gz"
            }
            
            for version, url in hmp_versions.items():
                output_dir = self.hmp_dir / version
                output_dir.mkdir(exist_ok=True)
                
                # Download abundance table
                self._download_file_with_progress(
                    url,
                    output_dir / "otu_table.gz"
                )
                
                # Convert to BIOM format
                self._convert_to_biom(
                    output_dir / "otu_table.gz",
                    output_dir / "otu_table.biom"
                )
                
                # Download metadata
                metadata_url = f"https://downloads.hmpdacc.org/dacc/{'HMQCP' if version == 'hmp1' else 'HMCP2'}/metadata.csv"
                self._download_file_with_progress(
                    metadata_url,
                    output_dir / "metadata.csv"
                )
        except Exception as e:
            logger.error(f"Error with HMP data: {e}")
            logger.info("You can manually download HMP data from: https://portal.hmpdacc.org/")
            raise
    
    def download_metahit_data(self):
        """Download MetaHIT data"""
        logger.info("Downloading MetaHIT data...")
        
        try:
            # MetaHIT data from EBI
            abundance_url = "https://www.ebi.ac.uk/metagenomics/api/v1/studies/MGYS00001608/pipelines/4.1/file/abundance_table.tsv.gz"
            self._download_file_with_progress(
                abundance_url,
                self.meta_dir / "abundance_table.tsv.gz"
            )
            
            self._convert_to_biom(
                self.meta_dir / "abundance_table.tsv.gz",
                self.meta_dir / "abundance_table.biom"
            )
            
            # Download metadata
            metadata_url = "https://www.ebi.ac.uk/metagenomics/api/v1/studies/MGYS00001608/samples"
            self._download_file_with_progress(
                metadata_url,
                self.meta_dir / "metadata.json"
            )
            
            self._convert_metadata_to_tsv(
                self.meta_dir / "metadata.json",
                self.meta_dir / "metadata.tsv"
            )
        except Exception as e:
            logger.error(f"Error with MetaHIT data: {e}")
            logger.info("You can manually download MetaHIT data from: https://www.ebi.ac.uk/metagenomics/studies/MGYS00001608")
            raise
    
    def download_gmrepo_data(self):
        """Download GMrepo data"""
        logger.info("Downloading GMrepo data...")
        
        try:
            # GMrepo data
            datasets = {
                "healthy": "https://gmrepo.humangut.info/downloads/healthy.tsv.gz",
                "disease": "https://gmrepo.humangut.info/downloads/disease.tsv.gz"
            }
            
            for dataset_type, url in datasets.items():
                output_dir = self.gmrepo_dir / dataset_type
                output_dir.mkdir(exist_ok=True)
                
                self._download_file_with_progress(
                    url,
                    output_dir / f"{dataset_type}_abundance.tsv.gz"
                )
                
                self._convert_to_biom(
                    output_dir / f"{dataset_type}_abundance.tsv.gz",
                    output_dir / f"{dataset_type}_abundance.biom"
                )
        except Exception as e:
            logger.error(f"Error with GMrepo data: {e}")
            logger.info("You can manually download GMrepo data from: https://gmrepo.humangut.info/downloads")
            raise
    
    def download_img_data(self):
        """Download IMG/M data"""
        logger.info("IMG/M data requires registration...")
        logger.info("Please register at https://img.jgi.doe.gov/ to download the data")
        logger.info("After registration, download the data manually and place it in the img/ directory")
    
    def _convert_to_biom(self, input_path: Path, output_path: Path):
        """Convert TSV/CSV file to BIOM format"""
        logger.info(f"Converting {input_path.name} to BIOM format...")
        
        try:
            # Read the input file
            if input_path.suffix == '.gz':
                with gzip.open(input_path, 'rb') as f_in:
                    df = pd.read_csv(f_in, sep='\t', index_col=0)
            else:
                df = pd.read_csv(input_path, sep='\t', index_col=0)
            
            # Convert to BIOM format
            table = Table(
                data=df.values,
                observation_ids=df.index,
                sample_ids=df.columns
            )
            
            # Save BIOM file
            with open(output_path, 'wb') as f:
                save_table(table, f)
        except Exception as e:
            logger.error(f"Error converting {input_path} to BIOM format: {e}")
            raise
    
    def _convert_metadata_to_tsv(self, input_path: Path, output_path: Path):
        """Convert metadata to TSV format"""
        logger.info(f"Converting metadata to TSV format...")
        
        try:
            with open(input_path, 'r') as f:
                metadata = pd.read_json(f)
            metadata.to_csv(output_path, sep='\t')
        except Exception as e:
            logger.error(f"Error converting metadata to TSV: {e}")
            raise
    
    def _download_file_with_progress(self, url: str, output_path: Path):
        """Download a file with progress bar"""
        try:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f, tqdm(
                desc=output_path.name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
                colour='green'
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            logger.info("Trying alternative download method...")
            try:
                urllib.request.urlretrieve(url, output_path)
            except Exception as e2:
                logger.error(f"Alternative download method failed: {e2}")
                raise
    
    def prepare_data(self):
        """Prepare downloaded data for analysis"""
        logger.info("\nPreparing data for analysis...")
        
        # Validate all available datasets
        self._validate_all_data()
        
        logger.info("Data preparation completed")
    
    def _validate_all_data(self):
        """Validate all downloaded datasets"""
        datasets = {
            "AGP": self.agp_dir / "abundance_table.biom",
            "HMP1": self.hmp_dir / "hmp1" / "otu_table.biom",
            "HMP2": self.hmp_dir / "hmp2" / "otu_table.biom",
            "MetaHIT": self.meta_dir / "abundance_table.biom",
            "GMrepo-Healthy": self.gmrepo_dir / "healthy" / "healthy_abundance.biom",
            "GMrepo-Disease": self.gmrepo_dir / "disease" / "disease_abundance.biom"
        }
        
        logger.info("\nValidating downloaded datasets:")
        for name, path in datasets.items():
            if path.exists():
                try:
                    table = load_table(str(path))
                    logger.info(f"✓ {name}: {table.shape[0]} samples, {table.shape[1]} features")
                except Exception as e:
                    logger.error(f"✗ Error validating {name} data: {e}")
            else:
                logger.warning(f"⚠ {name} data not found at {path}")

def run_download():
    """Run the download process"""
    downloader = DataDownloader()
    
    try:
        # Download all datasets
        downloader.download_all_datasets()
        
        # Prepare data
        downloader.prepare_data()
        
    except Exception as e:
        logger.error(f"\nError: {e}")
        logger.info("\nManual download links:")
        logger.info("AGP: https://www.ebi.ac.uk/metagenomics/studies/MGYS00001935")
        logger.info("HMP: https://portal.hmpdacc.org/")
        logger.info("MetaHIT: https://www.ebi.ac.uk/metagenomics/studies/MGYS00001608")
        logger.info("IMG/M: https://img.jgi.doe.gov/ (requires registration)")
        logger.info("GMrepo: https://gmrepo.humangut.info/downloads")

if __name__ == '__main__':
    run_download() 