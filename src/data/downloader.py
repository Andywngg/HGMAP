import os
from pathlib import Path
import requests
from tqdm import tqdm
import gzip
import shutil
import logging
import json
from urllib.parse import urljoin
import urllib3

# Disable SSL warnings since we'll handle verification manually
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataDownloader:
    """Download microbiome datasets"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Set up session for requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            "american_gut",
            "hmp/hmp1",
            "hmp/hmp2",
            "metahit",
            "gmrepo",
            "diabimmune",
            "ibd",
            "qiita/obesity",
            "qiita/ibd",
            "processed"
        ]
        
        for directory in directories:
            (self.data_dir / directory).mkdir(parents=True, exist_ok=True)
    
    def _download_file(self, url: str, path: Path, verify=True):
        """Download a file from a URL to a path."""
        try:
            if verify:
                self.logger.info(f"Downloading {path.name}...")
            
            # Make the request with allow_redirects=True
            response = self.session.get(url, stream=True, verify=verify)
            response.raise_for_status()
            
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get content length for progress bar
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            
            # Download with progress bar
            with open(path, 'wb') as f, tqdm(
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
                desc=path.name or "Downloading"
            ) as pbar:
                for data in response.iter_content(block_size):
                    size = f.write(data)
                    pbar.update(size)
            
            if verify:
                self.logger.info(f"Downloaded {path}")
            
            return True
            
        except Exception as e:
            if verify:
                self.logger.error(f"Error downloading {path.name}: {str(e)}")
            if path.exists():
                path.unlink()
            raise
    
    def download_agp_data(self):
        """Download American Gut Project data."""
        self.logger.info("Downloading American Gut Project data...")
        
        # Provide manual download instructions
        self.logger.warning("AGP data requires manual download from: https://qiita.ucsd.edu/study/description/10317")
        self.logger.warning("Please download the following files:")
        self.logger.warning("1. Abundance table (BIOM format) - processed_data/10317_temp/10317_otu_table_filtered_100.biom")
        self.logger.warning("2. Sample information - sample_information/10317_20160906-112348.txt")
        self.logger.warning("Place them in the american_gut directory as abundance.biom and metadata.tsv")
    
    def download_hmp(self):
        """Download HMP data."""
        self.logger.info("Downloading HMP data...")
        
        # HMP1 Data
        self.logger.info("Downloading HMP1 abundance data...")
        hmp1_dir = self.data_dir / "hmp" / "hmp1"
        
        try:
            # Try downloading with SSL verification disabled
            abundance_url = "https://data.hmpdacc.org/project/HMQCP/16s/otu_table_psn_v35.txt.gz"
            metadata_url = "https://data.hmpdacc.org/project/HMQCP/metadata/metadata.csv"
            
            self._download_file(
                abundance_url,
                hmp1_dir / "abundance.txt.gz",
                verify=False
            )
            self._download_file(
                metadata_url,
                hmp1_dir / "metadata.csv",
                verify=False
            )
        except Exception as e:
            self.logger.error(f"Error downloading HMP1 data: {str(e)}")
            self.logger.warning("You can manually download from: https://data.hmpdacc.org/")
        
        # HMP2 Data
        self.logger.info("Downloading HMP2 abundance data...")
        hmp2_dir = self.data_dir / "hmp" / "hmp2"
        
        try:
            abundance_url = "https://data.hmpdacc.org/project/HMCP2/16s/otu_table_v2.txt.gz"
            metadata_url = "https://data.hmpdacc.org/project/HMCP2/metadata/metadata.csv"
            
            self._download_file(
                abundance_url,
                hmp2_dir / "abundance.txt.gz",
                verify=False
            )
            self._download_file(
                metadata_url,
                hmp2_dir / "metadata.csv",
                verify=False
            )
        except Exception as e:
            self.logger.error(f"Error downloading HMP2 data: {str(e)}")
            self.logger.warning("You can manually download from: https://data.hmpdacc.org/")
    
    def download_metahit(self):
        """Download MetaHIT data"""
        self.logger.info("Downloading MetaHIT data...")
        self.logger.warning("MetaHIT data requires registration at https://www.ebi.ac.uk/metagenomics/studies/MGYS00001608")
        self.logger.warning("Please download the following files:")
        self.logger.warning("1. Abundance table (TSV format) - pipelines/4.1/file/abundance_table.tsv.gz")
        self.logger.warning("2. Sample information")
        self.logger.warning("Place them in the metahit directory as abundance.tsv.gz and metadata.tsv")
    
    def download_gmrepo(self):
        """Download GMrepo data"""
        self.logger.info("Downloading GMrepo data...")
        self.logger.warning("GMrepo data requires registration at https://gmrepo.humangut.info/")
        self.logger.warning("Please download the following files:")
        self.logger.warning("1. Healthy samples abundance data")
        self.logger.warning("2. Disease samples abundance data")
        self.logger.warning("Place them in the gmrepo directory as healthy_abundance.tsv.gz and disease_abundance.tsv.gz")
    
    def download_diabimmune(self):
        """Download DIABIMMUNE data"""
        self.logger.info("Downloading DIABIMMUNE data...")
        self.logger.warning("DIABIMMUNE data requires registration at https://diabimmune.broadinstitute.org/")
        self.logger.warning("Please download the data manually and place it in the diabimmune directory")
    
    def download_ibd(self):
        """Download IBD data"""
        self.logger.info("Downloading IBD data...")
        self.logger.warning("IBD data requires registration at https://ibdmdb.org/")
        self.logger.warning("Please download the data manually and place it in the ibd directory")
    
    def download_qiita_studies(self):
        """Download data from selected QIITA studies."""
        self.logger.info("Downloading QIITA studies...")
        
        # Obesity study (10317)
        self.logger.warning("OBESITY data requires manual download from: https://qiita.ucsd.edu/study/description/10317")
        self.logger.warning("Please download the following files for study 10317:")
        self.logger.warning("1. Abundance table (BIOM format) - processed_data/10317_temp/10317_otu_table_filtered_100.biom")
        self.logger.warning("2. Sample information - sample_information/10317_20160906-112348.txt")
        self.logger.warning("Place them in the qiita/obesity directory as abundance.biom and metadata.tsv")
        
        # IBD study (1939)
        self.logger.warning("\nIBD data requires manual download from: https://qiita.ucsd.edu/study/description/1939")
        self.logger.warning("Please download the following files for study 1939:")
        self.logger.warning("1. Abundance table (BIOM format) - processed_data/1939_temp/1939_otu_table_filtered_100.biom")
        self.logger.warning("2. Sample information - sample_information/1939_20160906-112348.txt")
        self.logger.warning("Place them in the qiita/ibd directory as abundance.biom and metadata.tsv")
    
    def download_all(self):
        """Download all datasets"""
        self.download_agp_data()
        self.download_hmp()
        self.download_metahit()
        self.download_gmrepo()
        self.download_diabimmune()
        self.download_ibd()
        self.download_qiita_studies()

def main():
    """Main function to download all datasets"""
    downloader = DataDownloader()
    downloader.download_all()

if __name__ == '__main__':
    main() 