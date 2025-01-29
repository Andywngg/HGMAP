import os
import requests
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm

def download_file(url: str, output_path: Path, description: str = None):
    """Download a file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        desc = description or output_path.name
        with open(output_path, 'wb') as f, tqdm(
            desc=desc,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
            colour='green'
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
        print(f"✓ Downloaded {desc}")
    except Exception as e:
        print(f"✗ Error downloading {desc}: {e}")
        raise

def main():
    # Create data directories
    data_dir = Path("data")
    for subdir in ["american_gut", "hmp", "metahit", "gmrepo", "diabimmune", "ibd", "qiita"]:
        (data_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Download American Gut Project data
    print("\nDownloading American Gut Project data...")
    agp_urls = {
        "abundance": "https://www.ebi.ac.uk/metagenomics/api/v1/studies/MGYS00001935/pipelines/4.1/file/abundance_table.tsv.gz",
        "metadata": "https://www.ebi.ac.uk/metagenomics/api/v1/studies/MGYS00001935/samples"
    }
    
    try:
        for name, url in agp_urls.items():
            output_path = data_dir / "american_gut" / f"{name}.{'gz' if name == 'abundance' else 'json'}"
            download_file(url, output_path, f"AGP {name}")
    except Exception as e:
        print(f"Error with AGP data. You can manually download from: https://www.ebi.ac.uk/metagenomics/studies/MGYS00001935")
    
    # Download HMP data
    print("\nDownloading Human Microbiome Project data...")
    hmp_versions = {
        "hmp1": {
            "abundance": "https://downloads.hmpdacc.org/dacc/HMQCP/otu_table_psn_v35.txt.gz",
            "metadata": "https://downloads.hmpdacc.org/dacc/HMQCP/metadata.csv"
        },
        "hmp2": {
            "abundance": "https://downloads.hmpdacc.org/dacc/HMCP2/otu_table_v2.txt.gz",
            "metadata": "https://downloads.hmpdacc.org/dacc/HMCP2/metadata.csv"
        }
    }
    
    try:
        for version, urls in hmp_versions.items():
            version_dir = data_dir / "hmp" / version
            version_dir.mkdir(exist_ok=True)
            
            for name, url in urls.items():
                output_path = version_dir / f"{name}.{'gz' if name == 'abundance' else 'csv'}"
                download_file(url, output_path, f"HMP {version} {name}")
    except Exception as e:
        print(f"Error with HMP data. You can manually download from: https://portal.hmpdacc.org/")
    
    # Download MetaHIT data
    print("\nDownloading MetaHIT data...")
    metahit_urls = {
        "abundance": "https://www.ebi.ac.uk/metagenomics/api/v1/studies/MGYS00001608/pipelines/4.1/file/abundance_table.tsv.gz",
        "metadata": "https://www.ebi.ac.uk/metagenomics/api/v1/studies/MGYS00001608/samples"
    }
    
    try:
        for name, url in metahit_urls.items():
            output_path = data_dir / "metahit" / f"{name}.{'gz' if name == 'abundance' else 'json'}"
            download_file(url, output_path, f"MetaHIT {name}")
    except Exception as e:
        print(f"Error with MetaHIT data. You can manually download from: https://www.ebi.ac.uk/metagenomics/studies/MGYS00001608")
    
    # Download GMrepo data
    print("\nDownloading GMrepo data...")
    gmrepo_datasets = {
        "healthy": "https://gmrepo.humangut.info/downloads/healthy.tsv.gz",
        "disease": "https://gmrepo.humangut.info/downloads/disease.tsv.gz"
    }
    
    try:
        for dataset_type, url in gmrepo_datasets.items():
            output_dir = data_dir / "gmrepo" / dataset_type
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{dataset_type}_abundance.tsv.gz"
            download_file(url, output_path, f"GMrepo {dataset_type}")
    except Exception as e:
        print(f"Error with GMrepo data. You can manually download from: https://gmrepo.humangut.info/downloads")
        
    # Download DIABIMMUNE data
    print("\nDownloading DIABIMMUNE data...")
    diabimmune_urls = {
        "t1d": "https://diabimmune.broadinstitute.org/diabimmune/t1d/16s/abundance_table.tsv.gz",
        "metadata": "https://diabimmune.broadinstitute.org/diabimmune/t1d/metadata.tsv"
    }
    
    try:
        for name, url in diabimmune_urls.items():
            output_path = data_dir / "diabimmune" / f"{name}.{'gz' if name == 't1d' else 'tsv'}"
            download_file(url, output_path, f"DIABIMMUNE {name}")
    except Exception as e:
        print(f"Error with DIABIMMUNE data. You can manually download from: https://diabimmune.broadinstitute.org/")
        
    # Download IBD Multi-omics data
    print("\nDownloading IBD Multi-omics data...")
    ibd_urls = {
        "abundance": "https://ibdmdb.org/tunnel/public/HMP2/16S/1750.taxonomy.tsv.gz",
        "metadata": "https://ibdmdb.org/tunnel/public/HMP2/metadata.tsv"
    }
    
    try:
        for name, url in ibd_urls.items():
            output_path = data_dir / "ibd" / f"{name}.{'gz' if name == 'abundance' else 'tsv'}"
            download_file(url, output_path, f"IBD {name}")
    except Exception as e:
        print(f"Error with IBD data. You can manually download from: https://ibdmdb.org/")
        
    # Download selected QIITA studies
    print("\nDownloading QIITA studies...")
    qiita_studies = {
        "obesity": {
            "abundance": "https://qiita.ucsd.edu/public_download/?data=abundance_table&study_id=10317",
            "metadata": "https://qiita.ucsd.edu/public_download/?data=sample_information&study_id=10317"
        },
        "ibd": {
            "abundance": "https://qiita.ucsd.edu/public_download/?data=abundance_table&study_id=1939",
            "metadata": "https://qiita.ucsd.edu/public_download/?data=sample_information&study_id=1939"
        }
    }
    
    try:
        for study, urls in qiita_studies.items():
            study_dir = data_dir / "qiita" / study
            study_dir.mkdir(exist_ok=True)
            
            for name, url in urls.items():
                output_path = study_dir / f"{name}.tsv"
                download_file(url, output_path, f"QIITA {study} {name}")
    except Exception as e:
        print(f"Error with QIITA data. You can manually download from: https://qiita.ucsd.edu/")
    
    print("\nDownload Summary:")
    print("1. Check the data/ directory for downloaded files")
    print("2. If any downloads failed, you can use these direct links:")
    print("   - AGP: https://www.ebi.ac.uk/metagenomics/studies/MGYS00001935")
    print("   - HMP: https://portal.hmpdacc.org/")
    print("   - MetaHIT: https://www.ebi.ac.uk/metagenomics/studies/MGYS00001608")
    print("   - GMrepo: https://gmrepo.humangut.info/downloads")
    print("   - DIABIMMUNE: https://diabimmune.broadinstitute.org/")
    print("   - IBD: https://ibdmdb.org/")
    print("   - QIITA: https://qiita.ucsd.edu/")

if __name__ == "__main__":
    main() 