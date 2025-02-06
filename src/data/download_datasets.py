import os
import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import h5py
import gc
import scipy.sparse
import csv
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataDownloader:
    def __init__(self):
        # Focus on studies that have shown successful metadata retrieval
        self.reliable_studies = {
            'MGYS00002673': {  # Human Microbiome Project
                'description': 'HMP study',
                'expected_samples': 10,
                'body_site': 'multiple',
                'pipeline_version': '4.1',
                'taxonomy_file': 'OTUs, counts and taxonomic assignments for SSU rRNA'  # Updated to match exact file name
            }
        }
        
        # Configure session with more lenient timeouts and more retries
        self.session = requests.Session()
        retries = Retry(
            total=5,  # Increased from 3 to 5
            backoff_factor=1,  # Increased from 0.5 to 1
            status_forcelist=[429, 500, 502, 503, 504],  # Added 429 for rate limiting
            allowed_methods=["HEAD", "GET", "OPTIONS"]  # Explicitly specify allowed methods
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)
        
        # Increased timeouts (connect timeout, read timeout)
        self.session.timeout = (30, 300)  # 30 seconds connect, 5 minutes read
        
        self.base_url = "https://www.ebi.ac.uk/metagenomics/api/v1"
    
    def get_study_analyses(self, study_id: str) -> List[Dict]:
        """Get all analyses for a study."""
        try:
            url = f"{self.base_url}/studies/{study_id}/analyses"
            analyses = []
            
            while url:
                try:
                    response = self.session.get(url, timeout=(30, 300))
                    response.raise_for_status()
                    data = response.json()
                    
                    if 'data' in data:
                        # Log analysis information for debugging
                        for analysis in data['data']:
                            analysis_id = analysis.get('id', '')
                            pipeline = analysis.get('attributes', {}).get('pipeline-version', '')
                            logging.info(f"Found analysis {analysis_id} with pipeline {pipeline}")
                        
                        analyses.extend(data['data'])
                        next_url = data.get('links', {}).get('next')
                        url = next_url if isinstance(next_url, str) else None
                    else:
                        break
                except requests.exceptions.Timeout:
                    logging.warning(f"Timeout while fetching analyses for {study_id}, retrying...")
                    continue
                except requests.exceptions.RequestException as e:
                    logging.error(f"Request error while fetching analyses for {study_id}: {str(e)}")
                    break
            
            return analyses
        except Exception as e:
            logging.error(f"Error getting analyses for study {study_id}: {str(e)}")
            return []
    
    def get_analysis_files(self, analysis_id: str) -> List[Dict]:
        """Get files available for an analysis."""
        try:
            url = f"{self.base_url}/analyses/{analysis_id}/downloads"
            try:
                response = self.session.get(url, timeout=(30, 300))
                response.raise_for_status()
                data = response.json()
                
                # Log available files for debugging
                if 'data' in data:
                    for file_info in data.get('data', []):
                        attrs = file_info.get('attributes', {})
                        if isinstance(attrs, dict):
                            label = attrs.get('label', '')
                            desc = attrs.get('description', '')
                            logging.info(f"Available file:  - {{'label': '{label}', 'description': '{desc}'}}")
                
                return data.get('data', [])
            except requests.exceptions.Timeout:
                logging.warning(f"Timeout while fetching files for analysis {analysis_id}")
                return []
            except requests.exceptions.RequestException as e:
                logging.error(f"Request error while fetching files for analysis {analysis_id}: {str(e)}")
                return []
        except Exception as e:
            logging.error(f"Error getting files for analysis {analysis_id}: {str(e)}")
            return []
    
    def download_file(self, url: str, output_path: Path) -> bool:
        """Download a file with improved timeout handling."""
        try:
            # Stream the download with larger chunk size
            response = self.session.get(url, stream=True, timeout=(30, 300))
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024  # 1MB chunks
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
            
            # Verify file size if content-length was provided
            if total_size > 0:
                actual_size = output_path.stat().st_size
                if actual_size != total_size:
                    logging.error(f"Downloaded file size mismatch. Expected: {total_size}, Got: {actual_size}")
                    return False
            
            return True
        except requests.exceptions.Timeout:
            logging.error(f"Timeout downloading {url}")
            return False
        except Exception as e:
            logging.error(f"Error downloading {url}: {str(e)}")
            return False
    
    def get_sample_metadata(self, study_id: str) -> Optional[pd.DataFrame]:
        """Get sample metadata for a study."""
        try:
            url = f"{self.base_url}/studies/{study_id}/samples"
            samples = []
            
            while url:
                response = self.session.get(url)
                response.raise_for_status()
                data = response.json()
                
                if 'data' in data:
                    for sample in data['data']:
                        attrs = sample.get('attributes', {})
                        sample_dict = {
                            'sample_id': sample.get('id', ''),
                            'biosample': attrs.get('biosample', ''),
                            'health_status': 'Unknown',
                            'body_site': attrs.get('environment-feature', ''),
                            'collection_date': attrs.get('collection-date', ''),
                            'investigation_type': attrs.get('investigation-type', ''),
                            'host_scientific_name': attrs.get('host-scientific-name', '')
                        }
                        
                        # Extract metadata fields
                        metadata_list = attrs.get('sample-metadata', [])
                        if isinstance(metadata_list, list):
                            for metadata in metadata_list:
                                if isinstance(metadata, dict):
                                    key = str(metadata.get('key', '')).lower()
                                    value = str(metadata.get('value', ''))
                                    if any(term in key for term in ['health', 'disease', 'condition']):
                                        sample_dict['health_status'] = value
                                    elif any(term in key for term in ['body_site', 'site', 'location']):
                                        sample_dict['body_site'] = value
                        
                        samples.append(sample_dict)
                    
                    next_url = data.get('links', {}).get('next')
                    url = next_url if isinstance(next_url, str) else None
                else:
                    break
            
            if samples:
                df = pd.DataFrame(samples)
                # Add quality control metrics
                df['has_health_status'] = df['health_status'] != 'Unknown'
                df['has_body_site'] = df['body_site'].notna()
                return df
            return None
            
        except Exception as e:
            logging.error(f"Error getting metadata for study {study_id}: {str(e)}")
            return None
    
    def find_taxonomy_file(self, analysis_files: List[Dict], target_file: str) -> Optional[str]:
        """Find the taxonomy file URL from analysis files."""
        logging.info(f"Searching for taxonomy file matching: {target_file}")
        
        for file_info in analysis_files:
            try:
                attrs = file_info.get('attributes', {})
                if not isinstance(attrs, dict):
                    continue
                    
                # Get the description which contains the actual file metadata
                desc_str = str(attrs.get('description', ''))
                if not desc_str:
                    continue
                    
                # Clean up the description string if it's a nested structure
                if desc_str.startswith("{'") and desc_str.endswith("'}"):
                    try:
                        desc_dict = eval(desc_str)  # Safe in this context since we control the input
                        if not isinstance(desc_dict, dict):
                            continue
                            
                        label = str(desc_dict.get('label', '')).strip()
                        desc = str(desc_dict.get('description', '')).strip()
                        
                        # Check for exact match first
                        if label == target_file:
                            url = file_info.get('links', {}).get('self')
                            if url:
                                logging.info(f"Found exact match: {label}")
                                return url
                        
                        # Check for OTU files with taxonomic assignments for SSU rRNA
                        if ('OTUs' in label and 'taxonomic assignments' in label and 
                            'SSU rRNA' in label):
                            url = file_info.get('links', {}).get('self')
                            if url:
                                logging.info(f"Found matching OTU file: {label}")
                                return url
                                
                    except Exception as e:
                        logging.warning(f"Error parsing description dict: {str(e)}")
                        continue
                        
            except Exception as e:
                logging.warning(f"Error processing file info: {str(e)}")
                continue
        
        logging.warning("No matching taxonomy file found")
        return None
    
    def convert_hdf5_to_tsv(self, file_path: Path) -> bool:
        """Convert HDF5 file to TSV format."""
        try:
            with h5py.File(file_path, 'r') as f:
                # Extract data from HDF5 file
                data = f['observation']['matrix']['data'][:]
                indices = f['observation']['matrix']['indices'][:]
                indptr = f['observation']['matrix']['indptr'][:]
                observation_ids = f['observation']['ids'][:]
                sample_ids = f['sample']['ids'][:]
                taxonomy_data = f['observation']['metadata']['taxonomy'][:]

                # Log shapes for debugging
                logging.info(f"Found dataset: observation/matrix/data with shape {len(data)}")
                logging.info(f"Found dataset: observation/matrix/indices with shape {len(indices)}")
                logging.info(f"Found dataset: observation/matrix/indptr with shape {len(indptr)}")
                logging.info(f"Found dataset: observation/metadata/taxonomy with shape {len(taxonomy_data)}")
                logging.info(f"Found dataset: sample/ids with shape {len(sample_ids)}")

                # Determine matrix format and dimensions
                n_observations = len(observation_ids)
                n_samples = len(sample_ids)
                
                logging.info(f"Matrix dimensions: {n_observations} observations x {n_samples} samples")
                
                # Try to determine the format based on indptr size
                if len(indptr) == n_samples + 1:
                    # CSC format
                    logging.info("Detected CSC format")
                    matrix = scipy.sparse.csc_matrix(
                        (data, indices, indptr),
                        shape=(n_observations, n_samples)
                    )
                elif len(indptr) == n_observations + 1:
                    # CSR format
                    logging.info("Detected CSR format")
                    matrix = scipy.sparse.csr_matrix(
                        (data, indices, indptr),
                        shape=(n_observations, n_samples)
                    ).tocsc()
                else:
                    # Try to infer format from data
                    logging.info("Attempting to infer matrix format from data")
                    try:
                        # Try CSR first
                        matrix = scipy.sparse.csr_matrix(
                            (data, indices, indptr),
                            shape=(n_observations, n_samples)
                        ).tocsc()
                    except:
                        try:
                            # Try CSC if CSR fails
                            matrix = scipy.sparse.csc_matrix(
                                (data, indices, indptr),
                                shape=(n_samples, n_observations)
                            ).transpose()
                        except Exception as e:
                            raise ValueError(f"Could not determine matrix format: {str(e)}")

                # Convert to dense format
                dense_matrix = matrix.toarray()
                logging.info(f"Converted to dense matrix with shape {dense_matrix.shape}")

                # Create output path
                output_path = file_path.parent / f"{file_path.stem}_converted.tsv"

                # Write to TSV
                with open(output_path, 'w', newline='') as tsv_file:
                    writer = csv.writer(tsv_file, delimiter='\t')
                    
                    # Write header (sample IDs)
                    header = ['OTU_ID'] + [str(sid) for sid in sample_ids]
                    writer.writerow(header)
                    
                    # Write data rows
                    for i, (obs_id, row) in enumerate(zip(observation_ids, dense_matrix)):
                        # Convert taxonomy array to string if needed
                        taxonomy_str = ';'.join(str(t) for t in taxonomy_data[i]) if i < len(taxonomy_data) else ''
                        writer.writerow([f"{obs_id}|{taxonomy_str}"] + list(row))

                logging.info(f"Successfully wrote TSV file to {output_path}")

                # Clean up original file with retries
                max_retries = 3
                retry_delay = 2  # seconds
                
                for attempt in range(max_retries):
                    try:
                        time.sleep(retry_delay)
                        os.remove(file_path)
                        logging.info(f"Successfully removed original file: {file_path}")
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logging.warning(f"Attempt {attempt + 1} failed to remove file {file_path}: {str(e)}")
                            retry_delay *= 2  # Exponential backoff
                        else:
                            logging.warning(f"Failed to remove original file {file_path} after {max_retries} attempts: {str(e)}")

                return True

        except Exception as e:
            logging.error(f"Failed to convert HDF5 to TSV: {str(e)}")
            return False
    
    def verify_file_format(self, file_path: Path) -> bool:
        """Verify the format of a downloaded file and convert if necessary."""
        try:
            # Read the first few bytes to determine file type
            with open(file_path, 'rb') as f:
                magic_bytes = f.read(8)
                
            if magic_bytes.startswith(b'\x89HDF'):  # HDF5 magic number
                logging.info(f"Detected HDF5 format for {file_path}")
                # Convert to TSV with a unique name
                tsv_path = file_path.with_name(f"{file_path.stem}_converted.tsv")
                if self.convert_hdf5_to_tsv(file_path):
                    return True
                return False
            elif magic_bytes.startswith(b'\x1f\x8b'):  # gzip magic number
                import gzip
                with gzip.open(file_path, 'rt') as f:
                    header = f.readline()
                    if '\t' in header:  # Check if it's tab-delimited
                        logging.info(f"Verified gzipped TSV format for {file_path}")
                        return True
            else:
                # Try different encodings for text files
                for encoding in ['utf-8', 'latin1', 'iso-8859-1']:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            header = f.readline()
                            if '\t' in header:  # Check if it's tab-delimited
                                logging.info(f"Verified TSV format with {encoding} encoding for {file_path}")
                                return True
                    except UnicodeDecodeError:
                        continue
            
            logging.error(f"Invalid file format for {file_path}")
            return False
            
        except Exception as e:
            logging.error(f"Error verifying file format: {str(e)}")
            return False
    
    def download_study_data(self, study_id: str) -> bool:
        """Download data for a specific study."""
        if study_id not in self.reliable_studies:
            logging.error(f"Study {study_id} not in reliable studies list")
            return False
        
        study_info = self.reliable_studies[study_id]
        study_dir = Path(f"data/mgnify/{study_id}")
        study_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Get metadata first to check data quality
            logging.info(f"Getting metadata for {study_id}")
            metadata_df = self.get_sample_metadata(study_id)
            
            if metadata_df is None:
                logging.error(f"Failed to get metadata for {study_id}")
                return False
            
            if len(metadata_df) < study_info['expected_samples']:
                logging.error(f"Insufficient samples for {study_id}. Expected {study_info['expected_samples']}, got {len(metadata_df)}")
                return False
            
            # Save metadata
            metadata_file = study_dir / "metadata.tsv"
            metadata_df.to_csv(metadata_file, sep='\t', index=False)
            logging.info(f"Saved metadata for {study_id}")
            
            # Get analyses
            analyses = self.get_study_analyses(study_id)
            if not analyses:
                logging.error(f"No analyses found for {study_id}")
                return False
            
            target_file = study_info['taxonomy_file']
            pipeline_version = study_info['pipeline_version']
            
            # Filter analyses by pipeline version first
            matching_analyses = [
                analysis for analysis in analyses 
                if analysis.get('attributes', {}).get('pipeline-version', '') == pipeline_version
            ]
            
            if not matching_analyses:
                logging.error(f"No analyses found with pipeline version {pipeline_version}")
                return False
            
            logging.info(f"Found {len(matching_analyses)} analyses with matching pipeline version {pipeline_version}")
            
            # Try each analysis until we find a valid taxonomy file
            success = False
            converted_files = []
            for analysis in matching_analyses:
                analysis_id = analysis.get('id', '')
                logging.info(f"Processing analysis {analysis_id}")
                
                files = self.get_analysis_files(analysis_id)
                taxonomy_url = self.find_taxonomy_file(files, target_file)
                
                if taxonomy_url:
                    abundance_file = study_dir / f"abundance_{analysis_id}.tsv"
                    if self.download_file(taxonomy_url, abundance_file):
                        logging.info(f"Successfully downloaded abundance data for {study_id} from analysis {analysis_id}")
                        
                        # Verify and convert the file if necessary
                        if self.verify_file_format(abundance_file):
                            logging.info(f"Successfully processed abundance file for {analysis_id}")
                            converted_files.append(abundance_file)
                            success = True
                            continue  # Process all analyses instead of returning early
            
            # Clean up original HDF5 files after all processing is complete
            if success:
                import time
                time.sleep(1)  # Give OS time to release file handles
                
                # Remove original HDF5 files that were successfully converted
                for orig_file in converted_files:
                    try:
                        if orig_file.exists():
                            os.remove(orig_file)
                            logging.info(f"Removed original HDF5 file: {orig_file}")
                    except OSError as e:
                        logging.warning(f"Could not remove original file {orig_file}: {e}")
                
                # Rename converted files to their final names
                for orig_file in converted_files:
                    converted_file = orig_file.with_name(f"{orig_file.stem}_converted.tsv")
                    final_file = orig_file
                    try:
                        if converted_file.exists():
                            os.rename(converted_file, final_file)
                            logging.info(f"Renamed {converted_file} to {final_file}")
                    except OSError as e:
                        logging.warning(f"Could not rename file {converted_file}: {e}")
            
            return success
            
        except Exception as e:
            logging.error(f"Error processing study {study_id}: {str(e)}")
            return False

def main():
    downloader = DataDownloader()
    
    # Process each reliable study
    for study_id, info in downloader.reliable_studies.items():
        logging.info(f"Processing study {study_id} ({info['description']})")
        if downloader.download_study_data(study_id):
            logging.info(f"Successfully processed {study_id}")
        else:
            logging.error(f"Failed to process {study_id}")

if __name__ == "__main__":
    main() 