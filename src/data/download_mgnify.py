#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Download data from MGnify."""

import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import gzip
import json
from tqdm import tqdm
import shutil
import tempfile
import h5py
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MGnifyDownloader:
    """Download and process MGnify gut microbiome data"""
    
    def __init__(self, studies=None):
        """Initialize the downloader with a list of studies."""
        self.base_url = "https://www.ebi.ac.uk/metagenomics/api/v1"
        self.studies = studies or [
            "MGYS00005745",  # Human gut metagenome (healthy)
            "MGYS00001248",  # Human gut microbiome (IBD)
            "MGYS00005102",  # Human gut microbiome (atherosclerosis)
            "MGYS00004620",  # Human gut microbiome (healthy)
            "MGYS00004018",  # Human gut microbiome (IBD)
            "MGYS00003923",  # Human gut microbiome (colorectal cancer)
            "MGYS00003843"   # Human gut microbiome (type 2 diabetes)
        ]
        
        self.data_dir = Path("data")
        self.mgnify_dir = self.data_dir / "mgnify"
        self.mgnify_dir.mkdir(parents=True, exist_ok=True)
    
    def get_latest_analysis(self, study_id):
        """Get the latest analysis ID for a study."""
        try:
            analyses_url = f"{self.base_url}/studies/{study_id}/analyses"
            response = requests.get(analyses_url)
            response.raise_for_status()
            analyses = response.json()['data']
            
            if not analyses:
                return None
            
            return analyses[0]['id']  # Assuming the first one is the latest
        except Exception as e:
            logging.error(f"Error getting latest analysis for study {study_id}: {str(e)}")
            return None

    def get_study_metadata(self, study_id):
        """Get metadata for a study."""
        try:
            study_url = f"{self.base_url}/studies/{study_id}"
            response = requests.get(study_url)
            response.raise_for_status()
            study_data = response.json()['data']
            
            metadata = {
                'study_id': study_id,
                'study_name': study_data['attributes'].get('study_name', ''),
                'study_abstract': study_data['attributes'].get('study_abstract', ''),
                'samples': study_data['attributes'].get('samples_count', 0)
            }
            
            return pd.DataFrame([metadata])
        except Exception as e:
            logging.error(f"Error getting metadata for study {study_id}: {str(e)}")
            return None

    def get_analysis_files(self, analysis_id):
        """Get all files for an analysis."""
        try:
            # Get the analysis details
            analysis_url = f"{self.base_url}/analyses/{analysis_id}"
            logging.info(f"Fetching analysis from: {analysis_url}")
            response = requests.get(analysis_url)
                response.raise_for_status()
            analysis_data = response.json()['data']
            
            # Log analysis data structure
            logging.info(f"Analysis data keys: {list(analysis_data.keys())}")
            if 'relationships' in analysis_data:
                logging.info(f"Relationship keys: {list(analysis_data['relationships'].keys())}")
            
            # Get the file relationships
            if 'relationships' not in analysis_data:
                logging.warning(f"No relationships found in analysis {analysis_id}")
                return None
                
            # Try different relationship paths
            file_data = []
            relationship_paths = [
                ['downloads', 'data'],
                ['files', 'data'],
                ['outputFiles', 'data'],
                ['analysisResults', 'data']
            ]
            
            for path in relationship_paths:
                current = analysis_data['relationships']
                valid_path = True
                for key in path:
                    if key in current:
                        current = current[key]
                    else:
                        valid_path = False
                        break
                if valid_path and isinstance(current, list):
                    logging.info(f"Found files using path: {path}")
                    file_data.extend(current)
            
            if not file_data:
                logging.warning(f"No files found in analysis {analysis_id}")
                return None
            
            # Get detailed file information
            detailed_files = []
            for file_ref in file_data:
                file_id = file_ref['id']
                logging.info(f"Getting details for file: {file_id}")
                
                # Try different file endpoints
                endpoints = [
                    f"{self.base_url}/analyses/{analysis_id}/downloads/{file_id}",
                    f"{self.base_url}/analyses/{analysis_id}/files/{file_id}",
                    f"{self.base_url}/downloads/{file_id}",
                    f"{self.base_url}/files/{file_id}",
                    f"{self.base_url}/analyses/{analysis_id}/analysisResults/{file_id}"
                ]
                
                for endpoint in endpoints:
                    try:
                        logging.info(f"Trying endpoint: {endpoint}")
                        file_response = requests.get(endpoint)
                        file_response.raise_for_status()
                        file_data = file_response.json()['data']
                        logging.info(f"Successfully got file data from: {endpoint}")
                        detailed_files.append(file_data)
                        break
                    except Exception as e:
                        logging.debug(f"Failed to get file data from {endpoint}: {str(e)}")
                        continue
            
            return detailed_files
        except Exception as e:
            logging.error(f"Error getting files for analysis {analysis_id}: {str(e)}")
            return None

    def get_download_url(self, analysis_id, file_id):
        """Get the download URL for a file."""
        try:
            # Try different endpoints
            endpoints = [
                f"{self.base_url}/analyses/{analysis_id}/downloads/{file_id}",
                f"{self.base_url}/analyses/{analysis_id}/files/{file_id}",
                f"{self.base_url}/downloads/{file_id}",
                f"{self.base_url}/files/{file_id}",
                f"{self.base_url}/analyses/{analysis_id}/analysisResults/{file_id}"
            ]
            
            for endpoint in endpoints:
                try:
                    logging.info(f"Trying to get download URL from: {endpoint}")
                    response = requests.get(endpoint)
                    response.raise_for_status()
                    file_data = response.json()['data']
                    
                    # Try different attribute paths for the download URL
                    url_paths = [
                        ['attributes', 'links', 'self'],
                        ['attributes', 'url'],
                        ['attributes', 'download-url'],
                        ['links', 'self'],
                        ['links', 'download'],
                        ['attributes', 'downloadUrl']
                    ]
                    
                    for path in url_paths:
                        url = file_data
                        for key in path:
                            if isinstance(url, dict) and key in url:
                                url = url[key]
                            else:
                                url = None
                                break
                        
                        if url and isinstance(url, str):
                            logging.info(f"Found download URL using path {path}: {url}")
                            return url
                except Exception as e:
                    logging.debug(f"Failed to get download URL from {endpoint}: {str(e)}")
                    continue
            
            return None
        except Exception as e:
            logging.error(f"Error getting download URL for file {file_id}: {str(e)}")
            return None

    def _is_otu_file(self, file_obj):
        """Check if a file object is an OTU file."""
        # Check file type in attributes
        attributes = file_obj.get('attributes', {})
        file_type = attributes.get('file-type', {})
        if isinstance(file_type, dict):
            file_type = file_type.get('id', '')
        elif isinstance(file_type, str):
            file_type = file_type
            
        # Check description and label
        description = attributes.get('description', {})
        if isinstance(description, dict):
            label = description.get('label', '').lower()
            desc = description.get('description', '').lower()
        else:
            label = str(description).lower()
            desc = label
            
        # Look for OTU-related terms
        otu_terms = ['otu', 'taxonomic assignments', 'taxonomy', 'abundance']
        return (any(term in file_type.lower() for term in otu_terms) or
                any(term in label for term in otu_terms) or
                any(term in desc for term in otu_terms))
    
    def parse_abundance_file(self, file_path):
        """Parse abundance file and convert to DataFrame"""
        try:
            # Read the file
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Parse header to find column names
            header_line = None
            for i, line in enumerate(lines):
                if not line.startswith('#'):
                    header_line = i
                    break
            
            if header_line is None:
                raise ValueError("Could not find header line")
            
            # Get column names
            columns = lines[header_line].strip().split('\t')
            
            # Parse data
            data = []
            for line in lines[header_line + 1:]:
                if line.strip():  # Skip empty lines
                    values = line.strip().split('\t')
                    if len(values) == len(columns):
                        data.append(values)
            
            # Create DataFrame
            df = pd.DataFrame(data, columns=columns)
            
            # Find OTU ID column
            otu_col = None
            for col in df.columns:
                if any(term in col.lower() for term in ['otu', 'id', '#otu']):
                    otu_col = col
                    break
            
            if otu_col is None:
                # If no OTU column found, use first column
                otu_col = df.columns[0]
            
            # Set index to OTU ID
            df = df.set_index(otu_col)
            
            # Find abundance columns (numeric columns)
            abundance_cols = []
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                    abundance_cols.append(col)
                except:
                    continue
                
            if not abundance_cols:
                raise ValueError("No abundance columns found")
            
            # Keep only abundance columns
            df = df[abundance_cols]
            
            # Convert to float
            df = df.astype(float)
            
            return df
            
        except Exception as e:
            logger.error(f"Error parsing abundance file: {e}")
            raise
    
    def get_sample_taxonomy(self, analysis_id):
        """Get taxonomy data through sample relationships."""
        try:
            # First get the sample ID from the analysis
            analysis_url = f"{self.base_url}/analyses/{analysis_id}"
            logging.info(f"Getting sample ID from analysis: {analysis_url}")
            response = requests.get(analysis_url)
            response.raise_for_status()
            analysis_data = response.json()['data']
            
            sample_id = analysis_data.get('relationships', {}).get('sample', {}).get('data', {}).get('id')
            if not sample_id:
                logging.warning(f"No sample ID found for analysis {analysis_id}")
                return None
            
            # Get sample data
            sample_url = f"{self.base_url}/samples/{sample_id}"
            logging.info(f"Getting sample data from: {sample_url}")
            response = requests.get(sample_url)
            response.raise_for_status()
            sample_data = response.json()['data']
            
            # Try to get taxonomy data from sample relationships
            taxonomy_data = []
            relationship_types = [
                'taxonomy-ssu',
                'taxonomy-lsu',
                'taxonomy',
                'biome'
            ]
            
            for rel_type in relationship_types:
                if rel_type in sample_data.get('relationships', {}):
                    rel_url = f"{self.base_url}/samples/{sample_id}/{rel_type}"
                    logging.info(f"Getting {rel_type} data from: {rel_url}")
                    response = requests.get(rel_url)
                    if response.ok:
                        data = response.json().get('data', [])
                        if isinstance(data, list):
                            taxonomy_data.extend(data)
                        elif isinstance(data, dict):
                            taxonomy_data.append(data)
            
            if taxonomy_data:
                logging.info(f"Found {len(taxonomy_data)} taxonomy entries from sample relationships")
                rows = []
                for entry in taxonomy_data:
                    try:
                        attributes = entry.get('attributes', {})
                        taxonomy = attributes.get('name', '')
                        count = attributes.get('count', 1)  # Default to 1 if count not available
                        lineage = attributes.get('lineage', {})
                        
                        # Add lineage information if available
                        if lineage:
                            taxonomy_parts = []
                            for rank in ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']:
                                if rank in lineage:
                                    taxonomy_parts.append(f"{rank[0]}__" + lineage[rank])
                            if taxonomy_parts:
                                taxonomy = '; '.join(taxonomy_parts)
                        
                        # If no proper taxonomy found, try to extract from other fields
                        if not taxonomy:
                            taxonomy = attributes.get('taxonomy', '')
                        if not taxonomy:
                            taxonomy = attributes.get('scientific_name', '')
                        if not taxonomy:
                            taxonomy = attributes.get('label', '')
                        
                        if taxonomy:  # Only add if we have taxonomy
                            rows.append({
                                'taxonomy': taxonomy,
                                'count': count
                            })
                    except Exception as e:
                        logging.debug(f"Error processing taxonomy entry: {str(e)}")
                        continue
                
                if rows:
                    df = pd.DataFrame(rows)
                    df['study_id'] = analysis_id
                    logging.info(f"Created DataFrame with {len(df)} rows from sample relationships")
                    return df
            
            return None
        except Exception as e:
            logging.error(f"Error getting sample taxonomy for analysis {analysis_id}: {str(e)}")
            return None

    def get_run_taxonomy(self, analysis_id):
        """Get taxonomy data from run-level information."""
        try:
            # First get the run ID from the analysis
            analysis_url = f"{self.base_url}/analyses/{analysis_id}"
            logging.info(f"Getting run ID from analysis: {analysis_url}")
            response = requests.get(analysis_url)
            response.raise_for_status()
            analysis_data = response.json()['data']
            
            run_id = analysis_data.get('relationships', {}).get('run', {}).get('data', {}).get('id')
            if not run_id:
                logging.warning(f"No run ID found for analysis {analysis_id}")
                return None
            
            # Get run data
            run_url = f"{self.base_url}/runs/{run_id}"
            logging.info(f"Getting run data from: {run_url}")
            response = requests.get(run_url)
            response.raise_for_status()
            run_data = response.json()['data']
            
            # Try to get taxonomy data from run relationships
            taxonomy_data = []
            relationship_types = [
                'taxonomy-ssu',
                'taxonomy-lsu',
                'taxonomy',
                'taxonomy-itsunite',
                'taxonomy-itsonedb'
            ]
            
            for rel_type in relationship_types:
                if rel_type in run_data.get('relationships', {}):
                    rel_url = f"{self.base_url}/runs/{run_id}/{rel_type}"
                    logging.info(f"Getting {rel_type} data from: {rel_url}")
                    response = requests.get(rel_url)
                    if response.ok:
                        data = response.json().get('data', [])
                        if isinstance(data, list):
                            taxonomy_data.extend(data)
                        elif isinstance(data, dict):
                            taxonomy_data.append(data)
            
            if taxonomy_data:
                logging.info(f"Found {len(taxonomy_data)} taxonomy entries from run relationships")
                rows = []
                for entry in taxonomy_data:
                    try:
                        attributes = entry.get('attributes', {})
                        taxonomy = attributes.get('name', '')
                        count = attributes.get('count', 1)  # Default to 1 if count not available
                        lineage = attributes.get('lineage', {})
                        
                        # Add lineage information if available
                        if lineage:
                            taxonomy_parts = []
                            for rank in ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']:
                                if rank in lineage:
                                    taxonomy_parts.append(f"{rank[0]}__" + lineage[rank])
                            if taxonomy_parts:
                                taxonomy = '; '.join(taxonomy_parts)
                        
                        # If no proper taxonomy found, try to extract from other fields
                        if not taxonomy:
                            taxonomy = attributes.get('taxonomy', '')
                        if not taxonomy:
                            taxonomy = attributes.get('scientific_name', '')
                        if not taxonomy:
                            taxonomy = attributes.get('label', '')
                        
                        if taxonomy:  # Only add if we have taxonomy
                            rows.append({
                                'taxonomy': taxonomy,
                                'count': count
                            })
                    except Exception as e:
                        logging.debug(f"Error processing taxonomy entry: {str(e)}")
                        continue
                
                if rows:
                    df = pd.DataFrame(rows)
                    df['study_id'] = analysis_id
                    logging.info(f"Created DataFrame with {len(df)} rows from run relationships")
                    return df
            
            return None
        except Exception as e:
            logging.error(f"Error getting run taxonomy for analysis {analysis_id}: {str(e)}")
            return None

    def get_otu_taxonomy(self, analysis_id):
        """Get taxonomy data from OTU files."""
        try:
            # Get the analysis data to find OTU files
            analysis_url = f"{self.base_url}/analyses/{analysis_id}"
            logging.info(f"Getting analysis data from: {analysis_url}")
            response = requests.get(analysis_url)
            response.raise_for_status()
            analysis_data = response.json()['data']
            
            # Look for OTU files in the analysis relationships
            if 'relationships' in analysis_data:
                relationships = analysis_data['relationships']
                
                # Check for OTU downloads
                if 'downloads' in relationships:
                    downloads_url = f"{self.base_url}/analyses/{analysis_id}/downloads"
                    logging.info(f"Getting downloads from: {downloads_url}")
                    response = requests.get(downloads_url)
                    if response.ok:
                        downloads = response.json().get('data', [])
                        logging.info(f"Found {len(downloads)} downloads")
                        
                        # Look for OTU files
                        otu_files = []
                        for d in downloads:
                            if 'attributes' in d:
                                desc = d['attributes'].get('description', {})
                                if isinstance(desc, dict):
                                    label = desc.get('label', '')
                                    description = desc.get('description', '')
                                    if any(x in label.lower() for x in ['otu', 'taxonomic assignments']) or \
                                       any(x in description.lower() for x in ['otu', 'taxonomic assignments']):
                                        # Try to get URL from different locations
                                        url = None
                                        if 'links' in d:
                                            url = d['links'].get('self', '')
                                        if not url and 'relationships' in d:
                                            url = d['relationships'].get('download', {}).get('links', {}).get('related', '')
                                        if url:
                                            logging.info(f"Found OTU file - Label: {label}, Description: {description}, URL: {url}")
                                            otu_files.append({'url': url, 'description': description})
                        
                        for otu_file in otu_files:
                            try:
                                file_url = otu_file['url']
                                logging.info(f"Downloading OTU file from: {file_url}")
                                response = requests.get(file_url)
                                if response.ok:
                                    # Try to parse the file content
                                    content = response.text
                                    rows = []
                                    
                                    # Skip comment lines
                                    lines = [line for line in content.split('\n') if line and not line.startswith('#')]
                                    
                                    # Try to identify header line
                                    header = None
                                    for line in lines[:5]:  # Check first 5 lines
                                        if '\t' in line:
                                            parts = line.strip().split('\t')
                                        else:
                                            parts = line.strip().split()
                                        
                                        # Look for common OTU table headers
                                        if any(x in part.lower() for part in parts for x in ['otu', 'taxonomy', '#otuid']):
                                            header = parts
                                            break
                                    
                                    if header:
                                        # Try to identify taxonomy and count columns
                                        tax_col = None
                                        count_cols = []
                                        for i, col in enumerate(header):
                                            if any(x in col.lower() for x in ['taxonomy', 'classification', 'taxon']):
                                                tax_col = i
                                            elif col.lower() not in ['otu', 'otuid', '#otuid', 'id', 'taxonomy']:
                                                count_cols.append(i)
                                        
                                        if tax_col is not None and count_cols:
                                            # Process data lines
                                            for line in lines[1:]:  # Skip header
                                                try:
                                                    if '\t' in line:
                                                        parts = line.strip().split('\t')
                                                    else:
                                                        parts = line.strip().split()
                                                        
                                                    if len(parts) > max(tax_col, max(count_cols)):
                                                        taxonomy = parts[tax_col]
                                                        # Sum counts across samples
                                                        count = sum(float(parts[i]) for i in count_cols if parts[i].replace('.', '').isdigit())
                                                        
                                                        if taxonomy and count > 0:
                                                            rows.append({
                                                                'taxonomy': taxonomy,
                                                                'count': count
                                                            })
                                                except Exception as e:
                                                    logging.debug(f"Error processing line: {str(e)}")
                                                    continue
                                    
                                    if rows:
                                        df = pd.DataFrame(rows)
                                        df['study_id'] = analysis_id
                                        logging.info(f"Created DataFrame with {len(df)} rows from OTU file")
                                        return df
                            except Exception as e:
                                logging.debug(f"Error processing OTU file: {str(e)}")
                                continue
            
            return None
            
        except Exception as e:
            logging.error(f"Error getting OTU taxonomy for analysis {analysis_id}: {str(e)}")
            return None

    def get_taxonomy_data(self, analysis_id):
        """Get taxonomy data for an analysis."""
        try:
            # First try direct taxonomy endpoints
            df = self._get_direct_taxonomy(analysis_id)
            if df is not None:
                return df
            
            # If that fails, try getting taxonomy through OTU files
            logging.info("Direct taxonomy lookup failed, trying through OTU files")
            df = self.get_otu_taxonomy(analysis_id)
            if df is not None:
                return df
            
            # If that fails, try getting taxonomy through run relationships
            logging.info("OTU taxonomy lookup failed, trying through run relationships")
            df = self.get_run_taxonomy(analysis_id)
            if df is not None:
                return df
            
            # If that fails, try getting taxonomy through sample relationships
            logging.info("Run taxonomy lookup failed, trying through sample relationships")
            return self.get_sample_taxonomy(analysis_id)
            
        except Exception as e:
            logging.error(f"Error getting taxonomy data for analysis {analysis_id}: {str(e)}")
            return None

    def _get_direct_taxonomy(self, analysis_id):
        """Get taxonomy data directly from analysis endpoints."""
        try:
            # Try different taxonomy endpoints
            taxonomy_types = [
                'taxonomy-ssu',
                'taxonomy-lsu',
                'taxonomy',
                'taxonomy-itsunite',
                'taxonomy-itsonedb'
            ]
            
            for tax_type in taxonomy_types:
                try:
                    # Get first page
                    base_url = f"{self.base_url}/analyses/{analysis_id}/{tax_type}"
                    logging.info(f"Trying taxonomy endpoint: {base_url}")
                    
                    all_tax_data = []
                    next_url = base_url
                    page = 1
                    
                    while next_url:
                        logging.info(f"Fetching page {page} from: {next_url}")
                        response = requests.get(next_url)
                        response.raise_for_status()
                        response_data = response.json()
                        
                        # Add data from this page
                        tax_data = response_data.get('data', [])
                        if tax_data:
                            all_tax_data.extend(tax_data)
                            logging.info(f"Found {len(tax_data)} entries on page {page}")
                        
                        # Get next page URL
                        next_url = response_data.get('links', {}).get('next')
                        if next_url:
                            logging.info(f"Found next page: {next_url}")
                            page += 1
                    
                    if all_tax_data:
                        logging.info(f"Found total of {len(all_tax_data)} taxonomy entries using {tax_type}")
                        # Extract taxonomy and abundance data
                        rows = []
                        for entry in all_tax_data:
                            try:
                                attributes = entry.get('attributes', {})
                                taxonomy = attributes.get('name', '')
                                count = attributes.get('count', 0)
                                lineage = attributes.get('lineage', {})
                                
                                # Add lineage information if available
                                if lineage:
                                    taxonomy_parts = []
                                    for rank in ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']:
                                        if rank in lineage:
                                            taxonomy_parts.append(f"{rank[0]}__" + lineage[rank])
                                    if taxonomy_parts:
                                        taxonomy = '; '.join(taxonomy_parts)
                                
                                # If no proper taxonomy found, try to extract from other fields
                                if not taxonomy:
                                    taxonomy = attributes.get('taxonomy', '')
                                if not taxonomy:
                                    taxonomy = attributes.get('scientific_name', '')
                                
                                if taxonomy and count > 0:  # Only add if we have both taxonomy and count
                                    rows.append({
                                        'taxonomy': taxonomy,
                                        'count': count
                                    })
                            except Exception as e:
                                logging.debug(f"Error processing taxonomy entry: {str(e)}")
                                continue
                        
                        if rows:
                            df = pd.DataFrame(rows)
                            df['study_id'] = analysis_id
                            logging.info(f"Created DataFrame with {len(df)} rows")
                            return df
                        else:
                            logging.warning(f"No valid taxonomy entries found in {len(all_tax_data)} entries")
                except Exception as e:
                    logging.debug(f"Failed to get taxonomy data from {tax_type}: {str(e)}")
                    continue
            
            logging.warning(f"No taxonomy data found in any endpoint for {analysis_id}")
            return None
        except Exception as e:
            logging.error(f"Error getting taxonomy data for analysis {analysis_id}: {str(e)}")
            return None

    def download_study_data(self, study_id):
        """Download and process data for a single study."""
        try:
            # Get the latest analysis for this study
            analysis_id = self.get_latest_analysis(study_id)
            if not analysis_id:
                logging.warning(f"No analysis found for study {study_id}")
                return None, None
            
            logging.info(f"Using analysis {analysis_id} for study {study_id}")
            
            # Get metadata
            metadata_df = self.get_study_metadata(study_id)
            
            # Get taxonomy data
            abundance_df = self.get_taxonomy_data(analysis_id)
            if abundance_df is None:
                logging.warning(f"No taxonomy data found for {study_id}")
                return None, metadata_df
            
            return abundance_df, metadata_df
            
        except Exception as e:
            logging.error(f"Error processing study {study_id}: {str(e)}")
            return None, None

    def download_and_process(self):
        """Download and process data for all studies."""
        all_abundance = []
        all_metadata = []
        
        for study_id in self.studies:
            logging.info(f"Downloading data for study {study_id}...")
            try:
                abundance_df, metadata_df = self.download_study_data(study_id)
                
                if abundance_df is not None:
                    all_abundance.append(abundance_df)
                
                if metadata_df is not None:
                    all_metadata.append(metadata_df)
                    
            except Exception as e:
                logging.error(f"Error processing study {study_id}: {str(e)}")
                continue
        
        if not all_metadata and not all_abundance:
            raise ValueError("No data could be downloaded")
        
        # Combine metadata
        if all_metadata:
            metadata_df = pd.concat(all_metadata, ignore_index=True)
        else:
            metadata_df = pd.DataFrame()
            
        # Combine abundance data
        if all_abundance:
            # Combine all processed DataFrames
            abundance_df = pd.concat(all_abundance, ignore_index=True)
            
            # Remove duplicate taxonomies within each study
            abundance_df = abundance_df.groupby(['study_id', 'taxonomy'])['count'].sum().reset_index()
        else:
            abundance_df = pd.DataFrame()
        
        return abundance_df, metadata_df

    def read_biom_file(self, file_path):
        """Read a BIOM file and return a pandas DataFrame."""
        try:
            with h5py.File(file_path, 'r') as f:
                # Check if this is a valid BIOM HDF5 file
                if 'observation' not in f or 'sample' not in f:
                    raise ValueError("Not a valid BIOM HDF5 file")
                
                # Get observation IDs and metadata
                obs_ids = [id.decode('utf-8') for id in f['observation/ids'][()]]
                n_obs = len(obs_ids)
                
                # Get sample IDs
                sample_ids = [id.decode('utf-8') for id in f['sample/ids'][()]]
                n_samples = len(sample_ids)
                
                # Get taxonomy data
                taxonomy = None
                if 'observation/metadata/taxonomy' in f:
                    taxonomy_data = f['observation/metadata/taxonomy'][()]
                    taxonomy = []
                    for tax in taxonomy_data:
                        if isinstance(tax, bytes):
                            tax = tax.decode('utf-8')
                        elif isinstance(tax, (list, np.ndarray)):
                            tax = '; '.join([t.decode('utf-8') if isinstance(t, bytes) else str(t) for t in tax])
                        taxonomy.append(tax)
                
                # Get abundance data
                if 'observation/matrix/data' in f:
                    data = f['observation/matrix/data'][()]
                    if isinstance(data, h5py.Dataset):
                        data = data[()]
                    
                    # Check if we need to construct a sparse matrix
                    if len(data.shape) == 1:
                        indices = f['observation/matrix/indices'][()]
                        indptr = f['observation/matrix/indptr'][()]
                        matrix = csr_matrix((data, indices, indptr), shape=(n_obs, n_samples))
                        abundance_data = matrix.toarray()
                    else:
                        abundance_data = data
            else:
                    raise ValueError("Could not find abundance data in BIOM file")
                
                # Create DataFrame
                df = pd.DataFrame(abundance_data, index=obs_ids, columns=sample_ids)
                if taxonomy is not None:
                    df['taxonomy'] = taxonomy
                
                # Add presence column
                count_cols = [col for col in df.columns if col != 'taxonomy']
                df['presence'] = (df[count_cols] > 0).sum(axis=1)
                
                logging.info(f"Successfully read HDF5 BIOM file: {df.shape}")
                return df
                
        except Exception as e:
            logging.error(f"Error reading BIOM file: {str(e)}")
            return None

    def download_abundance_data(self, study_id, analysis_id):
        """Download abundance data for a study."""
        try:
            # Get all files for this analysis
            files = self.get_analysis_files(analysis_id)
            if not files:
                logging.warning(f"No files found for {study_id}")
                return None, None
            
            # Log all files found
            for file in files:
                logging.info("Found file: ")
                logging.info(f"Description: {file.get('attributes', {}).get('description', {})}")
            
            # Look for OTU or abundance files
            otu_files = [f for f in files if any(term in f.get('attributes', {}).get('description', {}).get('label', '').lower() 
                        for term in ['otu', 'taxonomic assignments'])]
            
            if not otu_files:
                logging.warning(f"No abundance file URL found for {study_id}")
                return None, None
            
            # Get the file URL
            file_url = None
            for otu_file in otu_files:
                file_url = otu_file.get('links', {}).get('self', None)
                if file_url:
                    break
            
            if not file_url:
                logging.warning(f"No abundance file URL found for {study_id}")
                return None, None
            
            logging.info(f"Found TSV OTU file: ")
            logging.info(f"URL: {file_url}")
            
            # Download the file
            logging.info(f"Downloading abundance file from {file_url}")
            response = requests.get(file_url)
            if response.status_code != 200:
                logging.warning(f"Failed to download abundance file for {study_id}")
                return None, None
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(response.content)
                temp_path = temp_file.name
            
            logging.info(f"Saved to temporary file: {temp_path}")
            
            # Try reading as BIOM file first
            try:
                df = self.read_biom_file(temp_path)
                if df is not None:
                    logging.info(f"Successfully read BIOM file: {df.shape}")
                    return df, None
            except Exception as e:
                logging.info(f"Not a BIOM file or error reading it: {str(e)}")
            
            # If not a BIOM file, try reading as TSV/CSV
            return self.read_abundance_file(temp_path), None
            
        except Exception as e:
            logging.error(f"Error downloading abundance data: {str(e)}")
            return None, None

    def process_abundance_data(self, df):
        """Process the abundance data by filtering rare taxa and applying log transformation."""
        try:
            # Identify taxonomy and count columns
            taxonomy_patterns = ['taxonomy', 'tax', 'species', 'genus', 'family', 'phylum', 'class', 'order']
            taxonomy_cols = []
            
            # First look for exact matches in column names
            for col in df.columns:
                if any(pattern in str(col).lower() for pattern in taxonomy_patterns):
                    taxonomy_cols.append(col)
                    logging.info(f"Found taxonomy column by name: {col}")
            
            # If no taxonomy columns found by name, look for columns with taxonomic-like content
            if not taxonomy_cols:
                for col in df.columns:
                    # Check first few values for taxonomic-like content
                    sample_values = df[col].head().astype(str)
                    if any(any(tax_term in val.lower() for tax_term in ['k__', 'p__', 'c__', 'o__', 'f__', 'g__', 's__'])
                           for val in sample_values):
                        taxonomy_cols.append(col)
                        logging.info(f"Found taxonomy column by content: {col}")
            
            # If still no taxonomy columns, look for columns with semicolons
            if not taxonomy_cols:
                for col in df.columns:
                    sample_values = df[col].head().astype(str)
                    if any(';' in val for val in sample_values):
                        taxonomy_cols.append(col)
                        logging.info(f"Found taxonomy column by semicolon: {col}")
                        
            # If still no taxonomy columns, take the last column if it's string type
            if not taxonomy_cols and df.shape[1] > 1:
                last_col = df.columns[-1]
                if pd.api.types.is_string_dtype(df[last_col]):
                    taxonomy_cols = [last_col]
                    logging.info(f"Using last string column as taxonomy: {last_col}")
            
            if not taxonomy_cols:
                logging.warning("Could not identify taxonomy columns")
                return None
            
            # Identify count columns (numeric columns that aren't taxonomy)
            count_cols = []
            for col in df.columns:
                if col not in taxonomy_cols:
                    # Try converting to numeric
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        if not df[col].isna().all():  # If not all values are NA
                            count_cols.append(col)
                            logging.info(f"Found count column: {col}")
                    except:
                        continue
            
            if not count_cols:
                logging.warning("Could not identify count columns")
                return None
                
            # Keep only taxonomy and count columns
            df = df[taxonomy_cols + count_cols]
            logging.info(f"Final columns: {df.columns.tolist()}")
            
            # Filter rare taxa (present in less than 1% of samples)
            min_samples = max(1, int(0.01 * len(count_cols)))
            df['presence'] = (df[count_cols] > 0).sum(axis=1)
            df = df[df['presence'] >= min_samples]
            df = df.drop('presence', axis=1)
            
            # Apply log transformation to counts (log(x + 1))
            for col in count_cols:
                df[col] = np.log1p(df[col])
            
            return df
            
        except Exception as e:
            logging.warning(f"Error processing abundance data: {str(e)}")
            return None
    
    def read_abundance_file(self, file_path):
        """Read an abundance file in text format (TSV/CSV)."""
        try:
            # Try different encodings and delimiters
            encodings = ['utf-8', 'latin1', 'cp1252']
            delimiters = ['\t', ',', ';']
            abundance_df = None
            
            for encoding in encodings:
                if abundance_df is not None:
                    break
                    
                try:
                    # First try to detect the delimiter
                    with open(file_path, 'r', encoding=encoding) as f:
                        # Skip comment lines
                        line = f.readline()
                        while line.startswith('#'):
                            line = f.readline()
                        first_line = line.strip()
                        logging.info(f"First non-comment line with {encoding}: {first_line[:100]}")
                        
                        # Try to detect the delimiter from the first line
                        if '\t' in first_line:
                            delimiters.insert(0, '\t')  # Try tab first if found
                        
                        # Try each delimiter
                        for delimiter in delimiters:
                            try:
                                # Try reading with this delimiter
                                abundance_df = pd.read_csv(file_path, encoding=encoding, sep=delimiter, 
                                                      comment='#', skip_blank_lines=True)
                                logging.info(f"Successfully read file with {encoding} and delimiter '{delimiter}'")
                                logging.info(f"Shape: {abundance_df.shape}")
                                logging.info(f"Columns: {abundance_df.columns.tolist()}")
                                
                                # Verify this looks like an OTU table
                                if abundance_df.shape[1] > 1:  # Should have multiple columns
                                    # Check if any column has taxonomic information
                                    has_taxonomy = False
                                    for col in abundance_df.columns:
                                        sample_values = abundance_df[col].head().astype(str)
                                        if any(any(tax_term in val.lower() for tax_term in ['k__', 'p__', 'c__', 'o__', 'f__', 'g__', 's__'])
                                               for val in sample_values):
                                            has_taxonomy = True
                                            break
                                    if has_taxonomy:
                                        logging.info("Found taxonomic information in the data")
                                        break
                                    else:
                                        logging.info("No taxonomic information found in the data")
                                        abundance_df = None
        else:
                                    logging.info("File has only one column")
                                    abundance_df = None
                                    
                            except Exception as e:
                                logging.info(f"Error reading with {encoding} and delimiter '{delimiter}': {str(e)}")
                                abundance_df = None
                            
                except Exception as e:
                    logging.info(f"Failed to decode with {encoding}")
                    abundance_df = None

            if abundance_df is not None:
                # Process the abundance data
                try:
                    # Find taxonomy column
                    taxonomy_col = None
                    for col in abundance_df.columns:
                        sample_values = abundance_df[col].head().astype(str)
                        if any(any(tax_term in val.lower() for tax_term in ['k__', 'p__', 'c__', 'o__', 'f__', 'g__', 's__'])
                               for val in sample_values):
                            taxonomy_col = col
                            logging.info(f"Found taxonomy column by content: {col}")
                            break
                    
                    if taxonomy_col is None:
                        # Try to find taxonomy column by name
                        possible_tax_cols = ['taxonomy', 'Taxonomy', 'tax', 'Tax']
                        for col in possible_tax_cols:
                            if col in abundance_df.columns:
                                taxonomy_col = col
                                logging.info(f"Found taxonomy column by name: {col}")
                                break
                    
                    if taxonomy_col is None:
                        # Use the last column as taxonomy if no other is found
                        taxonomy_col = abundance_df.columns[-1]
                        logging.info(f"Using last column as taxonomy: {taxonomy_col}")
                    
                    # Get count columns (all columns except taxonomy)
                    count_cols = [col for col in abundance_df.columns if col != taxonomy_col]
                    logging.info(f"Found count columns: {count_cols}")
                    
                    # Reorder columns to put taxonomy first
                    cols = [taxonomy_col] + count_cols
                    abundance_df = abundance_df[cols]
                    
                    # Add presence column (number of samples where OTU is present)
                    abundance_df['presence'] = (abundance_df[count_cols] > 0).sum(axis=1)
                    
                    logging.info(f"Successfully processed abundance data: {abundance_df.shape}")
                    return abundance_df
                    
                except Exception as e:
                    logging.error(f"Error processing abundance data: {str(e)}")
                    return None
            else:
                logging.warning("Could not read abundance file with any encoding/delimiter combination")
                return None
                
        except Exception as e:
            logging.error(f"Error reading abundance file: {str(e)}")
            return None

if __name__ == "__main__":
    downloader = MGnifyDownloader()
    abundance_df, metadata_df = downloader.download_and_process() 