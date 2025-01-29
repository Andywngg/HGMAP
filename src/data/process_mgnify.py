#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict, Any
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MGnifyDataProcessor:
    """Process and combine MGnify data."""
    
    def __init__(self, data_dir: str = 'data/mgnify'):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / 'processed'
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_study_data(self, study_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load abundance and metadata for a study."""
        try:
            # Load abundance data
            abundance_file = study_dir / 'abundance.tsv'
            if not abundance_file.exists():
                logger.warning(f"No abundance file found in {study_dir}")
                return None, None
            
            # Read abundance file with specific format handling
            try:
                # Read the file skipping comment lines but keeping the header
                with open(abundance_file) as f:
                    header = None
                    for line in f:
                        if line.startswith('# OTU ID') or line.startswith('#OTU ID'):
                            header = line.strip('# \n').split('\t')
                            break
                
                if header is None:
                    logger.warning(f"Could not find header in {abundance_file}")
                    return None, None
                
                # Read the file with the correct header
                abundance_df = pd.read_csv(abundance_file, sep='\t', comment='#', names=header)
                
                # Standardize column names by removing any spaces and #
                abundance_df.columns = [col.strip('# ') for col in abundance_df.columns]
                
                # Check if we have the expected columns
                if 'OTU ID' not in abundance_df.columns or 'taxonomy' not in abundance_df.columns:
                    logger.warning(f"Missing required columns in {abundance_file}")
                    return None, None
                
                # Set OTU ID as index
                abundance_df = abundance_df.set_index('OTU ID')
                
                # Create metadata
                study_name = study_dir.name
                metadata = {
                    'study_id': study_name,
                    'study_name': study_name,
                    'n_otus': len(abundance_df)
                }
                
                # Add disease status based on study name
                if 'healthy' in study_name.lower():
                    metadata['condition'] = 'healthy'
                else:
                    # Extract condition from study name (e.g., MGYS00001248 (IBD) -> IBD)
                    try:
                        condition = study_name.split('(')[-1].strip(')').strip()
                        metadata['condition'] = condition
                    except:
                        metadata['condition'] = 'unknown'
                
                metadata_df = pd.DataFrame([metadata])
                
                # Get abundance columns (all except taxonomy)
                abundance_cols = [col for col in abundance_df.columns if col != 'taxonomy']
                
                # Sum abundances across samples if multiple samples exist
                if abundance_cols:
                    abundance_df['abundance'] = abundance_df[abundance_cols].sum(axis=1)
                else:
                    logger.warning(f"No abundance columns found in {abundance_file}")
                    return None, None
                
                # Create final abundance DataFrame
                final_abundance = pd.DataFrame(index=abundance_df.index)
                final_abundance['taxonomy'] = abundance_df['taxonomy']
                final_abundance['abundance'] = abundance_df['abundance']
                
                return final_abundance, metadata_df
                
            except Exception as e:
                logger.error(f"Error parsing abundance file {abundance_file}: {str(e)}")
                return None, None
            
        except Exception as e:
            logger.error(f"Error loading data from {study_dir}: {str(e)}")
            return None, None
    
    def process_abundance_data(self, abundance_df: pd.DataFrame) -> pd.DataFrame:
        """Process abundance data."""
        try:
            # Convert abundance to relative abundance
            total_abundance = abundance_df['abundance'].sum()
            abundance_df['relative_abundance'] = abundance_df['abundance'] / total_abundance
            
            # Filter low abundance taxa (relative abundance > 0.0001)
            abundance_df = abundance_df[abundance_df['relative_abundance'] > 0.0001]
            
            # Extract taxonomic levels
            tax_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
            
            for i, level in enumerate(tax_levels):
                abundance_df[level] = abundance_df['taxonomy'].apply(
                    lambda x: x.split(';')[i].split('__')[-1] if len(x.split(';')) > i else 'Unknown'
                )
            
            return abundance_df
            
        except Exception as e:
            logger.error(f"Error processing abundance data: {str(e)}")
            return None
    
    def combine_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Combine abundance and metadata from all studies."""
        all_abundance = []
        all_metadata = []
        
        for study_dir in self.data_dir.glob('MGYS*'):
            if study_dir.is_dir():
                logger.info(f"Processing {study_dir.name}...")
                abundance_df, metadata = self.load_study_data(study_dir)
                
                if abundance_df is not None and metadata is not None:
                    # Process abundance data
                    processed_df = self.process_abundance_data(abundance_df)
                    
                    if processed_df is not None:
                        # Add study ID
                        processed_df['study_id'] = study_dir.name
                        
                        # Append to lists
                        all_abundance.append(processed_df)
                        all_metadata.append(metadata)
        
        if not all_abundance:
            raise ValueError("No data could be processed")
        
        # Combine all abundance data
        combined_abundance = pd.concat(all_abundance, axis=0)
        
        # Create metadata DataFrame
        metadata_df = pd.DataFrame(pd.concat(all_metadata, ignore_index=True))
        
        logger.info(f"Combined abundance data shape: {combined_abundance.shape}")
        logger.info(f"Combined metadata shape: {metadata_df.shape}")
        
        return combined_abundance, metadata_df
    
    def save_processed_data(self, abundance_df: pd.DataFrame, metadata_df: pd.DataFrame):
        """Save processed data."""
        # Save full data
        abundance_df.to_csv(self.processed_dir / 'abundance.csv')
        metadata_df.to_csv(self.processed_dir / 'metadata.csv')
        
        # Create a summary at phylum level
        phylum_abundance = abundance_df.groupby(['study_id', 'phylum'])['relative_abundance'].sum().reset_index()
        phylum_abundance.to_csv(self.processed_dir / 'phylum_abundance.csv', index=False)
        
        # Create a summary of top 10 genera per condition
        genus_abundance = abundance_df.groupby(['study_id', 'genus'])['relative_abundance'].sum().reset_index()
        top_genera = (genus_abundance.groupby('study_id')
                     .apply(lambda x: x.nlargest(10, 'relative_abundance'))
                     .reset_index(drop=True))
        top_genera.to_csv(self.processed_dir / 'top_genera.csv', index=False)
        
        logger.info(f"Saved processed data to {self.processed_dir}")

def main():
    try:
        # Initialize processor
        processor = MGnifyDataProcessor()
        
        # Process and combine data
        abundance_df, metadata_df = processor.combine_data()
        
        # Save processed data
        processor.save_processed_data(abundance_df, metadata_df)
        
        logger.info(f"Successfully processed {len(metadata_df)} studies")
        logger.info(f"Total OTUs: {len(abundance_df)}")
        logger.info(f"Total samples: {len([col for col in abundance_df.columns if col != 'taxonomy'])}")
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise

if __name__ == "__main__":
    main() 