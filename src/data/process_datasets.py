import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import json
import ast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetProcessor:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.label_encoders = {}
    
    def safe_lower(self, value):
        """Safely convert a value to lowercase string."""
        if value is None:
            return ""
        return str(value).lower()

    def extract_body_site(self, sample_desc):
        """Extract body site from sample description."""
        if not sample_desc:
            return "unknown"
        
        sample_desc = str(sample_desc).lower()
        if "vaginal" in sample_desc:
            return "vaginal"
        elif "saliva" in sample_desc:
            return "oral"
        elif any(term in sample_desc for term in ["rectum", "stool", "ructum"]):
            return "gut"
        else:
            return "unknown"

    def load_mgnify_data(self, study_id):
        """Load data from MGnify study."""
        try:
            study_dir = self.data_dir / "mgnify" / study_id
            
            # Find abundance file
            abundance_files = list(study_dir.glob("*_taxonomy_abundances_*.tsv"))
            if not abundance_files:
                logger.error(f"No abundance file found for study {study_id}")
                return None, None
            
            # Load abundance data
            logger.info(f"Loading abundance data from {abundance_files[0]}")
            abundance_df = pd.read_csv(abundance_files[0], sep='\t')
            abundance_df.set_index('#SampleID', inplace=True)
            
            # Get run IDs from columns
            run_ids = [col for col in abundance_df.columns if col.startswith('DRR')]
            logger.info(f"Found {len(run_ids)} run IDs in abundance data")
            
            # Load metadata
            metadata_path = study_dir / "samples.tsv"
            if not metadata_path.exists():
                logger.error(f"No metadata file found for study {study_id}")
                return None, None
            
            logger.info(f"Loading metadata from {metadata_path}")
            metadata_df = pd.read_csv(metadata_path, sep='\t')
            
            # Process metadata
            processed_metadata = []
            for _, row in metadata_df.iterrows():
                try:
                    attrs = ast.literal_eval(row['attributes'])
                    sample_desc = self.safe_lower(attrs.get('sample-desc'))
                    
                    metadata_entry = {
                        'sample_id': row['id'],
                        'biosample': self.safe_lower(attrs.get('biosample')),
                        'sample_desc': sample_desc,
                        'body_site': self.extract_body_site(sample_desc),
                        'species': self.safe_lower(attrs.get('species')),
                        'environment': self.safe_lower(attrs.get('environment-biome', '')),
                        'disease_state': 'disease' if any(term in sample_desc for term in 
                            ['disease', 'infection', 'disorder', 'patient']) else 'healthy'
                    }
                    processed_metadata.append(metadata_entry)
                except Exception as e:
                    logger.warning(f"Error processing metadata for sample {row['id']}: {e}")
                    continue
            
            if not processed_metadata:
                logger.error("No metadata could be processed")
                return None, None
            
            metadata_df = pd.DataFrame(processed_metadata)
            
            # Count body sites
            site_counts = metadata_df['body_site'].value_counts()
            for site, count in site_counts.items():
                logger.info(f"Found {count} {site} samples")
            
            return abundance_df[run_ids], metadata_df
            
        except Exception as e:
            logger.error(f"Error loading MGnify data for {study_id}: {str(e)}")
            return None, None
    
    def preprocess_abundance_data(self, abundance_df, min_prevalence=0.1):
        """Preprocess abundance data."""
        try:
            logger.info("Preprocessing abundance data...")
            
            # Remove features with low prevalence
            prevalence = (abundance_df > 0).mean()
            keep_features = prevalence[prevalence >= min_prevalence].index
            filtered_df = abundance_df[keep_features]
            
            # Log transform
            transformed_df = np.log1p(filtered_df)
            
            # Scale features
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(transformed_df)
            scaled_df = pd.DataFrame(scaled_data, 
                                   index=transformed_df.index,
                                   columns=transformed_df.columns)
            
            return scaled_df
            
        except Exception as e:
            logger.error(f"Error preprocessing abundance data: {str(e)}")
            return None
    
    def prepare_features_for_training(self, classification_type='disease_state'):
        """Prepare features for model training.
        
        Args:
            classification_type: Either 'disease_state' or 'body_site'
        """
        try:
            all_features = []
            all_targets = []
            
            # Process MGnify data
            for study_id in ["MGYS00005745"]:  # Add more study IDs as needed
                logger.info(f"Processing MGnify study {study_id}")
                abundance_df, metadata_df = self.load_mgnify_data(study_id)
                
                if abundance_df is not None and metadata_df is not None:
                    # Preprocess abundance data
                    processed_df = self.preprocess_abundance_data(abundance_df)
                    if processed_df is not None:
                        # Create mapping between run IDs and metadata
                        run_to_metadata = {}
                        for _, meta_row in metadata_df.iterrows():
                            sample_desc = meta_row['sample_desc']
                            body_site = meta_row['body_site']
                            disease_state = meta_row['disease_state']
                            
                            # Find matching run IDs in abundance data
                            for run_id in processed_df.index:
                                run_to_metadata[run_id] = {
                                    'body_site': body_site,
                                    'disease_state': disease_state
                                }
                        
                        # Filter processed data to include only runs with metadata
                        valid_runs = list(run_to_metadata.keys())
                        processed_df = processed_df.loc[valid_runs]
                        
                        # Get targets based on classification type
                        if classification_type == 'disease_state':
                            targets = [1 if run_to_metadata[run]['disease_state'] == 'disease' else 0 
                                     for run in processed_df.index]
                        else:  # body_site
                            if 'body_site' not in self.label_encoders:
                                self.label_encoders['body_site'] = LabelEncoder()
                            body_sites = [run_to_metadata[run]['body_site'] 
                                        for run in processed_df.index]
                            targets = self.label_encoders['body_site'].fit_transform(body_sites)
                        
                        all_features.append(processed_df)
                        all_targets.extend(targets)
            
            if not all_features:
                logger.error("No features could be prepared")
                return False
            
            # Combine features
            combined_features = pd.concat(all_features, axis=0)
            combined_targets = pd.Series(all_targets)
            
            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                combined_features, combined_targets, test_size=0.2, random_state=42
            )
            
            # Handle class imbalance with SMOTE
            logger.info("Handling class imbalance with SMOTE...")
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            
            # Train a simple model to check accuracy
            logger.info("Training a Random Forest model to check accuracy...")
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train_resampled, y_train_resampled)
            
            # Evaluate the model
            y_pred = rf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"\nModel Accuracy: {accuracy:.2%}")
            logger.info("\nClassification Report:")
            logger.info(classification_report(y_test, y_pred))
            
            # Save processed data
            output_dir = self.processed_dir / classification_type
            output_dir.mkdir(parents=True, exist_ok=True)
            
            X_train_resampled = pd.DataFrame(X_train_resampled, columns=X_train.columns)
            X_test = pd.DataFrame(X_test, columns=X_train.columns)
            
            X_train_resampled.to_csv(output_dir / "features_train.csv")
            X_test.to_csv(output_dir / "features_test.csv")
            pd.Series(y_train_resampled).to_csv(output_dir / "labels_train.csv")
            pd.Series(y_test).to_csv(output_dir / "labels_test.csv")
            
            # Save label encoder mapping if used
            if classification_type == 'body_site' and 'body_site' in self.label_encoders:
                encoder_mapping = dict(zip(
                    self.label_encoders['body_site'].classes_,
                    self.label_encoders['body_site'].transform(
                        self.label_encoders['body_site'].classes_)
                ))
                with open(output_dir / "label_mapping.json", 'w') as f:
                    json.dump(encoder_mapping, f, indent=2)
            
            logger.info(f"Successfully saved processed data to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return False

def get_body_site(sample_desc):
    if 'vaginal' in sample_desc.lower():
        return 'vaginal'
    elif 'oral' in sample_desc.lower():
        return 'oral'
    elif 'intestinal' in sample_desc.lower() or 'fecal' in sample_desc.lower():
        return 'intestinal'
    else:
        return 'unknown'

def process_data():
    logging.info("\nProcessing data for body site classification...")
    
    # Process MGnify studies
    studies = ["MGYS00005745"]
    abundance_dfs = []
    metadata_dfs = []
    
    for study in studies:
        logging.info(f"Processing MGnify study {study}")
        
        # Load abundance data
        abundance_file = f"data/mgnify/{study}/DRP006831_taxonomy_abundances_SSU_v5.0.tsv"
        logging.info(f"Loading abundance data from {abundance_file}")
        try:
            abundance_df = pd.read_csv(abundance_file, sep='\t', index_col=0)
            abundance_dfs.append(abundance_df)
        except Exception as e:
            logging.error(f"Error loading abundance data for study {study}: {str(e)}")
            continue
            
        # Load metadata
        metadata_file = f"data/mgnify/{study}/samples.tsv"
        logging.info(f"Loading metadata from {metadata_file}")
        try:
            metadata_df = pd.read_csv(metadata_file, sep='\t')
            # Extract sample description from attributes
            metadata_df['sample_desc'] = metadata_df['attributes'].apply(lambda x: eval(x)['sample-desc'])
            # Get body site from sample description
            metadata_df['body_site'] = metadata_df['sample_desc'].apply(get_body_site)
            metadata_dfs.append(metadata_df)
        except Exception as e:
            logging.error(f"Error loading metadata for study {study}: {str(e)}")
            continue
    
    if not abundance_dfs or not metadata_dfs:
        logging.error("No data could be loaded")
        return
        
    # Combine data
    abundance_df = pd.concat(abundance_dfs, axis=1)
    metadata_df = pd.concat(metadata_dfs)
    
    # Prepare features
    try:
        prepare_features(abundance_df, metadata_df, target_col='body_site')
    except Exception as e:
        logging.error(f"Error preparing features: {str(e)}")

if __name__ == "__main__":
    # Process data for both classification tasks
    processor = DatasetProcessor()
    
    logger.info("\nProcessing data for body site classification...")
    processor.prepare_features_for_training(classification_type='body_site') 