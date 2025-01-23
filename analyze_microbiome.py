import pandas as pd
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class MicrobiomeAnalyzer:
    def __init__(self):
        self.features_dir = Path("data/features")
        self.integrated_dir = Path("data/integrated")
        self.output_dir = Path("data/analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for plots
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
    
    def load_data(self):
        """Load processed features and metadata."""
        self.features = pd.read_csv(self.features_dir / "processed_features.csv")
        self.metadata = pd.read_csv(self.integrated_dir / "integrated_metadata.csv", skiprows=[0])  # Skip the long header
        self.health_assoc = pd.read_csv(self.integrated_dir / "health_associations.csv")
        
        # Ensure metadata and features are aligned
        if len(self.metadata) != len(self.features):
            logging.warning(f"Size mismatch: metadata ({len(self.metadata)}) vs features ({len(self.features)})")
            # Keep only the first n rows where n is the minimum length
            min_len = min(len(self.metadata), len(self.features))
            self.metadata = self.metadata.iloc[:min_len]
            self.features = self.features.iloc[:min_len]
        
        logging.info(f"Loaded {self.features.shape[1]} features for {self.features.shape[0]} samples")
    
    def analyze_diversity_metrics(self):
        """Analyze and visualize diversity metrics."""
        diversity_cols = ['shannon_diversity', 'species_richness', 'evenness']
        
        # Create distribution plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, col in enumerate(diversity_cols):
            sns.histplot(data=self.features, x=col, kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col.replace("_", " ").title()}')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "diversity_distributions.png")
        plt.close()
        
        # Calculate summary statistics
        diversity_stats = self.features[diversity_cols].describe()
        diversity_stats.to_csv(self.output_dir / "diversity_statistics.csv")
        logging.info("Saved diversity analysis results")
    
    def analyze_pca_components(self):
        """Analyze PCA components and create TSNE visualization."""
        pca_cols = [col for col in self.features.columns if col.startswith('PC')]
        
        # Calculate explained variance per component
        explained_var = pd.DataFrame({
            'Component': range(1, len(pca_cols) + 1),
            'Explained_Variance': [
                np.var(self.features[f'PC{i+1}']) for i in range(len(pca_cols))
            ]
        })
        explained_var['Cumulative_Variance'] = explained_var['Explained_Variance'].cumsum()
        
        # Plot explained variance
        plt.figure(figsize=(10, 6))
        plt.plot(explained_var['Component'], explained_var['Cumulative_Variance'], 'b-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Cumulative Explained Variance')
        plt.savefig(self.output_dir / "pca_explained_variance.png")
        plt.close()
        
        # Perform t-SNE for visualization
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(self.features[pca_cols])
        
        # Create t-SNE plot colored by health status
        plt.figure(figsize=(10, 8))
        health_status = self.metadata['Subject health status (Healthy or Non-healthy)']
        
        # Create scatter plot
        if not health_status.isna().all():
            # Create categorical mapping for health status
            health_categories = pd.Categorical(health_status)
            category_colors = health_categories.codes
            
            scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                                c=category_colors,
                                cmap='tab20', alpha=0.6)
            
            # Create legend
            legend_elements = [plt.scatter([], [], c=plt.cm.tab20(i/len(health_categories.categories)), 
                                        label=cat) for i, cat in enumerate(health_categories.categories)]
            plt.legend(handles=legend_elements,
                      title="Health Status",
                      bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.6)
        
        plt.title('t-SNE Visualization of PCA Components')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.tight_layout()
        plt.savefig(self.output_dir / "tsne_visualization.png", bbox_inches='tight')
        plt.close()
        
        logging.info("Saved PCA analysis results")
    
    def analyze_metadata_correlations(self):
        """Analyze correlations between metadata and diversity metrics."""
        diversity_cols = ['shannon_diversity', 'species_richness', 'evenness']
        
        # Convert age to numeric, handling non-numeric values
        self.metadata['Age (Years)'] = pd.to_numeric(self.metadata['Age (Years)'], errors='coerce')
        
        # Convert BMI to numeric, handling non-numeric values
        self.metadata['BMI (kgm²)'] = self.metadata['BMI (kgm²)'].replace('–', np.nan)
        self.metadata['BMI (kgm²)'] = pd.to_numeric(self.metadata['BMI (kgm²)'], errors='coerce')
        
        metadata_cols = ['Age (Years)', 'BMI (kgm²)']
        
        # Calculate correlations for samples with available metadata
        correlation_data = pd.concat([
            self.features[diversity_cols],
            self.metadata[metadata_cols]
        ], axis=1).dropna()
        
        if len(correlation_data) > 0:
            corr_matrix = correlation_data.corr()
            
            # Create correlation heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix: Metadata vs Diversity')
            plt.tight_layout()
            plt.savefig(self.output_dir / "metadata_correlations.png")
            plt.close()
            
            logging.info(f"Saved metadata correlation analysis using {len(correlation_data)} complete samples")
        else:
            logging.warning("No complete samples available for correlation analysis")
    
    def run_analysis(self):
        """Run all analyses."""
        self.load_data()
        self.analyze_diversity_metrics()
        self.analyze_pca_components()
        self.analyze_metadata_correlations()
        logging.info("Completed all analyses")

def main():
    try:
        analyzer = MicrobiomeAnalyzer()
        analyzer.run_analysis()
    except Exception as e:
        logging.error(f"Error in analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 