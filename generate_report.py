import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ReportGenerator:
    def __init__(self):
        self.analysis_dir = Path("data/analysis")
        self.features_dir = Path("data/features")
        self.integrated_dir = Path("data/integrated")
        self.output_dir = Path("reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_analysis_data(self):
        """Load analysis results and source data."""
        self.diversity_stats = pd.read_csv(self.analysis_dir / "diversity_statistics.csv")
        self.metadata = pd.read_csv(self.integrated_dir / "integrated_metadata.csv", skiprows=[0])
        self.features = pd.read_csv(self.features_dir / "processed_features.csv")
    
    def generate_dataset_summary(self):
        """Generate summary of the dataset composition."""
        # Count samples by health status
        health_counts = self.metadata['Subject health status (Healthy or Non-healthy)'].value_counts()
        
        # Count samples by phenotype
        phenotype_counts = self.metadata['Phenotype'].value_counts()
        
        # Count samples by geography
        geography_counts = self.metadata['Geography (Country)'].value_counts()
        
        # Calculate age and BMI statistics
        self.metadata['Age (Years)'] = pd.to_numeric(self.metadata['Age (Years)'], errors='coerce')
        self.metadata['BMI (kgm²)'] = self.metadata['BMI (kgm²)'].replace('–', np.nan)
        self.metadata['BMI (kgm²)'] = pd.to_numeric(self.metadata['BMI (kgm²)'], errors='coerce')
        
        age_stats = self.metadata['Age (Years)'].describe()
        bmi_stats = self.metadata['BMI (kgm²)'].describe()
        
        return {
            'health_counts': health_counts,
            'phenotype_counts': phenotype_counts,
            'geography_counts': geography_counts,
            'age_stats': age_stats,
            'bmi_stats': bmi_stats
        }
    
    def generate_diversity_summary(self):
        """Generate summary of diversity metrics."""
        return self.diversity_stats
    
    def generate_report(self):
        """Generate the full analysis report."""
        self.load_analysis_data()
        dataset_summary = self.generate_dataset_summary()
        diversity_summary = self.generate_diversity_summary()
        
        # Create report
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = []
        
        # Header
        report.append("# Microbiome Analysis Report")
        report.append(f"Generated on: {timestamp}\n")
        
        # Dataset Overview
        report.append("## 1. Dataset Overview")
        report.append(f"Total samples analyzed: {len(self.metadata)}")
        report.append(f"Total features: {self.features.shape[1]}\n")
        
        # Health Status Distribution
        report.append("### 1.1 Health Status Distribution")
        report.append("```")
        report.append(dataset_summary['health_counts'].to_string())
        report.append("```\n")
        
        # Phenotype Distribution
        report.append("### 1.2 Phenotype Distribution")
        report.append("Top 10 phenotypes:")
        report.append("```")
        report.append(dataset_summary['phenotype_counts'].head(10).to_string())
        report.append("```\n")
        
        # Geographic Distribution
        report.append("### 1.3 Geographic Distribution")
        report.append("Top 10 countries:")
        report.append("```")
        report.append(dataset_summary['geography_counts'].head(10).to_string())
        report.append("```\n")
        
        # Demographics
        report.append("### 1.4 Demographics")
        report.append("Age Statistics:")
        report.append("```")
        report.append(dataset_summary['age_stats'].to_string())
        report.append("```\n")
        
        report.append("BMI Statistics:")
        report.append("```")
        report.append(dataset_summary['bmi_stats'].to_string())
        report.append("```\n")
        
        # Diversity Analysis
        report.append("## 2. Diversity Analysis")
        report.append("### 2.1 Diversity Metrics Summary")
        report.append("```")
        report.append(diversity_summary.to_string())
        report.append("```\n")
        
        # Visualization References
        report.append("## 3. Visualizations")
        report.append("The following visualizations have been generated:")
        report.append("1. `diversity_distributions.png`: Distribution of diversity metrics")
        report.append("2. `pca_explained_variance.png`: Cumulative explained variance by PCA components")
        report.append("3. `tsne_visualization.png`: t-SNE visualization of samples colored by health status")
        report.append("4. `metadata_correlations.png`: Correlation heatmap between metadata and diversity metrics\n")
        
        # Key Findings
        report.append("## 4. Key Findings")
        
        # Diversity findings
        mean_shannon = self.features['shannon_diversity'].mean()
        mean_richness = self.features['species_richness'].mean()
        report.append(f"- Average Shannon diversity: {mean_shannon:.2f}")
        report.append(f"- Average species richness: {mean_richness:.2f}")
        
        # Health status distribution
        healthy_pct = (dataset_summary['health_counts'].get('Healthy', 0) / len(self.metadata)) * 100
        report.append(f"- Healthy samples comprise {healthy_pct:.1f}% of the dataset")
        
        # Geographic diversity
        n_countries = len(dataset_summary['geography_counts'])
        report.append(f"- Samples collected from {n_countries} different countries")
        
        # Save report
        report_path = self.output_dir / "analysis_report.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        logging.info(f"Report generated and saved to {report_path}")

def main():
    try:
        generator = ReportGenerator()
        generator.generate_report()
    except Exception as e:
        logging.error(f"Error generating report: {str(e)}")
        raise

if __name__ == "__main__":
    main() 