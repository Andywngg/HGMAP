from pathlib import Path
import logging
from data.integrator import DataIntegrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the data integration process"""
    try:
        # Set up paths
        data_dir = Path("data/processed")
        output_dir = Path("data/integrated")
        
        logger.info("Starting data integration process...")
        
        # Initialize integrator
        integrator = DataIntegrator()
        
        # Run integration
        metadata, taxonomy, health_assoc = integrator.integrate_all_data(
            data_dir=data_dir,
            output_dir=output_dir
        )
        
        # Log summary statistics
        logger.info("\nIntegration Summary:")
        logger.info(f"Total samples: {len(metadata)}")
        logger.info(f"Total species: {len(taxonomy)}")
        logger.info(f"Health associations: {len(health_assoc)}")
        
        # Log health status distribution
        if 'health_status' in metadata.columns:
            health_dist = metadata['health_status'].value_counts()
            logger.info("\nHealth Status Distribution:")
            for status, count in health_dist.items():
                logger.info(f"{status}: {count}")
        
        logger.info("\nData integration completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in data integration process: {e}")
        raise

if __name__ == "__main__":
    main() 