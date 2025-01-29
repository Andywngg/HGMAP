import logging
import os
from pathlib import Path
from processor_v4 import EnhancedMicrobiomeProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Initialize processor
        processor = EnhancedMicrobiomeProcessor()
        
        # Set file paths
        base_dir = Path(os.getcwd())
        data_dir = base_dir / 'data' / 'integrated'
        abundance_path = data_dir / 'abundance_data.csv'
        metadata_path = data_dir / 'integrated_metadata.csv'
        
        # Process data
        logger.info("Starting enhanced microbiome processing...")
        results = processor.process_integrated_data(abundance_path, metadata_path)
        
        # Generate and print report
        report = processor.generate_report(results)
        
        logger.info("Analysis Report:")
        logger.info(f"Data Summary: {report['data_summary']}")
        logger.info(f"Model Performance: {report['model_performance']}")
        
        if 'top_features' in report:
            logger.info("Top 10 Important Features:")
            for feature in report['top_features']:
                logger.info(f"  {feature['feature']}: {feature['importance']:.4f}")
                
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 