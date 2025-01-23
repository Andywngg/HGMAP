from pathlib import Path
from utils.data_converter import DataConverter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Set up paths
    input_dir = Path("data/raw")
    output_dir = Path("data/processed")
    
    logger.info("Starting data conversion...")
    
    # Convert all files
    try:
        conversion_map = DataConverter.convert_all(input_dir, output_dir)
        
        # Print results
        logger.info("\nConversion Results:")
        for original, converted in conversion_map.items():
            logger.info(f"\nOriginal file: {original}")
            logger.info("Converted to:")
            for path in converted:
                logger.info(f"  - {path}")
                
        logger.info("\nConversion completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        raise

if __name__ == "__main__":
    main() 