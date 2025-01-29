from data.processor import MicrobiomeDataProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        processor = MicrobiomeDataProcessor()
        features, targets = processor.prepare_features_for_training()
        
        if features is not None and targets is not None:
            logger.info(f"Successfully processed data: {features.shape[0]} samples, {features.shape[1]} features")
            logger.info(f"Class distribution: {targets.value_counts()}")
        else:
            logger.error("Failed to process data")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main() 