from src.data.processor_v3 import MicrobiomeProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Initializing MicrobiomeProcessor...")
        processor = MicrobiomeProcessor()
        
        logger.info("Starting feature preparation and model evaluation...")
        ensemble, results = processor.prepare_and_evaluate()
        
        if ensemble is None:
            logger.error("Model evaluation failed")
            return
            
        logger.info("Model evaluation completed successfully")
        logger.info("Results saved to data/results/model_evaluation.json")
        logger.info("Feature importance plot saved to data/results/feature_importance.png")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")

if __name__ == "__main__":
    main() 