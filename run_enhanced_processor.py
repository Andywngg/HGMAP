from src.data.processor_v4 import EnhancedMicrobiomeProcessor
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Starting enhanced microbiome processing...")
        
        # Initialize and run processor
        processor = EnhancedMicrobiomeProcessor()
        results = processor.train_and_evaluate()
        
        # Display results
        logger.info("\nModel Performance Summary:")
        for model_name, scores in results.items():
            logger.info(f"{model_name.upper()}:")
            logger.info(f"  Mean ROC-AUC: {scores['mean_score']:.3f}")
            logger.info(f"  Std ROC-AUC: {scores['std_score']:.3f}")
            logger.info(f"  CV Scores: {scores['cv_scores']}\n")
            
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()