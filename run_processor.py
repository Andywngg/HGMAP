from src.data.processor_v4 import EnhancedMicrobiomeProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info('Starting enhanced microbiome processing...')
        processor = EnhancedMicrobiomeProcessor()
        pipeline, results = processor.train_and_evaluate()
        
        logger.info('\nModel Performance Summary:')
        for model_name, scores in results.items():
            logger.info(f'{model_name.upper()}:')
            logger.info(f'Mean ROC-AUC: {scores["mean_score"]:.3f}')
            logger.info(f'Std ROC-AUC: {scores["std_score"]:.3f}')
        
        logger.info('Processing completed successfully!')
        
    except Exception as e:
        logger.error(f'Error in main: {str(e)}')
        raise

if __name__ == '__main__':
    main() 