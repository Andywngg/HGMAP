#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.data.process_datasets import DatasetProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Process the datasets for health status classification."""
    processor = DatasetProcessor()
    
    # Process data for health status classification
    logger.info("Processing data for health status classification...")
    success = processor.prepare_features_for_training(classification_type='health_status')
    if success:
        logger.info("Successfully processed data for health status classification")
    else:
        logger.error("Failed to process data for health status classification")

if __name__ == "__main__":
    main() 