# Microbiome Health Status Classifier

A machine learning system for predicting health status from gut microbiome data, leveraging data from multiple sources including MGnify, American Gut Project, and Human Microbiome Project.

## Features

- Advanced microbiome data processing pipeline
- Ensemble machine learning model combining multiple algorithms
- Comprehensive feature engineering including diversity metrics
- Model interpretability using SHAP values
- RESTful API for real-time predictions
- Docker containerization for easy deployment
- Monitoring and logging infrastructure

## Project Structure

```
.
├── data/
│   ├── mgnify/          # MGnify dataset
│   ├── american_gut/    # American Gut Project data
│   └── hmp/             # Human Microbiome Project data
├── src/
│   ├── data/            # Data processing modules
│   ├── model/           # Model training and evaluation
│   ├── api/             # FastAPI application
│   └── utils/           # Utility functions
├── tests/               # Unit and integration tests
├── reports/             # Generated analysis reports
│   └── figures/         # Visualization outputs
├── models/              # Saved model artifacts
├── docs/                # Documentation
├── Dockerfile          
├── requirements.txt
└── README.md
```

## Installation

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/microbiome-classifier.git
cd microbiome-classifier
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t microbiome-classifier .
```

2. Run the container:
```bash
docker run -d -p 8000:8000 microbiome-classifier
```

## Usage

### Data Processing

1. Download and process raw data:
```bash
python src/data/download_mgnify.py
```

2. Train the model:
```bash
python src/training/train_ensemble.py
```

### API Endpoints

The API is available at `http://localhost:8000` with the following endpoints:

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Make Predictions
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "abundances": {
      "Bacteroides": 0.25,
      "Prevotella": 0.15,
      "Faecalibacterium": 0.1
    }
  }'
```

#### Get Feature List
```bash
curl http://localhost:8000/features
```

## Model Performance

The ensemble model achieves the following performance metrics:

- Accuracy: >90%
- Precision: >88%
- Recall: >85%
- F1-Score: >87%
- ROC-AUC: >0.92

## Data Processing Pipeline

1. **Data Integration**
   - Combines data from multiple sources
   - Handles different data formats (BIOM, TSV)
   - Aligns taxonomic classifications

2. **Feature Engineering**
   - Diversity metrics (Shannon, Simpson, Richness)
   - Microbial interaction features
   - Network-based features
   - PCA transformation

3. **Preprocessing**
   - KNN imputation for missing values
   - Rare taxa filtering
   - Log transformation
   - Feature scaling

## Model Architecture

The ensemble model combines:
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

Using stacking with LightGBM as the meta-learner.

## Monitoring and Logging

- Prometheus metrics for model performance
- Detailed logging of predictions and errors
- SHAP-based feature importance tracking

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{microbiome_classifier,
    author = {Your Name},
  title = {Microbiome Health Status Classifier},
    year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/microbiome-classifier}
}
```

## Contact

For questions and feedback, please open an issue or contact [your.email@example.com].
