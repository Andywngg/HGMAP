# AI-Based Microbiome Analysis for Disease Detection

## Overview
This project develops an AI-powered system for analyzing microbiome data to detect early signs of disease. It leverages datasets from the American Gut Project and Human Microbiome Project to train machine learning models that can identify correlations between gut microbiota composition and health conditions.

## Features
- Advanced microbiome data preprocessing and feature engineering
- Ensemble machine learning models (Random Forest, Gradient Boosting, StackingClassifier)
- Model interpretability using SHAP values
- RESTful API for real-time analysis
- Comprehensive testing suite
- Containerized deployment

## Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/microbiome-analysis.git
cd microbiome-analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### Data Processing
```bash
python src/main.py process-data --input-dir data/raw --output-dir data/processed
```

### Model Training
```bash
python src/train_advanced.py --config config/training_config.yaml
```

### API
```bash
uvicorn src.api.main:app --reload
```

## Project Structure
```
├── config/              # Configuration files
├── data/               # Data directory
│   ├── raw/           # Raw data
│   └── processed/     # Processed data
├── src/               # Source code
│   ├── api/          # FastAPI implementation
│   ├── features/     # Feature engineering
│   ├── model/        # ML models
│   └── utils/        # Utility functions
├── tests/            # Test suite
├── deploy/           # Deployment configurations
└── notebooks/        # Jupyter notebooks
```

## API Documentation
The API is available at `http://localhost:8000` when running locally.
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Testing
Run tests with:
```bash
pytest tests/
```

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.
