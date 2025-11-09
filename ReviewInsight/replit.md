# Hotel Review Classification System

## Overview

This is an end-to-end Machine Learning system for automatic classification of hotel reviews in Italian. The system performs two main tasks:

1. **Department Classification** - Routes reviews to the appropriate hotel department:
   - Housekeeping (cleaning, room maintenance)
   - Reception (check-in/out, customer service)
   - F&B (Food & Beverage: breakfast, restaurant, bar)

2. **Sentiment Analysis** - Classifies reviews as Positive or Negative

The system includes a complete ML pipeline from synthetic data generation to model training and deployment via a Streamlit web dashboard. It enables both single review predictions and batch processing with CSV upload/download capabilities.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Core ML Pipeline

**Problem**: Automate the routing of hotel reviews to appropriate departments and identify customer satisfaction levels.

**Solution**: Multi-model approach using scikit-learn with TF-IDF vectorization and Logistic Regression classifiers.

**Key Design Decisions**:

1. **Separate Models for Each Task** - The system uses two independent Logistic Regression models:
   - Department classifier (3-class: Housekeeping, Reception, F&B)
   - Sentiment classifier (2-class: Positive, Negative)
   - **Rationale**: Decoupling allows independent optimization and easier debugging
   - **Trade-off**: Slightly more complex deployment vs. potential accuracy gains

2. **TF-IDF Feature Extraction** - Text vectorization using TfidfVectorizer
   - **Rationale**: Captures word importance while being computationally efficient
   - **Alternative Considered**: Word embeddings (Word2Vec, BERT) - rejected due to complexity for prototype
   - **Pros**: Fast, interpretable, works well with limited data
   - **Cons**: Doesn't capture semantic relationships or word order

3. **Italian Language Focus** - All text processing optimized for Italian reviews
   - No language detection or translation layer
   - Assumes all input is Italian text

### Data Layer

**Synthetic Data Generation**:
- Template-based review generation (`generate_dataset.py`)
- 500 reviews with balanced department and sentiment distribution
- Keyword-driven approach ensures clear decision boundaries
- **Rationale**: Enables rapid prototyping without real customer data

**Data Structure**:
- CSV format with columns: title, body, department, sentiment
- Combined title+body text for model input
- Clean separation between raw and preprocessed text

### Text Preprocessing Pipeline

**Components** (in `utils.py`):
1. Lowercasing - normalize case variations
2. Punctuation removal - reduce noise
3. Whitespace normalization - consistent tokenization
4. Title+body combination - maximize available features

**Design Choice**: Minimal preprocessing to preserve Italian language characteristics
- No stemming/lemmatization (can damage Italian morphology)
- No stopword removal (may lose sentiment signals)

### Model Training Architecture

**Pipeline** (`train_model.py`):
1. Data loading and validation
2. Train/test split (typically 80/20)
3. TF-IDF vectorizer fitting on training data
4. Separate model training for department and sentiment
5. Evaluation with accuracy, F1-score, classification reports
6. Model serialization to disk

**Model Persistence**:
- Models saved as pickle files in `models/` directory:
  - `vectorizer.pkl` - TF-IDF transformer
  - `department_model.pkl` - Department classifier
  - `sentiment_model.pkl` - Sentiment classifier
- Metrics saved to `results/metrics.pkl` for reporting

**Rationale**: Simple pickle serialization for rapid prototyping
- **Pros**: Easy to implement, works across Python environments
- **Cons**: Version sensitivity, less secure than alternatives (joblib, ONNX)

### Web Interface Architecture

**Framework**: Streamlit (`app.py`)
- **Rationale**: Rapid UI development, ideal for ML prototypes
- **Alternative Considered**: Flask/FastAPI - rejected as overkill for demo

**Features**:
1. **Single Review Prediction**:
   - Text input for title and body
   - Real-time classification with probability scores
   - Visual confidence indicators

2. **Batch Processing**:
   - CSV upload capability
   - Bulk prediction processing
   - Results download with predictions and probabilities

**Caching Strategy**:
- `@st.cache_resource` decorator for model loading
- Prevents repeated model loading on each interaction
- Improves response time significantly

### Workflow Automation

**Review Routing Process** (documented in `REPORT.md`):
1. Review collection (0 minutes)
2. Automatic preprocessing (<1 second)
3. Department classification (<1 second)
4. Sentiment analysis (<1 second)
5. Routing and notification (<1 minute)

**Target Response Times**:
- Negative Housekeeping reviews: 2 hours
- Negative Reception reviews: 1 hour
- Negative F&B reviews: 2 hours
- Positive reviews: 24 hours

## External Dependencies

### Python Libraries

**Core ML Stack**:
- `scikit-learn` - ML models, vectorization, evaluation metrics
- `pandas` - Data manipulation and CSV handling
- `numpy` - Numerical operations

**Visualization**:
- `matplotlib` - Plotting and charts
- `seaborn` - Statistical visualizations

**Web Framework**:
- `streamlit` - Interactive dashboard and web interface

**Python Version**: 3.11+ required

### File System Dependencies

**Model Artifacts** (created during training):
- `models/vectorizer.pkl` - Trained TF-IDF vectorizer
- `models/department_model.pkl` - Department classification model
- `models/sentiment_model.pkl` - Sentiment classification model

**Results and Reports**:
- `results/metrics.pkl` - Training metrics and evaluation results
- `REPORT.md` - Auto-generated system report with workflow documentation

**Dataset**:
- `dataset_recensioni.csv` - Synthetic training data (500 reviews)
- Generated via `generate_dataset.py` script

### No External APIs or Services

This system is **fully self-contained**:
- No database required (uses CSV files)
- No external ML APIs (models trained locally)
- No authentication services
- No cloud dependencies

**Design Philosophy**: Minimize external dependencies for easy deployment and maximum reproducibility in Replit environment.