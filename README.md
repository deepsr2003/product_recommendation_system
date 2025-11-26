# Product Recommendation System

A machine learning-based product recommendation system that uses RFM (Recency, Frequency, Monetary) analysis to predict high-value customers and provide real-time recommendations.

## 🚀 Features

- **RFM Analysis**: Engineered RFM features for 4,339 unique customers using Pandas
- **CatBoost Classification**: Trained CatBoostClassifier to predict high-value customer status
- **100% Accuracy**: Achieved perfect accuracy on 868-sample test set (20% of data)
- **Feature Importance**: Identified Monetary and Recency as most predictive features
- **Interactive Dashboard**: Real-time "Recommend / Do Not Recommend" decisions with ipywidgets
- **Comprehensive Visualizations**: Seaborn-based analysis and feature importance plots

## 📊 Model Performance

- **Dataset**: 4,339 customers, 48,006 transactions
- **Test Accuracy**: 100% (868 samples)
- **Most Predictive Features**:
  - Monetary: 98.6%
  - Frequency Score: 0.5%
  - Recency: 0.4%
- **High-Value Customer Split**: 50% (based on median monetary value)

## 🛠️ Technology Stack

- **Pandas**: Data manipulation and RFM feature engineering
- **Scikit-learn**: Model training and evaluation
- **CatBoost**: Gradient boosting classifier
- **Matplotlib/Seaborn**: Data visualization
- **ipywidgets**: Interactive dashboard components
- **Jupyter**: Notebook environment

## 📁 Project Structure

```
product_recommendation_system/
├── data/
│   ├── transactions.csv      # Raw transaction data
│   ├── customers.csv         # Customer profiles
│   └── rfm_features.csv      # Engineered RFM features
├── src/
│   ├── data_generator.py     # Synthetic data generation
│   ├── rfm_analysis.py       # RFM feature engineering
│   ├── model_training.py     # CatBoost model training
│   ├── visualization.py      # Analysis plots
│   └── dashboard.py          # Interactive dashboard
├── models/
│   ├── catboost_model.cbm    # Trained CatBoost model
│   └── catboost_model.pkl    # Serialized model
├── outputs/
│   ├── rfm_analysis_dashboard.png
│   ├── feature_analysis.png
│   ├── test_results.csv
│   └── feature_importance.csv
├── notebooks/
│   └── interactive_dashboard.ipynb
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
cd product_recommendation_system
pip install -r requirements.txt
```

### 2. Data Generation

```bash
python src/data_generator.py
```

### 3. RFM Analysis

```bash
python src/rfm_analysis.py
```

### 4. Model Training

```bash
python src/model_training.py
```

### 5. Visualization

```bash
python src/visualization.py
```

### 6. Interactive Dashboard

```bash
jupyter notebook notebooks/interactive_dashboard.ipynb
```

## 📈 RFM Analysis

### RFM Features
- **Recency**: Days since last purchase (lower = better)
- **Frequency**: Number of transactions (higher = better)
- **Monetary**: Total spending amount (higher = better)

### Customer Segments
- **Champions**: High recency, frequency, and monetary scores
- **Loyal Customers**: High recency and frequency
- **Potential Loyalists**: High recency and monetary
- **New Customers**: High recency, low frequency
- **At Risk**: Low recency, high frequency
- **Lost**: Low recency and frequency

## 🎯 Interactive Dashboard

The interactive dashboard allows you to:

1. **Adjust RFM Scores**: Use sliders (1-5 scale) to simulate customer profiles
2. **Input Raw Values**: Enter specific recency, frequency, and monetary values
3. **Get Real-time Predictions**: Instant "Recommend / Do Not Recommend" decisions
4. **View Customer Analysis**: Visual comparison with dataset
5. **See Confidence Scores**: Prediction probabilities and confidence levels

## 📊 Key Results

### Model Performance
```
Training set: 3,471 samples
Test set: 868 samples
High-value customers: 50% (balanced)
Model Accuracy: 100.00%
```

### Feature Importance
```
Monetary:         98.64%
Frequency Score:   0.52%
Recency:           0.38%
Recency Score:     0.25%
Frequency:         0.15%
Monetary Score:    0.06%
```

### Customer Distribution
```
Lost:                   1,075 (24.8%)
Champions:                998 (23.0%)
At Risk:                  661 (15.2%)
Others:                   495 (11.4%)
Loyal Customers:          400 (9.2%)
Potential Loyalists:      358 (8.3%)
New Customers:            352 (8.1%)
```

## 🔬 Methodology

1. **Data Generation**: Created synthetic transaction data for 4,339 customers
2. **RFM Engineering**: Calculated recency, frequency, and monetary features
3. **Feature Scoring**: Applied 1-5 scoring system for each RFM dimension
4. **Model Training**: Used CatBoostClassifier with stratified 80/20 split
5. **Evaluation**: Achieved 100% accuracy on test set
6. **Visualization**: Created comprehensive analysis plots
7. **Dashboard**: Built interactive simulation tool

## 📝 Usage Examples

### Python API Usage

```python
import sys
sys.path.append('src')
from model_training import load_model
from rfm_analysis import calculate_rfm_features

# Load trained model
model, feature_columns = load_model()

# Calculate RFM for new customer
new_customer_data = pd.DataFrame({
    'customer_id': [9999],
    'transaction_date': ['2024-11-20'],
    'amount': [250.00]
})

rfm_features = calculate_rfm_features(new_customer_data)
prediction = model.predict(rfm_features[feature_columns])
```

### Dashboard Usage

1. Open the interactive dashboard notebook
2. Adjust RFM scores or input raw values
3. Click "Predict Recommendation"
4. Review results and visualizations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🔮 Future Enhancements

- [ ] Real-time data integration
- [ ] Additional ML models comparison
- [ ] A/B testing framework
- [ ] Customer lifetime value prediction
- [ ] Product-level recommendations
- [ ] Time-series analysis for trends

## 📞 Contact

For questions or suggestions about this product recommendation system, please open an issue in the repository.

---

**Note**: This system uses synthetic data for demonstration purposes. In production, ensure proper data privacy and security measures are in place.