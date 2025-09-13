# House Price Prediction Model

A machine learning model for predicting California housing prices using **scikit-learn** and Random Forest Regressor.

## 📌 Overview

This project demonstrates a complete machine learning pipeline for predicting California housing prices. It utilizes scikit-learn's built-in California housing dataset and implements data preprocessing, model training, evaluation, and model persistence functionality.

**Key Features:**
- Random Forest Regressor implementation
- Data preprocessing pipeline
- Model evaluation metrics
- Feature importance analysis
- Model saving/loading capabilities

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Required Dependencies
```
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
```

### Installation Steps

**1. Clone the Repository**
```bash
git clone https://github.com/your-username/Predicting-House-Pricing-Model.git
cd Predicting-House-Pricing-Model
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Model**
```bash
python main.py
```

> **Note:** The trained model file (`model.pkl`) is generated locally and not tracked in version control. It will be created automatically when you run the script for the first time.

## 🚀 Usage

### Basic Usage
```bash
python main.py
```

This will:
1. Load and preprocess the California housing dataset
2. Train the Random Forest model
3. Evaluate model performance
4. Display evaluation metrics
5. Save the trained model pipeline

### Project Structure
```
Predicting-House-Pricing-Model/
├── main.py              # Main script to run the model
├── requirements.txt     # Project dependencies
├── pipeline.pkl         # Trained model (generated locally)
├── README.md           # Project documentation
└── .gitignore          # Git ignore file
```

## 📈 Model Performance

The Random Forest Regressor achieves excellent performance on the California housing dataset:

| Metric | Score |
|--------|-------|
| **R² Score** | High correlation between predicted and actual values |
| **Mean Absolute Error (MAE)** | Low average prediction error |
| **Mean Squared Error (MSE)** | Minimal squared error loss |

### Feature Importance
The model analyzes which factors most significantly influence housing prices, providing insights into the California real estate market.

## 📊 Sample Predictions

| Longitude | Latitude | Median Income | Ocean Proximity | Actual Price ($) | Predicted Price ($) |
|-----------|----------|---------------|----------------|-----------------|-------------------|
| -118.39   | 34.12    | 6,447         | <1H OCEAN      | $500,001        | $490,186.71       |
| -120.42   | 34.89    | 5,010         | <1H OCEAN      | $162,500        | $203,646.01       |
| -118.45   | 34.25    | 4,380         | <1H OCEAN      | $204,600        | $196,457.00       |
| -117.07   | 32.77    | 4,350         | NEAR OCEAN     | $184,000        | $209,096.00       |

## 🔧 Technical Details

### Dataset
- **Source:** Scikit-learn's California housing dataset
- **Features:** Location coordinates, median income, housing age, rooms, population, and ocean proximity
- **Target:** Median house value

### Model
- **Algorithm:** Random Forest Regressor
- **Framework:** scikit-learn
- **Pipeline:** Includes preprocessing and model training steps

### Preprocessing
- Feature scaling and normalization
- Categorical variable encoding
- Missing value handling (if applicable)

## 📋 Requirements File
Create a `requirements.txt` file with the following content:
```
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
joblib>=1.1.0
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

If you have any questions or suggestions, please feel free to reach out:

- GitHub: [@your-username](https://github.com/Sanskar1314)
- Email: sanskargupta1314@gmail.com

## ⭐ Acknowledgments

- Scikit-learn team for the excellent machine learning library
- California housing dataset contributors
- Open source community for inspiration and support

---

**Made with ❤️ and Python**
