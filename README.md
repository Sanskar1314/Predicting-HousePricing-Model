Predicting-HousePricing-Model

A basic house price predicting model using scikit-learn.

📌 Overview

This project predicts California housing prices using a Random Forest Regressor. It uses scikit-learn’s built-in dataset and demonstrates data preprocessing, model training, evaluation, and saving/loading pipelines.

🛠️ Requirements

Python 3.x

scikit-learn

pandas

numpy

matplotlib (optional for visualization)

Install dependencies:

pip install -r requirements.txt

▶️ Usage

Clone the repo:

git clone https://github.com/your-username/Predicting-HousePricing-Model.git
cd Predicting-HousePricing-Model


Train and evaluate:

python main.py


The trained model file (pipeline.pkl) is not pushed to GitHub due to size limits. After running main.py, it will be generated locally for making predictions.

📊 Results

Random Forest achieved strong performance (high R², low MAE/MSE).

Feature importance plots help interpret the model.

📑 Example Predictions

Below is a preview of model predictions for a subset of the California Housing dataset:

Longitude	Latitude	Median Income	Total Rooms	Population	Ocean Proximity	Predicted Price ($)
-118.39	34.12	8.28	6447	2184	<1H OCEAN	490,186.71
-120.42	34.89	5.01	2020	855	<1H OCEAN	203,646.01
-118.45	34.25	4.38	1453	808	<1H OCEAN	196,457.00
-118.10	33.91	3.27	1653	1072	<1H OCEAN	173,226.00
-117.07	32.77	4.35	3779	1495	NEAR OCEAN	209,096.00
📜 License

MIT License
