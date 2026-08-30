# E-Commerce Product Return Prediction

A machine learning project designed to predict whether an e-commerce order is likely to be **Returned** or **Not Returned** using customer, product, payment, shipping, and order-related information.

The goal of this project is to identify potentially return-prone orders before the return occurs, helping e-commerce businesses better understand return patterns and reduce operational costs.

## Project Overview

Product returns are a major challenge for e-commerce businesses because they can increase shipping costs, packaging waste, reverse-logistics expenses, and overall operational losses.

This project applies supervised machine learning to predict the `Return_Status` of an order using information available around the time the order is placed.

The target classes are:

* `Returned`
* `Not Returned`

The dataset contains **5,000 orders**.

### Target Distribution

| Return Status | Records | Percentage |
| ------------- | ------: | ---------: |
| Not Returned  |   3,550 |        71% |
| Returned      |   1,450 |        29% |

Because the dataset is imbalanced, accuracy alone is not sufficient to evaluate the model. Precision, recall, and F1-score are also considered.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Joblib
* Jupyter Notebook

---


### Features Used for Prediction

```text
Product_Category
Product_Price
Order_Quantity
Discount_Applied
Shipping_Method
Payment_Method
User_Age
User_Gender
User_Location
Order_Month
Order_DayOfWeek
Is_Weekend
```

Date-related features are extracted from `Order_Date`.

---

## Train-Test Split

The dataset is divided into:

```text
80% Training Data
20% Testing Data
```

The split uses:

```python
random_state=42
stratify=y
```

Stratification ensures that the proportion of returned and non-returned orders remains similar in both datasets.

---

## Machine Learning Models

Two classification algorithms were initially tested.

### Random Forest Classifier

Random Forest was used with class balancing to reduce the impact of the target-class imbalance.

The model achieved approximately:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 67.60% |
| Precision | 38.19% |
| Recall    | 18.97% |
| F1 Score  | 25.35% |

The model performs well at identifying `Not Returned` orders but has relatively low recall for the `Returned` class.

### Logistic Regression

A balanced Logistic Regression model was also evaluated.

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 58.00% |
| Precision | 35.49% |
| Recall    | 54.83% |
| F1 Score  | 43.09% |

Although Logistic Regression has lower overall accuracy, it detects a higher proportion of actual returned orders.

---

## Model Evaluation

Since approximately **71% of orders are Not Returned**, a model could achieve high accuracy by mostly predicting the majority class.

Therefore, this project evaluates models using:

```text
Accuracy
Precision
Recall
F1 Score
Classification Report
Confusion Matrix
```

For this problem, **recall and F1-score for the `Returned` class are particularly important**, because the objective is to detect orders that are likely to be returned.

---

## Prediction Output

The model supports probability-based predictions rather than only returning a class label.

Example:

```text
RETURN PREDICTION

Prediction: Returned

Prediction Probability:

Not Returned: 34.35%
Returned: 65.65%
```

This allows the system to show the confidence associated with each prediction.

For example, instead of only saying:

```text
Returned
```

the application can report:

```text
65.65% probability of being returned
```

---

## Machine Learning Pipeline

The project uses a Scikit-learn `Pipeline` to combine preprocessing and model training.

```text
Raw Order Data
      ↓
Feature Engineering
      ↓
Numerical Preprocessing
      ↓
Categorical Encoding
      ↓
Random Forest Classifier
      ↓
Return Prediction
      ↓
Return Probability
```

Using a pipeline ensures that the same transformations applied during training are also applied when predicting new orders.

---

## Project Structure

```text
ecommerce-return-prediction/
│
├── data.csv
├── model.ipynb
├── return_prediction_model.pkl
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
```

Move into the project folder:

```bash
cd ecommerce-return-prediction
```

Install the required packages:

```bash
pip install pandas numpy matplotlib scikit-learn joblib jupyter
```

---

## Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
model.ipynb
```

Run the notebook cells sequentially to:

1. Load the dataset
2. Explore the data
3. Engineer features
4. Preprocess the data
5. Train the model
6. Evaluate model performance
7. Predict new orders

---

## Saving the Model

The complete pipeline can be saved using Joblib:

```python
import joblib

joblib.dump(
    model,
    "return_prediction_model.pkl"
)
```

The model can later be loaded using:

```python
model = joblib.load(
    "return_prediction_model.pkl"
)
```

Saving the complete pipeline means the preprocessing steps do not need to be recreated separately during deployment.

---

## Limitations

The current dataset has limited predictive information about why customers return products.

The target is also imbalanced, with substantially more `Not Returned` orders than `Returned` orders.

Because of this, simply increasing model complexity may not significantly improve real-world performance.

A higher accuracy score does not automatically indicate a better model if it fails to identify returned orders.

---

## Future Improvements

Model performance could potentially be improved by collecting additional features such as:

* Customer's previous return rate
* Number of previous purchases
* Product historical return rate
* Seller rating
* Product rating
* Customer purchase history
* Delivery distance
* Estimated delivery delay
* Product size or fit information
* Product review information
* Customer-product interaction history

Additional improvements could include:

* Hyperparameter optimization
* Cross-validation
* XGBoost or LightGBM
* Threshold optimization
* SMOTE or other imbalance-handling techniques
* Feature selection
* Model explainability using SHAP
* Deployment using Streamlit or Flask

---

## Key Learning

One important observation from this project is that **accuracy should not be considered alone when working with an imbalanced classification dataset**.

For an e-commerce return prediction system, correctly detecting returned orders can be more valuable than simply maximizing overall accuracy.

This project therefore focuses on creating a realistic prediction pipeline while avoiding target leakage and evaluating the model using multiple classification metrics.

---

## Author

**Varsha A**

Machine Learning / Data Science Project

---

## Disclaimer

This project is developed for educational and machine learning experimentation purposes. Model predictions depend heavily on the quality and predictive strength of the available dataset and should not be treated as production-level business decisions without further validation.
