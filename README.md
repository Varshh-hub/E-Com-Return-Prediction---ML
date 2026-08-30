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

## Key Learning

One important observation from this project is that **accuracy should not be considered alone when working with an imbalanced classification dataset**.

For an e-commerce return prediction system, correctly detecting returned orders can be more valuable than simply maximizing overall accuracy.

This project therefore focuses on creating a realistic prediction pipeline while avoiding target leakage and evaluating the model using multiple classification metrics.

---

## Author

**Varsha A**

Machine Learning / Data Science Project

---
