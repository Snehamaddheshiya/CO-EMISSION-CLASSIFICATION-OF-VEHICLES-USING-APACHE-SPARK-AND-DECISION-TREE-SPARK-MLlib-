# 🚗 Vehicle CO2 Emission Classification using Apache Spark (PySpark & MLlib)

An end-to-end distributed data processing and machine learning pipeline built with **Apache Spark (PySpark)** and **Spark MLlib** to classify vehicles into **Low**, **Medium**, or **High** CO2 emission categories, paired with an interactive **Microsoft Excel Dashboard**.

## 📌 Project Overview
This project demonstrates a complete big-data machine learning workflow—from data ingestion and cleaning to categorical encoding, feature vectorization, model training, evaluation, and visual reporting.

* **Dataset:** 2,000 vehicle records across 11 columns (`1,938` complete records retained after removing 62 incomplete rows).
* **Target Variable:** `Emission_Class` (`Low`, `Medium`, `High`).
* **Input Features (8):** `Vehicle_Type`, `Fuel_Type`, `Engine_Size_L`, `Cylinders`, `Transmission`, `Vehicle_Age_Years`, `Fuel_Consumption_L_per_100km`, and `Region` *(raw `CO2_Emissions_g_per_km` is excluded to prevent target leakage)*.

## 🛠️ Tech Stack
* **Language:** Python
* **Framework:** Apache Spark (PySpark, Spark SQL, Spark MLlib)
* **Algorithm:** Decision Tree Classifier (`DecisionTreeClassifier`)
* **Visualization:** Microsoft Excel Dashboard

## ⚙️ Workflow
1. **Data Ingestion & Inspection:** Loaded CSV into a Spark DataFrame using `spark.read.csv()` with schema inference.
2. **Data Cleaning:** Removed 62 records with missing values, leaving 1,938 complete records.
3. **Feature Engineering:**
   * Encoded categorical columns and the target label using `StringIndexer`.
   * Combined all 8 input features into a single `features` vector using `VectorAssembler`.
4. **Train/Test Split:** Divided the dataset in a **70:30 ratio** (`1,324` training / `614` testing records).
5. **Model Training & Prediction:** Fitted a Spark MLlib `DecisionTreeClassifier` and generated predictions on the test set.
6. **Evaluation & Dashboarding:** Evaluated metrics via `MulticlassClassificationEvaluator`, analyzed feature importance and the confusion matrix, and visualized KPIs in Excel.

## 📊 Results & Key Insights
| Metric | Score |
| :--- | :--- |
| **Accuracy** | **88.93%** (546 / 614 correct) |
| **Weighted Precision** | **88.84%** |
| **Weighted Recall** | **88.93%** |
| **F1-Score** | **88.71%** |

* **Top Feature Drivers:** `Fuel_Consumption_L_per_100km` accounted for **90.56%** of the model's decision importance, followed by `Fuel_Type` at **7.03%**.
* **Zero Extreme Errors:** The confusion matrix confirmed that no `Low` emission vehicle was misclassified as `High` (or vice versa)—all misclassifications occurred along the `Medium` class boundary.

## 🚀 Future Scope
* Benchmark against ensemble models like **Random Forest** and **Gradient-Boosted Trees**.
* Implement cross-validation, hyperparameter tuning, and missing-value imputation.
* Extend with **Apache Spark GraphX** to analyze relationships between vehicle types, fuel types, and regions.
