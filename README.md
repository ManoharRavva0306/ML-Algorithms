# ML-Algorithms: Machine Learning Capstone Project

This repository contains an end-to-end Machine Learning pipeline built for the **23CSE301 Machine Learning Capstone Project**.

The project focuses on building, evaluating, and comparing machine learning models across three tracks:
1. **Regression Track** (BigMart Sales Prediction)
2. **Classification Track**
3. **Clustering Track**

---

## 📁 Repository Structure

```
ML-Algorithms/
├── README.md                  # Project overview, results, setup instructions
├── requirements.txt           # Python dependencies with version specifications
├── data/
│   ├── download_data.py       # Script to load/generate BigMart Sales dataset
│   └── Train.csv              # BigMart Sales raw dataset
├── notebooks/
│   ├── regression.ipynb       # Regression pipeline (EDA, preprocessing, first 5 algorithms)
│   ├── classification.ipynb   # Classification pipeline (to be added)
│   └── clustering.ipynb       # Clustering pipeline (to be added)
├── models/
│   └── regression_model.pkl   # Serialized model pipelines (.pkl via joblib)
└── app/
    └── app.py                 # Streamlit web GUI / deployment code (Bonus)
```

---

## 📊 Dataset: BigMart Sales Prediction

- **Source**: [BigMart Sales Prediction on Kaggle](https://www.kaggle.com/datasets/elahehkazemian/big-mart-sales-prediction)
- **Target Variable**: `Item_Outlet_Sales` (Continuous Sales Prediction)
- **Key Features**: `Item_Identifier`, `Item_Weight`, `Item_Fat_Content`, `Item_Visibility`, `Item_Type`, `Item_MRP`, `Outlet_Identifier`, `Outlet_Establishment_Year`, `Outlet_Size`, `Outlet_Location_Type`, `Outlet_Type`.

---

## 🚀 Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/ML-Algorithms.git
   cd ML-Algorithms
   ```

2. **Create a virtual environment & install dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Download / Verify Dataset**:
   ```bash
   python data/download_data.py
   ```

4. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook notebooks/regression.ipynb
   ```

---

## 📈 Model Performance Summary (Regression Track)

| # | Algorithm | $R^2$ Score | RMSE | MAE | 5-Fold CV $R^2$ |
|---|-----------|-----------|------|-----|----------------|
| 1 | Linear Regression | TBD | TBD | TBD | TBD |
| 2 | Ridge Regression | TBD | TBD | TBD | TBD |
| 3 | Lasso Regression | TBD | TBD | TBD | TBD |
| 4 | ElasticNet Regression | TBD | TBD | TBD | TBD |
| 5 | Polynomial Regression | TBD | TBD | TBD | TBD |

---

## 🛠 Tech Stack
- **Language**: Python 3.10+
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn, Joblib
- **Visualization**: Matplotlib, Seaborn
- **UI Framework**: Streamlit
