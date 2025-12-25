# 💧 Water Quality Prediction using Machine Learning

This repository contains a **Machine Learning–based Water Quality Prediction system** developed as part of the **B.Tech final year project** at **BML Munjal University**.  
The project predicts water quality classes using physicochemical parameters collected from major Indian rivers in India.

---

## 📌 Project Overview

Water quality monitoring is essential for public health, environmental protection, and sustainable water resource management.  
This project applies **Supervised Machine Learning models**, particularly **Support Vector Classifier (SVC)**, to predict water quality based on key water parameters.

- **Data Duration:** 2017 – 2021  
- **Rivers Covered:** Ganga, Beas, Brahmaputra, Godavari, Krishna, Satluj  
- **Data Source:** Central Pollution Control Board (CPCB), Government of India  

---

## 🎯 Objectives

- Analyze historical water quality data  
- Perform Exploratory Data Analysis (EDA)  
- Train and compare multiple ML classification models  
- Predict water quality categories  
- Visualize spatial distribution of water quality using maps  
- Study water quality variation during the COVID-19 period  

---

## 🧪 Features Used

The following physicochemical parameters are used as input features:

- **Dissolved Oxygen (DO)**
- **pH**
- **Biochemical Oxygen Demand (BOD)**
- **Nitrate**

---

## 🗂️ Dataset Description

| Year | Rows | Columns |
|----|----|----|
| 2017 | 202 | 9 |
| 2018 | 209 | 9 |
| 2019 | 249 | 9 |
| 2020 | 237 | 9 |
| 2021 | 263 | 9 |

An additional auxiliary dataset of **1,755 rows × 5 columns** is used to improve model training.

---

## 🧠 Machine Learning Models Used

- Logistic Regression  
- Random Forest Classifier  
- Decision Tree Classifier  
- **Support Vector Classifier (SVC)** ✅ *(Best performing)*  

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-score |
|----|----|----|----|----|
| Logistic Regression | 90% | 0.91 | 0.90 | 0.90 |
| Random Forest | 91% | 0.93 | 0.91 | 0.91 |
| Decision Tree | 82% | 0.88 | 0.82 | 0.81 |
| **Support Vector Classifier** | **94%** | **0.95** | **0.94** | **0.95** |

**SVC achieved the highest accuracy and was selected as the final model.**

---

## 🗺️ Visualizations & Analysis

- Correlation heatmaps  
- Water quality class distribution plots  
- Confusion matrices  
- **Interactive Folium maps** for spatial visualization  
- Year-wise water quality prediction (2017–2021)  
- COVID-19 period analysis (2019–2021)  

---

## 🔍 Key Insights

- Improved water quality observed during COVID-19 due to reduced industrial activity  
- Industrial pollution and agricultural runoff significantly affect water quality  
- Machine learning models effectively capture nonlinear environmental patterns  

---

## 🛠️ Technologies Used

- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **Matplotlib, Seaborn**
- **Folium**
- **Jupyter Notebook**

---

## 📚 References

- Central Pollution Control Board (CPCB): https://cpcb.nic.in
- ResearchGate – Water Quality Prediction using Machine Learning

---

## 📚 Future Scope

- Inclusion of additional environmental parameters

- Hyperparameter tuning and ensemble models

- Real-time water quality monitoring

- Integration with IoT and satellite data
---

## 🚀 How to Run the Project

```bash
# Clone the repository
git clone https://github.com/your-username/water-quality-prediction.git

# Navigate to project directory
cd water-quality-prediction

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook

