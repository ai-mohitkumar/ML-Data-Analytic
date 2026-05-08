# ML Data Analytic Dashboard

A full-stack Machine Learning Data Analytics platform that enables users to upload CSV files or manually enter data, perform clustering and anomaly detection, and visualize results in an interactive dashboard.

## 🚀 Features
- CSV Upload and Manual Data Entry
- Dynamic Field Creation
- Automatic KMeans Clustering
- ICSO Score Calculation
- Silhouette, Davies-Bouldin, and Calinski-Harabasz Metrics
- Anomaly Detection
- Interactive Scatter Plot Visualization
- Cluster Distribution Analysis
- Recommendation Engine
- Modern Responsive Dashboard

## 🛠 Tech Stack
**Frontend:** React, Recharts, CSS  
**Backend:** FastAPI, Pandas, NumPy, Scikit-learn

## 📂 Project Structure
```text
ML-Data-Analytic/
├── main.py
├── requirements.txt
├── README.md
└── UserBehaviorApp/
    └── frontend/
        ├── src/
        │   ├── DashboardNew.js
        │   └── DashboardStyles.css
        └── package.json
⚙️ Installation
Backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn main:app --reload
Frontend
cd UserBehaviorApp/frontend
npm install
npm start
▶️ Usage
Open http://localhost:3000
Upload a CSV file or manually enter data
Click Run ML Analysis
Explore clusters, metrics, anomalies, and recommendations
📊 Supported Data

Numeric CSV columns such as:

age
annual_income
spending_score
purchase_count
📈 Output Metrics
Total Users
Average Spending
Average Orders
ICSO Score
Silhouette Score
Cluster Distribution
Anomaly Percentage
👨‍💻 Author->Mohit Kumar

📄 License

MIT License