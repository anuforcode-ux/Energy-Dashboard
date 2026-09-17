# Energy-Dashboard
Forecasts household electricity usage and flags unusual spikes (LR, Random Forest, LSTM, Prophet) with a live Streamlit dashboard
⚡ Household Energy Consumption Forecasting & Anomaly Detection

A machine learning project that forecasts household electricity usage and automatically flags unusual consumption spikes (e.g., "AC left running"), presented as an interactive dashboard.

Built as a Project-Based Learning (PBL) project for CS5403 – Machine Learning.

📌 Problem Statement

Households often waste electricity without realizing it — an appliance left on, an AC running longer than needed, unusual late-night usage. This project builds a system that:

Forecasts expected hourly electricity usage using historical patterns
Detects anomalies by comparing actual usage against what the model expected
Generates plain-English alerts explaining likely causes of unusual spikes
Visualizes everything in an interactive dashboard, including a live cost projection
📊 Dataset

UCI Individual Household Electric Power Consumption

Minute-level electricity readings from a single household in Sceaux, France
Time range: December 2006 – November 2010 (~4 years)
~2 million raw rows, resampled to hourly/daily for modeling
Target variable: Global_active_power (kW)
~1.25% missing values, handled via forward/backward fill
🧠 Approach & Methodology
Stage	What happens
1. Data Cleaning	Load raw data, coerce to numeric, fill missing values
2. Resampling	Aggregate minute-level data to hourly (modeling) and daily (EDA / Prophet)
3. EDA	Visualize usage trends by hour, day of week, and month
4. Feature Engineering	12 features: time features, weekend/morning/evening flags, season, lag features (1h/24h/168h), rolling mean, simulated temperature
5. Train/Test Split	Chronological 80/20 split (never shuffled — this is time series data)
6. Modeling	Four models trained and compared (see below)
7. Anomaly Detection	Combines Z-score on residuals + Isolation Forest on raw features
8. Alert Generation	Rule-based plain-English messages for flagged anomalies
9. Dashboard	Streamlit app tying forecasts, anomalies, alerts, and cost projection together
🤖 Models & Results
Model	Type	RMSE (kWh)	MAE (kWh)	Notes
Linear Regression	Baseline	0.503	—	Simple linear relationship between features and usage
Random Forest	Ensemble (100 trees)	0.473	—	~6% improvement over baseline; also gives feature importance
LSTM	Deep learning	(varies by run)	—	Learns directly from 24-hour sequences instead of hand-crafted lags
Prophet	Seasonal forecasting	(daily scale)	—	Yearly/weekly seasonality + holiday calendar + confidence intervals

(Fill in your exact numbers from your latest run — they'll vary slightly run to run.)

Anomaly detection: flagged ~7% of test-set hours as unusual, using either Z-score (>2 std dev from expected residual) or Isolation Forest agreement.

🛠️ Tech Stack
Language: Python
Data & ML: pandas, numpy, scikit-learn, TensorFlow/Keras (LSTM), Prophet
Visualization: matplotlib, seaborn, Plotly
Dashboard: Streamlit
📁 Project Structure
├── household_power_consumption.txt   # dataset (not committed — see Setup)
├── energy_project_full.py            # full ML pipeline: EDA → models → anomaly detection
├── data_pipeline.py                  # shared load/clean/resample/feature-engineering logic
├── anomaly_detector.py               # Z-score + Isolation Forest anomaly detection
├── alert_generator.py                # plain-English alert generation
├── dashboard.py                      # Streamlit dashboard app
├── models/
│   ├── lstm_model.py                 # LSTM training module
│   └── prophet_model.py              # Prophet training module
└── requirements.txt
⚙️ Setup & Installation
bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset from UCI and place household_power_consumption.txt in the project root
#    https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption

# 4. Run the full ML pipeline
python energy_project_full.py

# 5. Launch the dashboard
streamlit run dashboard.py
🚀 Live Demo

(Add your deployed Streamlit Community Cloud link here once deployed, e.g.:) https://your-app-name.streamlit.app

📈 Sample Output

(Add screenshots here — e.g. eda_graphs.png, anomaly_detection.png, final_model_comparison.png, and a screenshot of the dashboard itself)

🔭 Future Work
Replace simulated temperature with a real weather API
Full production-grade anomaly alerting (push notifications, thresholds tuned per household)
Convert into a full-stack app (FastAPI backend + React frontend) instead of the Streamlit prototype
Multi-household support with per-household models
🎓 Course Context

This project was built for CS5403 – Machine Learning, as part of a Project-Based Learning (PBL) review, progressing from an initial baseline (EDA + Linear Regression) through advanced models (Random Forest, LSTM, Prophet), full anomaly detection, and a working dashboard.
