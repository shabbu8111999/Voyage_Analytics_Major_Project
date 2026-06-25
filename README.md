# Voyage Analytics — Integrating MLOps in Travel

This is an MLOps project where I built and deployed machine learning models on real travel data.

The project covers the full ML pipeline — from raw data to a working web application with 3 different ML features running inside Docker.

---

## Problem Statement

- Analyze travel data from users, flights, and hotels datasets.
- Identify travel patterns and customer preferences.
- Build machine learning models to predict travel-related decisions.
- Generate personalized travel recommendations for users.
- Create an end-to-end MLOps pipeline for model deployment and monitoring.

---

## What This Project Does

I have 3 datasets from a travel booking platform:

| File | Rows | What it contains |
|---|---|---|
| flights.csv | 271,888 | Flight booking records — price, distance, agency, flight type |
| users.csv | 1,340 | Traveller profiles — age, gender, company |
| hotels.csv | 40,552 | Hotel stay records — city, hotel name, price per night, days |

Using these 3 datasets I built 3 machine learning features and served them through one Flask web application.

---

## ML Features Built

### 1. Flight Price Prediction (Regression)
User fills in flight details and the model predicts the price in BRL.
Model used: Random Forest Regressor
Result: R² = 0.9889, MAE = 9.61 BRL

### 2. Flight Type Classifier (Classification)
User enters their booking history and the model predicts if they prefer economic, premium or firstClass flights.
Model used: Random Forest Classifier
Result: 89.51% accuracy

### 3. Hotel Recommendation (Recommendation System)
User enters budget per night and number of days. The system returns the top 3 most popular hotels within that budget with total estimated cost.
Algorithm: Budget filtering + popularity ranking

---

## My Work — Step by Step

### Step 1 — Data Loading
File: `notebooks/data_loading.ipynb`

Loaded all 3 CSV files and checked their shape, column names and data types.
Understood what each column means before doing anything else.

### Step 2 — Data Cleaning
File: `notebooks/data_cleaning.ipynb`

Merged flights and users into one table using the userCode column.
Checked for missing values — none found.
Checked for duplicate rows — none found.
Fixed the date column from text to datetime format.
Renamed the from column to origin to avoid Python keyword issues.
Dropped travelCode and name columns as they are not useful for prediction.
Checked for outliers using IQR method on price, time and distance — none found.
Saved cleaned data to data/processed/flights_merged.csv.

### Step 3 — Univariate EDA
File: `notebooks/eda_univariate.ipynb`

Analysed one column at a time using histograms and bar charts.
Found that flight price ranges from 301 to 1754 BRL with most flights around 900.
Found all categorical columns are well balanced across their categories.

### Step 4 — Bivariate EDA
File: `notebooks/eda_bivariate.ipynb`

Analysed two columns at a time using all 3 datasets.
Used 6 different chart types — box plot, scatter, line graph, bar chart, pie chart.
Key finding: first class is most expensive and price increases with distance.

### Step 5 — Multivariate EDA
File: `notebooks/eda_multivariate.ipynb`

Analysed 3 or more columns together using all 3 datasets.
Used 7 charts — correlation heatmap, scatter with colour, grouped bar, pair plot, stacked bar, multi-line, pivot heatmap.
Key finding: price and distance have the strongest correlation.

### Step 6 — Regression Model
File: `notebooks/model_training.ipynb`

Target variable is price — a continuous number so this is a regression problem.
Dropped columns not useful for training: userCode, origin, destination.
Extracted date features: month, day of week, is weekend, quarter.
Label encoded 4 text columns: flightType, agency, gender, company.
Split data 80% training and 20% testing.
Scaled numeric features using StandardScaler.
Trained Linear Regression as a baseline — R² 0.4867.
Trained Random Forest as the main model — R² 0.9889.
Checked feature importance — distance and flightType ranked highest.
Saved 3 pkl files: model.pkl, label_encoders.pkl, scaler.pkl.

### Step 7 — Classification Model
File: `notebooks/gender_classification.ipynb`

Initially attempted gender classification but discovered gender has no correlation with travel behaviour in this synthetic dataset — all genders book identically.
Pivoted to a more meaningful task — predicting a traveller's preferred flight class based on their booking history.
Features used: avg price paid, avg distance, avg time, total flights, max price, min price, age, company.
Trained Random Forest Classifier — 89.51% accuracy.
Saved 3 pkl files: classifier_model.pkl, classifier_encoders.pkl, classifier_features.pkl.

### Step 8 — Hotel Recommendation
File: `notebooks/hotel_recommendation.ipynb`

Built a catalog from hotels.csv showing price per night and total bookings for each hotel.
Wrote a budget matching algorithm — filters hotels within user budget and ranks by most popular.
Tested with 3 different budgets — all returned correct results.
Saved hotel_catalog.csv to data/processed/ for the API to use.

### Step 9 — Flask REST API
File: `api/app.py`

Built one web application with 3 tabs using Flask, HTML and CSS.
Loads all 7 pkl files and the hotel catalog once when the server starts.
Tab 1 — Flight Price Predictor — POST /predict
Tab 2 — Flight Type Classifier — POST /classify
Tab 3 — Hotel Recommendation — POST /recommend
Also has GET /health to confirm the server is running.

### Step 10 — Docker
Files: `Dockerfile`, `docker-compose.yml`

Containerized the entire Flask application using Docker.
Anyone can run the full project with one command — docker-compose up.
No need to install Python or any packages manually on their machine.

---

## Tech Stack

| Tool | Used For |
|---|---|
| Python 3.11 | Main language |
| pandas | Data loading, cleaning, merging |
| numpy | Math operations |
| matplotlib | Plotting charts |
| seaborn | Advanced visualisations |
| scikit-learn | Label encoding, scaling, model training |
| joblib | Saving and loading pkl files |
| Flask | REST API and web UI |
| Jupyter Lab | Writing and running notebooks |
| Docker | Containerization |
| uv | Package management |

---

## Folder Structure

```
Voyage_Project/
│
├── data/
│   ├── raw/
│   │   ├── flights.csv
│   │   ├── users.csv
│   │   └── hotels.csv
│   └── processed/
│       ├── flights_merged.csv
│       └── hotel_catalog.csv
│
├── notebooks/
│   ├── data_loading.ipynb
│   ├── data_cleaning.ipynb
│   ├── eda_univariate.ipynb
│   ├── eda_bivariate.ipynb
│   ├── eda_multivariate.ipynb
│   ├── model_training.ipynb
│   ├── gender_classification.ipynb
│   └── hotel_recommendation.ipynb
│
├── models/
│   └── saved/
│       ├── model.pkl
│       ├── label_encoders.pkl
│       ├── scaler.pkl
│       ├── classifier_model.pkl
│       ├── classifier_encoders.pkl
│       └── classifier_features.pkl
│
├── api/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── style.css
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## How to Set Up

```bash
# 1. Clone the project
git clone <your-repo-url>
cd Voyage_Project

# 2. Install dependencies
uv pip install -r requirements.txt

# 3. Add the 3 CSV files to data/raw/
#    flights.csv, users.csv, hotels.csv
```

---

## How to Run

### Option 1 — Run with Python directly

Step 1 — Open Jupyter Lab and run all notebooks in order from data_loading to hotel_recommendation.
After all notebooks are done, the models/saved/ folder will have all pkl files.

```bash
uv run jupyter lab
```

Step 2 — Start the Flask API

```bash
uv run python api/app.py
```

Step 3 — Open browser

```
http://localhost:5000
```

### Option 2 — Run with Docker

Make sure Docker Desktop is running first, then:

```bash
docker-compose up --build
```

Open browser at `http://localhost:5000` — same UI, now running inside a container.

To stop:
```bash
docker-compose down
```

---

## API Endpoints

### GET /health
Confirms the server is running.

Response:
```json
{ "status": "ok", "message": "API is running" }
```

### POST /predict — Flight Price

Sample request:
```json
{
    "flightType"  : "firstClass",
    "agency"      : "FlyingDrops",
    "gender"      : "male",
    "company"     : "4You",
    "time"        : 1.76,
    "distance"    : 676.53,
    "age"         : 21,
    "month"       : 9,
    "day_of_week" : 3,
    "is_weekend"  : 0,
    "quarter"     : 3
}
```

Response:
```json
{ "predicted_price": 1434.52, "currency": "BRL", "status": "success" }
```

### POST /classify — Flight Type Preference

Sample request:
```json
{
    "avg_price"    : 950,
    "avg_distance" : 550,
    "avg_time"     : 1.5,
    "total_flights": 50,
    "max_price"    : 1500,
    "min_price"    : 400,
    "age"          : 35,
    "company"      : "4You"
}
```

Response:
```json
{ "predicted_flight_type": "firstClass", "status": "success" }
```

### POST /recommend — Hotel Recommendation

Sample request:
```json
{
    "budget_per_night" : 200,
    "num_days"         : 3
}
```

Response:
```json
{
    "recommendations": [
        { "name": "Hotel CB", "place": "Rio de Janeiro (RJ)", "price_per_night": 165.99, "estimated_total": 497.97, "total_bookings": 5029 },
        { "name": "Hotel AF", "place": "Sao Paulo (SP)",      "price_per_night": 139.10, "estimated_total": 417.30, "total_bookings": 4828 },
        { "name": "Hotel BW", "place": "Campo Grande (MS)",   "price_per_night": 60.39,  "estimated_total": 181.17, "total_bookings": 4333 }
    ],
    "status": "success"
}
```

---

## Model Results

| Model | Type | Accuracy |
|---|---|---|
| Random Forest Regressor | Regression | R² = 0.9889, MAE = 9.61 BRL |
| Random Forest Classifier | Classification | 89.51% accuracy |
| Hotel Recommendation | Budget Matching | Returns top 3 within budget |
