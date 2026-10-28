
# ============================================================
# MARKET CRISIS ML PROJECT
# تحليل أزمة السوق باستخدام Machine Learning
# ملاحظة: البيانات المولدة هنا تجريبية وليست بيانات سوق حقيقية.
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score
)

# ============================================================
# 1. إنشاء بيانات تجريبية أكبر
# ============================================================

np.random.seed(42)

NUMBER_OF_DAYS = 365

goods = {
    "طحين": 18,
    "أرز": 12,
    "زيت": 20,
    "سكر": 10,
    "حليب": 8,
    "وقود": 25
}

dates = pd.date_range(
    start="2025-01-01",
    periods=NUMBER_OF_DAYS,
    freq="D"
)

rows = []
usd_rate = 3.65

print("=" * 65)
print("MARKET CRISIS ANALYSIS")
print("=" * 65)

for date in dates:

    usd_rate += np.random.normal(0, 0.015)
    usd_rate = max(usd_rate, 3.0)

    crisis_level = np.random.choice(
        [0, 1, 2, 3],
        p=[0.15, 0.30, 0.35, 0.20]
    )

    border_probability = max(
        0.15,
        0.85 - crisis_level * 0.20
    )

    border_open = int(
        np.random.random() < border_probability
    )

    for item, base_price in goods.items():

        price_noise = np.random.normal(0, 0.06)

        price = base_price * (
            1
            + crisis_level * 0.12
            + (1 - border_open) * 0.18
            + (usd_rate - 3.65) * 0.45
            + price_noise
        )

        price = max(price, 0.5)

        # تصنيف التوفر بناءً على ظروف السوق مع ضوضاء عشوائية.
        shortage_score = (
            crisis_level
            + (1 - border_open) * 1.2
            + np.random.normal(0, 0.55)
        )

        if shortage_score < 1.0:
            availability = "متوفر"
        elif shortage_score < 2.5:
            availability = "محدود"
        else:
            availability = "ناقص"

        rows.append({
            "date": date,
            "item": item,
            "price_shekel": round(price, 2),
            "usd_rate": round(usd_rate, 4),
            "border_open": border_open,
            "crisis_level": crisis_level,
            "availability": availability
        })

df = pd.DataFrame(rows)
df = df.sort_values(["date", "item"]).reset_index(drop=True)

availability_map = {
    "متوفر": 0,
    "محدود": 1,
    "ناقص": 2
}

df["avail_num"] = df["availability"].map(availability_map)

print("\n1. DATASET")
print("Total records:", len(df))
print("Number of goods:", df["item"].nunique())
print("Number of days:", df["date"].nunique())

print("\nSample:")
print(df.head(10).to_string(index=False))

print("\nPrice statistics:")
print(df["price_shekel"].describe().round(2))

print("\nMissing values:")
print(df.isnull().sum())

print("\nAvailability distribution:")
print(df["availability"].value_counts())

df.to_csv(
    "market_data.csv",
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 2. تحليل البيانات والرسومات
# ============================================================

print("\n" + "=" * 65)
print("2. EXPLORATORY DATA ANALYSIS")
print("=" * 65)

print("\nAverage price per good:")
print(
    df.groupby("item")["price_shekel"]
    .mean()
    .sort_values(ascending=False)
    .round(2)
)

daily_prices = df.groupby("date")["price_shekel"].mean()

plt.figure(figsize=(12, 5))
plt.plot(daily_prices.index, daily_prices.values)
plt.title("Average Market Price Over Time")
plt.xlabel("Date")
plt.ylabel("Average Price (Shekel)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("market_price_trend.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 5))
(
    df.groupby("item")["price_shekel"]
    .mean()
    .sort_values(ascending=False)
    .plot(kind="bar")
)
plt.title("Average Price by Item")
plt.xlabel("Item")
plt.ylabel("Average Price (Shekel)")
plt.tight_layout()
plt.savefig("average_prices_by_item.png", dpi=150)
plt.close()

# ============================================================
# 3. KMEANS CLUSTERING
# ============================================================

print("\n" + "=" * 65)
print("3. KMEANS MARKET CLUSTERING")
print("=" * 65)

cluster_features = [
    "price_shekel",
    "usd_rate",
    "border_open",
    "crisis_level",
    "avail_num"
]

cluster_scaler = StandardScaler()
X_cluster = cluster_scaler.fit_transform(
    df[cluster_features]
)

inertias = []

for k in range(1, 9):
    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    km.fit(X_cluster)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 9), inertias, marker="o")
plt.title("KMeans Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.tight_layout()
plt.savefig("kmeans_elbow.png", dpi=150)
plt.close()

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["market_cluster"] = kmeans.fit_predict(X_cluster)

silhouette = silhouette_score(
    X_cluster,
    df["market_cluster"]
)

print("Silhouette Score:", round(silhouette, 4))

print("\nCluster statistics:")
print(
    df.groupby("market_cluster")[
        [
            "price_shekel",
            "crisis_level",
            "border_open",
            "avail_num"
        ]
    ].mean().round(2)
)

df.to_csv(
    "market_data_with_clusters.csv",
    index=False,
    encoding="utf-8-sig"
)

plt.figure(figsize=(9, 6))

for cluster_id in sorted(df["market_cluster"].unique()):
    subset = df[df["market_cluster"] == cluster_id]

    plt.scatter(
        subset["crisis_level"],
        subset["price_shekel"],
        label=f"Cluster {cluster_id}",
        alpha=0.6
    )

plt.title("Market Clusters")
plt.xlabel("Crisis Level")
plt.ylabel("Price (Shekel)")
plt.legend()
plt.tight_layout()
plt.savefig("market_clusters.png", dpi=150)
plt.close()

# ============================================================
# 4. ANOMALY DETECTION
# ============================================================

print("\n" + "=" * 65)
print("4. ANOMALY DETECTION")
print("=" * 65)

anomaly_features = [
    "price_shekel",
    "usd_rate",
    "crisis_level",
    "border_open"
]

anomaly_scaler = StandardScaler()

X_anomaly = anomaly_scaler.fit_transform(
    df[anomaly_features]
)

center = X_anomaly.mean(axis=0)

distances = np.sqrt(
    np.sum((X_anomaly - center) ** 2, axis=1)
)

threshold = np.percentile(distances, 95)

df["anomaly_score"] = distances

df["anomaly_label"] = (
    df["anomaly_score"] > threshold
).astype(int)

anomalies = df[
    df["anomaly_label"] == 1
].copy()

print("Threshold:", round(threshold, 4))
print("Detected anomalies:", len(anomalies))
print(
    "Anomaly percentage:",
    round(df["anomaly_label"].mean() * 100, 2),
    "%"
)

print("\nAnomaly sample:")
print(
    anomalies[
        [
            "date",
            "item",
            "price_shekel",
            "crisis_level",
            "anomaly_score"
        ]
    ].head(10).to_string(index=False)
)

anomalies.to_csv(
    "market_anomalies.csv",
    index=False,
    encoding="utf-8-sig"
)

plt.figure(figsize=(10, 5))
plt.plot(df["anomaly_score"].values, label="Anomaly Score")
plt.axhline(
    threshold,
    linestyle="--",
    label="95th Percentile Threshold"
)
plt.title("Market Anomaly Detection")
plt.xlabel("Record")
plt.ylabel("Anomaly Score")
plt.legend()
plt.tight_layout()
plt.savefig("market_anomalies.png", dpi=150)
plt.close()

# ============================================================
# 5. PRICE FORECASTING
# ============================================================

print("\n" + "=" * 65)
print("5. PRICE FORECASTING")
print("=" * 65)

forecast_df = df.sort_values(
    ["item", "date"]
).copy()

forecast_df["previous_price"] = (
    forecast_df.groupby("item")["price_shekel"].shift(1)
)

forecast_df["previous_usd_rate"] = (
    forecast_df.groupby("item")["usd_rate"].shift(1)
)

forecast_df = forecast_df.dropna(
    subset=["previous_price", "previous_usd_rate"]
).copy()

item_dummies = pd.get_dummies(
    forecast_df["item"],
    prefix="item",
    dtype=int
)

forecast_df = pd.concat(
    [forecast_df, item_dummies],
    axis=1
)

forecast_features = [
    "previous_price",
    "previous_usd_rate",
    "border_open",
    "crisis_level"
] + list(item_dummies.columns)

X_forecast = forecast_df[forecast_features]
y_forecast = forecast_df["price_shekel"]

unique_dates = np.sort(
    forecast_df["date"].unique()
)

split_index = int(len(unique_dates) * 0.8)
split_index = max(
    1,
    min(split_index, len(unique_dates) - 1)
)

split_date = unique_dates[split_index]

train_mask = forecast_df["date"] < split_date
test_mask = forecast_df["date"] >= split_date

X_train_forecast = X_forecast.loc[train_mask]
X_test_forecast = X_forecast.loc[test_mask]

y_train_forecast = y_forecast.loc[train_mask]
y_test_forecast = y_forecast.loc[test_mask]

forecast_model = LinearRegression()

forecast_model.fit(
    X_train_forecast,
    y_train_forecast
)

forecast_predictions = forecast_model.predict(
    X_test_forecast
)

mae = mean_absolute_error(
    y_test_forecast,
    forecast_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test_forecast,
        forecast_predictions
    )
)

r2 = r2_score(
    y_test_forecast,
    forecast_predictions
)

print("Training records:", len(X_train_forecast))
print("Testing records:", len(X_test_forecast))
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))

forecast_results = forecast_df.loc[
    test_mask,
    ["date", "item", "price_shekel"]
].copy()

forecast_results["predicted_price"] = forecast_predictions

forecast_results["prediction_error"] = (
    forecast_results["price_shekel"]
    - forecast_results["predicted_price"]
)

forecast_results.to_csv(
    "price_forecasting_results.csv",
    index=False,
    encoding="utf-8-sig"
)

plt.figure(figsize=(10, 5))
plt.plot(
    y_test_forecast.to_numpy(),
    label="Actual Prices"
)
plt.plot(
    forecast_predictions,
    label="Predicted Prices"
)
plt.title("Actual vs Predicted Prices")
plt.xlabel("Test Record")
plt.ylabel("Price (Shekel)")
plt.legend()
plt.tight_layout()
plt.savefig("price_forecasting.png", dpi=150)
plt.close()

# ============================================================
# 6. IMPROVED AVAILABILITY CLASSIFICATION
# ============================================================

print("\n" + "=" * 65)
print("6. AVAILABILITY CLASSIFICATION")
print("=" * 65)

classification_df = df.copy()

classification_dummies = pd.get_dummies(
    classification_df["item"],
    prefix="item",
    dtype=int
)

classification_df = pd.concat(
    [classification_df, classification_dummies],
    axis=1
)

classification_features = [
    "price_shekel",
    "usd_rate",
    "border_open",
    "crisis_level"
] + list(classification_dummies.columns)

X = classification_df[classification_features]
y = classification_df["availability"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# البحث عن أفضل إعدادات Logistic Regression
logistic_pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=5000,
        random_state=42
    )
)

logistic_parameters = {
    "logisticregression__C": [
        0.01,
        0.1,
        1,
        10,
        100
    ],
    "logisticregression__class_weight": [
        None,
        "balanced"
    ]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

logistic_search = GridSearchCV(
    estimator=logistic_pipeline,
    param_grid=logistic_parameters,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1
)

logistic_search.fit(X_train, y_train)

# نقارن نماذج إضافية باستخدام التدريب فقط
random_forest = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

extra_trees = ExtraTreesClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

models = {
    "Logistic Regression": logistic_search.best_estimator_,
    "Random Forest": random_forest,
    "Extra Trees": extra_trees
}

model_results = []

# نستخدم تحققًا متقاطعًا لاختيار النموذج، دون لمس الاختبار
from sklearn.model_selection import cross_val_score

for name, candidate in models.items():
    scores = cross_val_score(
        candidate,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    model_results.append({
        "name": name,
        "model": candidate,
        "cv_accuracy": scores.mean()
    })

best_result = max(
    model_results,
    key=lambda result: result["cv_accuracy"]
)

best_model = best_result["model"]

# تدريب النموذج المختار ثم تقييمه مرة واحدة على الاختبار
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nCross-validation comparison:")
for result in model_results:
    print(
        result["name"],
        ":",
        round(result["cv_accuracy"] * 100, 2),
        "%"
    )

print("\nBest model:", best_result["name"])
print(
    "Best model cross-validation accuracy:",
    round(best_result["cv_accuracy"] * 100, 2),
    "%"
)
print(
    "Independent test accuracy:",
    round(accuracy * 100, 2),
    "%"
)

labels = ["متوفر", "محدود", "ناقص"]

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=labels,
        zero_division=0
    )
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(7, 5))
plt.imshow(cm, interpolation="nearest")
plt.title("Availability Classification Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.xticks(range(len(labels)), labels)
plt.yticks(range(len(labels)), labels)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.savefig(
    "availability_confusion_matrix.png",
    dpi=150
)
plt.close()

classification_results = X_test.copy()

classification_results["actual_availability"] = y_test
classification_results["predicted_availability"] = y_pred

classification_results.to_csv(
    "availability_classification_results.csv",
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 7. SAVE SUMMARY
# ============================================================

summary = pd.DataFrame({
    "Metric": [
        "Number of records",
        "Number of goods",
        "Number of days",
        "Detected anomalies",
        "Forecast MAE",
        "Forecast RMSE",
        "Forecast R2",
        "Best classifier",
        "Cross-validation accuracy percent",
        "Independent test accuracy percent"
    ],
    "Value": [
        len(df),
        df["item"].nunique(),
        df["date"].nunique(),
        len(anomalies),
        round(mae, 4),
        round(rmse, 4),
        round(r2, 4),
        best_result["name"],
        round(best_result["cv_accuracy"] * 100, 2),
        round(accuracy * 100, 2)
    ]
})

summary.to_csv(
    "market_crisis_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 65)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 65)

output_files = [
    "market_data.csv",
    "market_data_with_clusters.csv",
    "market_price_trend.png",
    "average_prices_by_item.png",
    "kmeans_elbow.png",
    "market_clusters.png",
    "market_anomalies.csv",
    "market_anomalies.png",
    "price_forecasting_results.csv",
    "price_forecasting.png",
    "availability_confusion_matrix.png",
    "availability_classification_results.csv",
    "market_crisis_summary.csv"
]

print("\nGenerated files:")
for filename in output_files:
    print("-", filename)

print(
    "\nImportant: The dataset is synthetic. "
    "A high score on synthetic data does not guarantee real-world performance."
)
