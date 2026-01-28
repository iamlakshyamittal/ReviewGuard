"""
Train ReviewGuard ML Model
Simple training script that matches the new predict.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from textblob import TextBlob
import joblib
import os

print("\nTraining started...")

# ===== LOAD DATA =====
try:
    data = pd.read_csv("train_reviews.csv")
    print(f"✅ Loaded {len(data)} reviews")
except FileNotFoundError:
    print("\n❌ Error: train_reviews.csv not found!")
    print("\nPlease run this first:")
    print("  python generate_training_data.py")
    print("\nThis will create sample training data.")
    exit(1)

# ===== EXTRACT FEATURES (SAME AS predict.py) =====
print("Extracting features...")

def extract_training_features(row):
    """Extract same features as predict.py"""
    review_text = str(row['review_text'])
    rating = row['rating']
    total_reviews = row.get('total_reviews', 1)
    avg_rating = row.get('avg_rating', rating)
    
    # Text analysis
    words = review_text.split()
    word_count = len(words)
    text_length = len(review_text)
    
    blob = TextBlob(review_text)
    sentiment = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    uppercase_ratio = sum(1 for c in review_text if c.isupper()) / text_length if text_length > 0 else 0
    exclamation_count = review_text.count("!")
    question_count = review_text.count("?")
    
    # Behavioral features
    rating_variance = abs(avg_rating - 3.0) * 0.5
    rating_std = np.sqrt(rating_variance)
    
    if avg_rating >= 4.7 or avg_rating <= 1.3:
        extreme_rating_ratio = 0.95
    elif avg_rating >= 4.3 or avg_rating <= 1.7:
        extreme_rating_ratio = 0.70
    else:
        extreme_rating_ratio = 0.30
    
    rating_deviation = abs(avg_rating - 3.5)
    
    if total_reviews <= 3:
        reviews_per_day = 0.1
        burst_score = 0.2
    elif total_reviews <= 10:
        reviews_per_day = 0.5
        burst_score = 0.6
    elif total_reviews <= 25:
        reviews_per_day = 0.3
        burst_score = 0.7
    else:
        reviews_per_day = 0.1
        burst_score = 0.2
    
    if abs(avg_rating - 5.0) < 0.2 or abs(avg_rating - 1.0) < 0.2:
        rating_consistency = 0.95
    elif abs(avg_rating - 5.0) < 0.5 or abs(avg_rating - 1.0) < 0.5:
        rating_consistency = 0.75
    else:
        rating_consistency = 0.40
    
    if total_reviews <= 5:
        unique_products = max(1, total_reviews - 1)
    elif total_reviews <= 15:
        unique_products = max(3, total_reviews - 5)
    else:
        unique_products = total_reviews - 10
    
    product_diversity_ratio = unique_products / total_reviews
    
    return {
        "total_reviews": total_reviews,
        "avg_rating": avg_rating,
        "rating_variance": rating_variance,
        "rating_std": rating_std,
        "avg_review_length": text_length,
        "avg_word_count": word_count,
        "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
        "avg_sentiment": sentiment,
        "avg_subjectivity": subjectivity,
        "avg_uppercase_ratio": uppercase_ratio,
        "avg_exclamations": exclamation_count,
        "avg_questions": question_count,
        "unique_products": unique_products,
        "review_span_days": 30,
        "reviews_per_day": reviews_per_day,
        "product_diversity_ratio": product_diversity_ratio,
        "extreme_rating_ratio": extreme_rating_ratio,
        "rating_deviation": rating_deviation,
        "burst_score": burst_score,
        "rating_consistency": rating_consistency,
        "degree_centrality": 0.0,
        "clustering_coefficient": 0.0,
        "betweenness_centrality": 0.0,
        "pagerank": 0.0,
        "num_connections": 0,
        "in_suspicious_community": 0,
        "network_suspicion_score": 0.0,
    }

# Extract features for all rows
features_list = [extract_training_features(row) for _, row in data.iterrows()]
X = pd.DataFrame(features_list)
y = data['label']

print(f"✅ Extracted {len(X.columns)} features")

# ===== SPLIT DATA =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

# ===== SCALE FEATURES =====
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===== TRAIN MODEL =====
print("Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)
print("✅ Model trained")

# ===== EVALUATE =====
print("\n" + "="*70)
print("MODEL EVALUATION")
print("="*70)

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fake']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC-AUC Score: {roc_auc:.4f}")

# ===== SAVE MODEL =====
print("\n" + "="*70)
print("SAVING MODEL")
print("="*70)

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/fake_review_detector.pkl")
joblib.dump(scaler, "models/feature_scaler.pkl")

print("✅ Saved models/fake_review_detector.pkl")
print("✅ Saved models/feature_scaler.pkl")

# ===== FEATURE IMPORTANCE =====
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

feature_importance.to_csv("models/feature_importance.csv", index=False)
print("✅ Saved models/feature_importance.csv")

print("\n" + "="*70)
print("Top 10 Most Important Features:")
print("="*70)
for _, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:30s} {row['importance']:.4f}")

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
print("\nYou can now:")
print("  1. Start the API: python app.py")
print("  2. Open index.html in browser")
print("  3. Test fake review detection!")