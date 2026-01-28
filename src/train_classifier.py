"""
Supervised machine learning training module for ReviewGuard
Trains classification models to detect fake reviews
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    precision_recall_curve,
    f1_score
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
from config.settings import (
    SUSPICIOUS_REVIEWERS_PATH,
    MODEL_PATH,
    SCALER_PATH,
    FEATURE_IMPORTANCE_PATH,
    TEST_SIZE,
    RANDOM_STATE
)
from src.utils import setup_logger, print_section_header

logger = setup_logger(__name__)


def train_fake_review_detector(
    input_path=SUSPICIOUS_REVIEWERS_PATH,
    model_path=MODEL_PATH,
    scaler_path=SCALER_PATH,
    feature_importance_path=FEATURE_IMPORTANCE_PATH
):
    """
    Train supervised models to detect fake reviewers
    
    Uses anomaly detection results as pseudo-labels for training
    Trains multiple models and selects the best based on ROC-AUC
    
    Args:
        input_path: Path to features with anomaly labels
        model_path: Path to save trained model
        scaler_path: Path to save feature scaler
        feature_importance_path: Path to save feature importances
    
    Returns:
        tuple: (best_model, scaler, feature_columns)
    """
    print_section_header("SUPERVISED MODEL TRAINING")
    logger.info(f"Loading data from {input_path}")
    
    try:
        # Load data with features and labels
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df):,} reviewers")
        
        # Use combined suspicion as labels (threshold-based)
        # This creates pseudo-labels from unsupervised detection
        df['label'] = (df['combined_suspicion_score'] > 0.5).astype(int)
        
        logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
        
        # Select features for training
        exclude_cols = [
            'reviewerID', 
            'is_suspicious',
            'iso_forest_anomaly',
            'lof_anomaly',
            'iso_forest_score',
            'lof_score',
            'iso_forest_score_norm',
            'lof_score_norm',
            'combined_suspicion_score',
            'risk_level',
            'label'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df['label'].copy()
        
        logger.info(f"Using {len(feature_cols)} features for training")
        
        # Handle missing values
        X.fillna(X.mean(), inplace=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE,
            stratify=y
        )
        
        logger.info(f"Train set: {len(X_train):,} samples")
        logger.info(f"Test set: {len(X_test):,} samples")
        
        # Handle class imbalance with SMOTE
        logger.info("Applying SMOTE for class balancing...")
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        logger.info(
            f"After SMOTE: {len(X_train_balanced):,} samples, "
            f"class distribution: {pd.Series(y_train_balanced).value_counts().to_dict()}"
        )
        
        # Scale features
        logger.info("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_balanced)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                random_state=RANDOM_STATE,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                random_state=RANDOM_STATE,
                class_weight='balanced',
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=RANDOM_STATE
            )
        }
        
        results = {}
        best_model = None
        best_score = 0
        best_name = ""
        
        print(f"\n{'='*70}")
        print("TRAINING MODELS")
        print(f"{'='*70}")
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            print(f"\n{name}:")
            print("-" * 50)
            
            # Train model
            model.fit(X_train_scaled, y_train_balanced)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            f1 = f1_score(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train_balanced, 
                cv=5, scoring='roc_auc', n_jobs=-1
            )
            
            results[name] = {
                'model': model,
                'roc_auc': roc_auc,
                'f1': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            # Print results
            print(f"ROC-AUC Score: {roc_auc:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            print(f"\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fake']))
            print(f"\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
            
            # Track best model
            if roc_auc > best_score:
                best_score = roc_auc
                best_model = model
                best_name = name
        
        # Summary
        print(f"\n{'='*70}")
        print("MODEL COMPARISON")
        print(f"{'='*70}")
        
        comparison_df = pd.DataFrame({
            'Model': results.keys(),
            'ROC-AUC': [r['roc_auc'] for r in results.values()],
            'F1-Score': [r['f1'] for r in results.values()],
            'CV Mean': [r['cv_mean'] for r in results.values()],
            'CV Std': [r['cv_std'] for r in results.values()]
        })
        
        print(comparison_df.to_string(index=False))
        
        print(f"\n{'='*70}")
        print(f"BEST MODEL: {best_name}")
        print(f"ROC-AUC: {best_score:.4f}")
        print(f"{'='*70}")
        
        # Save best model and scaler
        logger.info(f"Saving {best_name} to {model_path}")
        joblib.dump(best_model, model_path)
        joblib.dump(scaler, scaler_path)
        
        # Feature importance (if available)
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            feature_importance.to_csv(feature_importance_path, index=False)
            logger.info(f"Saved feature importances to {feature_importance_path}")
            
            print(f"\nTop 10 Most Important Features:")
            print("-" * 50)
            for idx, row in feature_importance.head(10).iterrows():
                print(f"{row['feature']:30s} {row['importance']:.4f}")
        
        elif hasattr(best_model, 'coef_'):
            # For logistic regression
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': np.abs(best_model.coef_[0])
            }).sort_values('importance', ascending=False)
            
            feature_importance.to_csv(feature_importance_path, index=False)
            logger.info(f"Saved feature importances to {feature_importance_path}")
            
            print(f"\nTop 10 Most Important Features:")
            print("-" * 50)
            for idx, row in feature_importance.head(10).iterrows():
                print(f"{row['feature']:30s} {row['importance']:.4f}")
        
        print(f"\n✅ Model training completed successfully")
        print(f"   Model saved to: {model_path}")
        print(f"   Scaler saved to: {scaler_path}")
        
        return best_model, scaler, feature_cols
        
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
        raise
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise


if __name__ == "__main__":
    train_fake_review_detector()