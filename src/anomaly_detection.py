"""
Anomaly detection module for ReviewGuard
Uses unsupervised learning to identify suspicious reviewers
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from config.settings import (
    REVIEWER_FEATURES_PATH,
    NETWORK_FEATURES_PATH,
    SUSPICIOUS_REVIEWERS_PATH,
    ANOMALY_CONTAMINATION,
    RANDOM_STATE
)
from src.utils import setup_logger, print_section_header

logger = setup_logger(__name__)


def detect_fake_reviewers(
    features_path=REVIEWER_FEATURES_PATH,
    network_path=NETWORK_FEATURES_PATH,
    output_path=SUSPICIOUS_REVIEWERS_PATH
):
    """
    Detect fake reviewers using anomaly detection algorithms
    
    Uses:
    - Isolation Forest: Tree-based anomaly detection
    - Local Outlier Factor: Density-based outlier detection
    
    Args:
        features_path: Path to reviewer features CSV
        network_path: Path to network features CSV
        output_path: Path to save results CSV
    
    Returns:
        pd.DataFrame: Reviewers with anomaly scores
    """
    print_section_header("ANOMALY DETECTION")
    logger.info("Loading features...")
    
    try:
        # Load features
        reviewer_df = pd.read_csv(features_path)
        network_df = pd.read_csv(network_path)
        
        # Merge features
        df = reviewer_df.merge(network_df, on="reviewerID", how="left")
        logger.info(f"Loaded features for {len(df):,} reviewers")
        
        # Fill missing network features (isolated reviewers)
        network_cols = network_df.columns.drop('reviewerID')
        df[network_cols] = df[network_cols].fillna(0)
        
        # Select features for anomaly detection
        feature_cols = [col for col in df.columns if col != 'reviewerID']
        X = df[feature_cols].copy()
        
        logger.info(f"Using {len(feature_cols)} features for anomaly detection")
        
        # Handle any remaining missing values
        X.fillna(X.mean(), inplace=True)
        
        # Standardize features (important for distance-based methods)
        logger.info("Standardizing features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Isolation Forest
        logger.info(f"Running Isolation Forest (contamination={ANOMALY_CONTAMINATION})...")
        iso_forest = IsolationForest(
            contamination=ANOMALY_CONTAMINATION,
            random_state=RANDOM_STATE,
            n_estimators=100,
            max_features=1.0,
            bootstrap=False
        )
        
        iso_predictions = iso_forest.fit_predict(X_scaled)
        iso_scores = iso_forest.score_samples(X_scaled)
        
        # Convert predictions: -1 (anomaly) -> 1, 1 (normal) -> 0
        df['iso_forest_anomaly'] = (iso_predictions == -1).astype(int)
        df['iso_forest_score'] = -iso_scores  # Negate so higher = more suspicious
        
        logger.info(
            f"Isolation Forest identified {df['iso_forest_anomaly'].sum():,} "
            f"suspicious reviewers ({df['iso_forest_anomaly'].mean()*100:.2f}%)"
        )
        
        # Local Outlier Factor
        logger.info(f"Running Local Outlier Factor...")
        lof = LocalOutlierFactor(
            contamination=ANOMALY_CONTAMINATION,
            n_neighbors=20,
            novelty=False
        )
        
        lof_predictions = lof.fit_predict(X_scaled)
        lof_scores = -lof.negative_outlier_factor_  # Convert to positive scores
        
        df['lof_anomaly'] = (lof_predictions == -1).astype(int)
        df['lof_score'] = lof_scores
        
        logger.info(
            f"LOF identified {df['lof_anomaly'].sum():,} "
            f"suspicious reviewers ({df['lof_anomaly'].mean()*100:.2f}%)"
        )
        
        # Ensemble: Flag as suspicious if either method detects
        df['is_suspicious'] = (
            (df['iso_forest_anomaly'] == 1) | (df['lof_anomaly'] == 1)
        ).astype(int)
        
        # Combined suspicion score (average of normalized scores)
        df['iso_forest_score_norm'] = (
            (df['iso_forest_score'] - df['iso_forest_score'].min()) /
            (df['iso_forest_score'].max() - df['iso_forest_score'].min())
        )
        
        df['lof_score_norm'] = (
            (df['lof_score'] - df['lof_score'].min()) /
            (df['lof_score'].max() - df['lof_score'].min())
        )
        
        df['combined_suspicion_score'] = (
            df['iso_forest_score_norm'] + df['lof_score_norm']
        ) / 2
        
        # Risk level categorization
        df['risk_level'] = pd.cut(
            df['combined_suspicion_score'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['LOW', 'MEDIUM', 'HIGH']
        )
        
        # Sort by suspicion score
        df.sort_values('combined_suspicion_score', ascending=False, inplace=True)
        
        # Save results
        df.to_csv(output_path, index=False)
        logger.info(f"Saved results to {output_path}")
        
        # Print summary
        print(f"\n✅ Anomaly detection completed successfully")
        print(f"\n   Detection Summary:")
        print(f"     - Total reviewers analyzed: {len(df):,}")
        print(f"     - Suspicious (Isolation Forest): {df['iso_forest_anomaly'].sum():,}")
        print(f"     - Suspicious (LOF): {df['lof_anomaly'].sum():,}")
        print(f"     - Suspicious (Combined): {df['is_suspicious'].sum():,} "
              f"({df['is_suspicious'].mean()*100:.2f}%)")
        
        print(f"\n   Risk Level Distribution:")
        print(f"     - HIGH: {(df['risk_level'] == 'HIGH').sum():,}")
        print(f"     - MEDIUM: {(df['risk_level'] == 'MEDIUM').sum():,}")
        print(f"     - LOW: {(df['risk_level'] == 'LOW').sum():,}")
        
        # Show top 5 most suspicious reviewers
        print(f"\n   Top 5 Most Suspicious Reviewers:")
        top_suspicious = df.nlargest(5, 'combined_suspicion_score')
        for idx, row in top_suspicious.iterrows():
            print(f"     - {row['reviewerID'][:15]}... "
                  f"(score: {row['combined_suspicion_score']:.3f}, "
                  f"risk: {row['risk_level']})")
        
        return df
        
    except FileNotFoundError as e:
        logger.error(f"Input file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during anomaly detection: {e}")
        raise


if __name__ == "__main__":
    detect_fake_reviewers()