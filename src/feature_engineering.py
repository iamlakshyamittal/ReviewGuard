"""
Feature engineering module for ReviewGuard
Creates behavioral and textual features to detect fake reviewers
"""
import pandas as pd
import numpy as np
from textblob import TextBlob
from config.settings import (
    PROCESSED_DATA_PATH, 
    REVIEWER_FEATURES_PATH,
    MAX_REVIEW_LENGTH
)
from src.utils import setup_logger, print_section_header

logger = setup_logger(__name__)


def extract_text_features(text):
    """
    Extract features from review text
    
    Args:
        text: Review text string
    
    Returns:
        dict: Dictionary of text features
    """
    blob = TextBlob(str(text))
    
    return {
        'length': len(text),
        'word_count': len(text.split()),
        'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
        'sentiment_polarity': blob.sentiment.polarity,
        'sentiment_subjectivity': blob.sentiment.subjectivity,
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0,
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
    }


def create_features(input_path=PROCESSED_DATA_PATH, output_path=REVIEWER_FEATURES_PATH):
    """
    Engineer comprehensive features for fake reviewer detection
    
    Features created:
    1. Basic reviewer stats (total reviews, avg rating)
    2. Rating patterns (variance, extreme rating ratio)
    3. Text patterns (length, sentiment, complexity)
    4. Temporal patterns (review frequency, burst detection)
    5. Product diversity
    
    Args:
        input_path: Path to processed reviews CSV
        output_path: Path to save reviewer features CSV
    
    Returns:
        pd.DataFrame: Reviewer features
    """
    print_section_header("FEATURE ENGINEERING")
    logger.info(f"Loading processed reviews from {input_path}")
    
    try:
        # Load data
        df = pd.read_csv(input_path)
        df["reviewTime"] = pd.to_datetime(df["reviewTime"])
        logger.info(f"Loaded {len(df):,} processed reviews")
        
        # Extract text features for each review
        logger.info("Extracting text features...")
        text_features = df["reviewText"].apply(extract_text_features)
        text_df = pd.DataFrame(text_features.tolist())
        
        # Add text features to main dataframe
        for col in text_df.columns:
            df[f"text_{col}"] = text_df[col]
        
        # Compute reviewer-level aggregations
        logger.info("Aggregating reviewer-level features...")
        
        reviewer_features = df.groupby("reviewerID").agg(
            # Basic statistics
            total_reviews=("reviewText", "count"),
            avg_rating=("rating", "mean"),
            rating_variance=("rating", lambda x: x.var() if len(x) > 1 else 0),
            rating_std=("rating", lambda x: x.std() if len(x) > 1 else 0),
            
            # Text features
            avg_review_length=("text_length", "mean"),
            avg_word_count=("text_word_count", "mean"),
            avg_word_length=("text_avg_word_length", "mean"),
            avg_sentiment=("text_sentiment_polarity", "mean"),
            avg_subjectivity=("text_sentiment_subjectivity", "mean"),
            avg_uppercase_ratio=("text_uppercase_ratio", "mean"),
            avg_exclamations=("text_exclamation_count", "mean"),
            avg_questions=("text_question_count", "mean"),
            
            # Product diversity
            unique_products=("productID", "nunique"),
            
            # Temporal features
            first_review=("reviewTime", "min"),
            last_review=("reviewTime", "max"),
        ).reset_index()
        
        logger.info(f"Created features for {len(reviewer_features):,} reviewers")
        
        # Derive additional features
        logger.info("Creating derived features...")
        
        # Time span and review frequency
        reviewer_features["review_span_days"] = (
            reviewer_features["last_review"] - reviewer_features["first_review"]
        ).dt.days
        
        reviewer_features["reviews_per_day"] = (
            reviewer_features["total_reviews"] / 
            (reviewer_features["review_span_days"] + 1)  # +1 to avoid division by zero
        )
        
        # Product diversity ratio
        reviewer_features["product_diversity_ratio"] = (
            reviewer_features["unique_products"] / reviewer_features["total_reviews"]
        )
        
        # Extreme rating ratio (only 1-star or 5-star)
        extreme_ratings = df.groupby("reviewerID")["rating"].apply(
            lambda x: ((x == 1) | (x == 5)).sum() / len(x)
        ).reset_index(name="extreme_rating_ratio")
        
        reviewer_features = reviewer_features.merge(
            extreme_ratings, on="reviewerID", how="left"
        )
        
        # Rating deviation from mean (users who always give extreme ratings)
        global_avg_rating = df["rating"].mean()
        reviewer_features["rating_deviation"] = abs(
            reviewer_features["avg_rating"] - global_avg_rating
        )
        
        # Review burst score (detect sudden bursts of reviews)
        # Reviews per day > 1 is suspicious
        reviewer_features["burst_score"] = reviewer_features["reviews_per_day"].apply(
            lambda x: min(x, 10) / 10  # Normalize to 0-1, cap at 10
        )
        
        # Consistency score (low variance = always same rating)
        reviewer_features["rating_consistency"] = 1 - (
            reviewer_features["rating_variance"] / reviewer_features["rating_variance"].max()
        ).fillna(0)
        
        # Drop datetime columns (not needed for modeling)
        reviewer_features.drop(columns=["first_review", "last_review"], inplace=True)
        
        # Handle any remaining NaN values
        reviewer_features.fillna(0, inplace=True)
        
        # Save features
        reviewer_features.to_csv(output_path, index=False)
        logger.info(f"Saved {len(reviewer_features.columns)} features to {output_path}")
        
        # Print feature summary
        print(f"\n✅ Feature engineering completed successfully")
        print(f"   Total reviewers: {len(reviewer_features):,}")
        print(f"   Features created: {len(reviewer_features.columns) - 1}")  # -1 for reviewerID
        print(f"\n   Feature categories:")
        print(f"     - Basic stats: 4")
        print(f"     - Text patterns: 8")
        print(f"     - Temporal patterns: 3")
        print(f"     - Behavioral patterns: 6")
        
        # Print some statistics
        print(f"\n   Reviewer statistics:")
        print(f"     - Avg reviews per reviewer: {reviewer_features['total_reviews'].mean():.1f}")
        print(f"     - Max reviews by one reviewer: {reviewer_features['total_reviews'].max()}")
        print(f"     - Avg rating: {reviewer_features['avg_rating'].mean():.2f}")
        print(f"     - High burst score (>0.5): {(reviewer_features['burst_score'] > 0.5).sum():,}")
        
        return reviewer_features
        
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
        raise
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        raise


if __name__ == "__main__":
    create_features()
