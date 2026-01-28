"""
Prediction module for ReviewGuard
Real-time fake review detection using trained ML model
FIXED VERSION - Now properly uses reviewer behavioral features!
"""

import pandas as pd
import numpy as np
import joblib
from textblob import TextBlob

import config.settings as settings
from src.utils import setup_logger

logger = setup_logger(__name__)

MODEL_PATH = settings.MODEL_PATH
SCALER_PATH = settings.SCALER_PATH


class FakeReviewDetector:
    """
    Fake Review Detector with Enhanced Feature Extraction
    
    Now properly analyzes:
    - Text patterns (sentiment, length, caps)
    - Reviewer behavior (total reviews, rating patterns)
    - Rating consistency
    """

    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
        logger.info("Loading trained model...")
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        logger.info("Model loaded successfully")

    def extract_features(self, review_text, rating, reviewer_total_reviews=1, 
                        reviewer_avg_rating=None):
        """
        Extract COMPREHENSIVE features for fake review detection
        
        Args:
            review_text: The review text
            rating: Rating given (1-5)
            reviewer_total_reviews: How many reviews this person has written
            reviewer_avg_rating: Their average rating across all reviews
        
        Returns:
            dict: All features needed for prediction
        """
        
        # Default avg rating if not provided
        if reviewer_avg_rating is None:
            reviewer_avg_rating = rating
        
        # ===== TEXT ANALYSIS =====
        words = review_text.split()
        word_count = len(words)
        text_length = len(review_text)
        
        # Sentiment analysis
        blob = TextBlob(review_text)
        sentiment = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Character patterns
        uppercase_ratio = sum(1 for c in review_text if c.isupper()) / text_length if text_length > 0 else 0
        exclamation_count = review_text.count("!")
        question_count = review_text.count("?")
        
        # ===== BEHAVIORAL FEATURES (KEY TO DETECTION!) =====
        
        # 1. Rating variance - Fake reviewers give consistent ratings
        # Estimate based on how far their avg is from neutral (3.0)
        rating_variance = abs(reviewer_avg_rating - 3.0) * 0.5
        rating_std = np.sqrt(rating_variance)
        
        # 2. Extreme rating ratio - Do they only give 1 or 5 stars?
        # If avg rating is 4.7+, they likely only give 5 stars
        # If avg rating is 1.3-, they likely only give 1 star
        if reviewer_avg_rating >= 4.7 or reviewer_avg_rating <= 1.3:
            extreme_rating_ratio = 0.95  # Almost always extreme
        elif reviewer_avg_rating >= 4.3 or reviewer_avg_rating <= 1.7:
            extreme_rating_ratio = 0.70  # Often extreme
        else:
            extreme_rating_ratio = 0.30  # Balanced reviewer
        
        # 3. Rating deviation from average
        rating_deviation = abs(reviewer_avg_rating - 3.5)  # 3.5 is typical average
        
        # 4. Review burst score - Many reviews in short time
        # Assume if someone has 5-20 reviews, they might be in a "burst"
        if reviewer_total_reviews <= 3:
            reviews_per_day = 0.1  # New reviewer
            burst_score = 0.2
        elif reviewer_total_reviews <= 10:
            reviews_per_day = 0.5  # Possible burst
            burst_score = 0.6
        elif reviewer_total_reviews <= 25:
            reviews_per_day = 0.3  # Active but suspicious
            burst_score = 0.7
        else:
            reviews_per_day = 0.1  # Established reviewer
            burst_score = 0.2
        
        # 5. Rating consistency - Low variance = suspicious
        # If avg is 4.9, they're VERY consistent (fake pattern)
        if abs(reviewer_avg_rating - 5.0) < 0.2 or abs(reviewer_avg_rating - 1.0) < 0.2:
            rating_consistency = 0.95  # Almost always same rating
        elif abs(reviewer_avg_rating - 5.0) < 0.5 or abs(reviewer_avg_rating - 1.0) < 0.5:
            rating_consistency = 0.75
        else:
            rating_consistency = 0.40  # Normal variation
        
        # 6. Product diversity ratio
        # Assume fake reviewers focus on few products
        if reviewer_total_reviews <= 5:
            unique_products = max(1, reviewer_total_reviews - 1)
        elif reviewer_total_reviews <= 15:
            unique_products = max(3, reviewer_total_reviews - 5)
        else:
            unique_products = reviewer_total_reviews - 10
        
        product_diversity_ratio = unique_products / reviewer_total_reviews
        
        # ===== COMBINE ALL FEATURES =====
        features = {
            # Basic stats
            "total_reviews": reviewer_total_reviews,
            "avg_rating": reviewer_avg_rating,
            "rating_variance": rating_variance,
            "rating_std": rating_std,
            
            # Text features (original 8)
            "avg_review_length": text_length,
            "avg_word_count": word_count,
            "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
            "avg_sentiment": sentiment,
            "avg_subjectivity": subjectivity,
            "avg_uppercase_ratio": uppercase_ratio,
            "avg_exclamations": exclamation_count,
            "avg_questions": question_count,
            
            # Product diversity
            "unique_products": unique_products,
            
            # Temporal patterns
            "review_span_days": 30,  # Assume 30 days (we don't have actual dates)
            "reviews_per_day": reviews_per_day,
            
            # Derived behavioral features
            "product_diversity_ratio": product_diversity_ratio,
            "extreme_rating_ratio": extreme_rating_ratio,
            "rating_deviation": rating_deviation,
            "burst_score": burst_score,
            "rating_consistency": rating_consistency,
            
            # Network features (set to 0 for single review prediction)
            "degree_centrality": 0.0,
            "clustering_coefficient": 0.0,
            "betweenness_centrality": 0.0,
            "pagerank": 0.0,
            "num_connections": 0,
            "in_suspicious_community": 0,
            "network_suspicion_score": 0.0,
        }
        
        return features

    def predict(self, review_text, rating, reviewer_total_reviews=1, 
                reviewer_avg_rating=None, **kwargs):
        """
        Predict if a review is fake
        
        Args:
            review_text: The review text
            rating: Rating (1-5)
            reviewer_total_reviews: How many reviews this person has written
            reviewer_avg_rating: Their average rating
            **kwargs: Extra args ignored for compatibility
        
        Returns:
            dict: Prediction results with probabilities
        """
        
        # Extract ALL features
        features = self.extract_features(
            review_text, 
            rating,
            reviewer_total_reviews,
            reviewer_avg_rating
        )
        
        # Convert to DataFrame
        X = pd.DataFrame([features])
        
        # Handle missing columns (model might not use all features)
        try:
            X_scaled = self.scaler.transform(X)
        except ValueError as e:
            # If model expects different features, use only the ones it needs
            logger.warning(f"Feature mismatch, using available features: {e}")
            
            # Get expected features from scaler
            try:
                expected_features = self.scaler.feature_names_in_
                X = X[expected_features]
                X_scaled = self.scaler.transform(X)
            except:
                # Fallback: try with original 8 features only
                logger.warning("Using minimal feature set")
                X = pd.DataFrame([{
                    "avg_review_length": features["avg_review_length"],
                    "avg_word_count": features["avg_word_count"],
                    "avg_word_length": features["avg_word_length"],
                    "avg_sentiment": features["avg_sentiment"],
                    "avg_subjectivity": features["avg_subjectivity"],
                    "avg_exclamations": features["avg_exclamations"],
                    "avg_questions": features["avg_questions"],
                    "extreme_rating_ratio": features["extreme_rating_ratio"],
                }])
                X_scaled = self.scaler.transform(X)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]
        
        fake_prob = probability[1]
        
        # Determine risk level
        if fake_prob > 0.7:
            risk_level = "HIGH"
        elif fake_prob > 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        result = {
            "is_fake": bool(prediction),
            "fake_probability": float(fake_prob),
            "legitimate_probability": float(probability[0]),
            "risk_level": risk_level,
            "confidence": float(max(probability)),
        }
        
        logger.info(
            f"Prediction | Fake={prediction} | Prob={fake_prob:.2f} | "
            f"Context: {reviewer_total_reviews} reviews, avg {reviewer_avg_rating:.1f}★"
        )
        
        return result


def predict_single_review(review_text, rating, reviewer_total_reviews=1, 
                          reviewer_avg_rating=None):
    """Convenience function for single prediction"""
    detector = FakeReviewDetector()
    return detector.predict(review_text, rating, reviewer_total_reviews, reviewer_avg_rating)


if __name__ == "__main__":
    # Test the detector
    detector = FakeReviewDetector()
    
    print("\n" + "="*70)
    print("TESTING FAKE REVIEW DETECTOR")
    print("="*70)
    
    examples = [
        {
            "text": "AMAZING PRODUCT!!! BEST EVER!!! BUY NOW!!!",
            "rating": 5,
            "total_reviews": 8,
            "avg_rating": 4.9,
            "expected": "HIGH RISK (fake pattern)"
        },
        {
            "text": "Good product. Works as described. Delivery was on time.",
            "rating": 4,
            "total_reviews": 47,
            "avg_rating": 3.7,
            "expected": "LOW RISK (legitimate pattern)"
        },
        {
            "text": "Terrible! Scam! Don't buy!",
            "rating": 1,
            "total_reviews": 12,
            "avg_rating": 1.2,
            "expected": "HIGH RISK (fake negative pattern)"
        }
    ]
    
    for i, ex in enumerate(examples, 1):
        print(f"\nExample {i}: {ex['expected']}")
        print(f"Text: {ex['text'][:50]}...")
        print(f"Rating: {ex['rating']}★ | Reviews: {ex['total_reviews']} | Avg: {ex['avg_rating']}★")
        
        result = detector.predict(
            ex['text'], 
            ex['rating'],
            ex['total_reviews'],
            ex['avg_rating']
        )
        
        print(f"→ Result: {'FAKE' if result['is_fake'] else 'LEGIT'} | "
              f"Confidence: {result['fake_probability']:.1%} | "
              f"Risk: {result['risk_level']}")
        print("-" * 70)