"""
Flask REST API for ReviewGuard
Real-time fake review detection service
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
from src.predict import FakeReviewDetector
from src.utils import setup_logger
import config.settings as settings

API_HOST = settings.API_HOST
API_PORT = settings.API_PORT
API_DEBUG = settings.API_DEBUG

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Setup logger
logger = setup_logger(__name__)

# Initialize detector (loaded once at startup)
try:
    detector = FakeReviewDetector()
    logger.info("ReviewGuard API initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize detector: {e}")
    detector = None


@app.route('/', methods=['GET'])
def home():
    """API home endpoint"""
    return jsonify({
        'service': 'ReviewGuard API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            '/': 'API information',
            '/health': 'Health check',
            '/predict': 'Predict single review (POST)',
            '/predict/batch': 'Predict multiple reviews (POST)'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    if detector is None:
        return jsonify({
            'status': 'unhealthy',
            'message': 'Model not loaded'
        }), 503
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': True
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict if a single review is fake
    
    Request JSON:
    {
        "review_text": "This product is amazing!",
        "rating": 5,
        "reviewer_total_reviews": 10,  # Optional
        "reviewer_avg_rating": 4.5     # Optional
    }
    
    Response JSON:
    {
        "is_fake": false,
        "fake_probability": 0.234,
        "legitimate_probability": 0.766,
        "risk_level": "LOW",
        "confidence": 0.766
    }
    """
    if detector is None:
        return jsonify({
            'error': 'Model not loaded. Please restart the service.'
        }), 503
    
    try:
        # Parse request
        data = request.get_json()
        
        # Validate required fields
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        if 'review_text' not in data:
            return jsonify({'error': 'Missing required field: review_text'}), 400
        
        if 'rating' not in data:
            return jsonify({'error': 'Missing required field: rating'}), 400
        
        review_text = data['review_text']
        rating = data['rating']
        
        # Validate rating
        try:
            rating = float(rating)
            if not (1 <= rating <= 5):
                return jsonify({'error': 'Rating must be between 1 and 5'}), 400
        except (ValueError, TypeError):
            return jsonify({'error': 'Rating must be a number'}), 400
        
        # Optional fields
        reviewer_total_reviews = data.get('reviewer_total_reviews', 1)
        reviewer_avg_rating = data.get('reviewer_avg_rating', None)
        if reviewer_total_reviews is None:
            reviewer_total_reviews = 1
        
        # Make prediction
        result = detector.predict(
            review_text=review_text,
            rating=rating,
            reviewer_total_reviews=reviewer_total_reviews,
            reviewer_avg_rating=reviewer_avg_rating
        )
        
        logger.info(
            f"Prediction made: {result['risk_level']} risk "
            f"(fake_prob: {result['fake_probability']:.3f})"
        )
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in /predict: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict multiple reviews at once
    
    Request JSON:
    {
        "reviews": [
            {"review_text": "Great product!", "rating": 5},
            {"review_text": "Terrible quality", "rating": 1}
        ]
    }
    
    Response JSON:
    {
        "results": [
            {
                "review_text": "Great product!",
                "rating": 5,
                "is_fake": false,
                "fake_probability": 0.234,
                "risk_level": "LOW"
            },
            ...
        ],
        "summary": {
            "total": 2,
            "fake": 0,
            "legitimate": 2
        }
    }
    """
    if detector is None:
        return jsonify({
            'error': 'Model not loaded. Please restart the service.'
        }), 503
    
    try:
        # Parse request
        data = request.get_json()
        
        if not data or 'reviews' not in data:
            return jsonify({'error': 'Missing required field: reviews'}), 400
        
        reviews = data['reviews']
        
        if not isinstance(reviews, list):
            return jsonify({'error': 'reviews must be a list'}), 400
        
        if len(reviews) == 0:
            return jsonify({'error': 'reviews list is empty'}), 400
        
        if len(reviews) > 100:
            return jsonify({'error': 'Maximum 100 reviews per batch'}), 400
        
        # Make predictions
        results = []
        for review in reviews:
            if 'review_text' not in review or 'rating' not in review:
                results.append({
                    'error': 'Missing review_text or rating'
                })
                continue
            
            try:
                result = detector.predict(
                    review_text=review['review_text'],
                    rating=review['rating'],
                    reviewer_total_reviews=review.get('reviewer_total_reviews', 1),
                    reviewer_avg_rating=review.get('reviewer_avg_rating', None)
                )
                
                results.append({
                    'review_text': review['review_text'],
                    'rating': review['rating'],
                    **result
                })
            except Exception as e:
                results.append({
                    'review_text': review.get('review_text', ''),
                    'error': str(e)
                })
        
        # Calculate summary
        fake_count = sum(1 for r in results if r.get('is_fake', False))
        legitimate_count = len(results) - fake_count
        
        response = {
            'results': results,
            'summary': {
                'total': len(results),
                'fake': fake_count,
                'legitimate': legitimate_count,
                'fake_percentage': (fake_count / len(results) * 100) if results else 0
            }
        }
        
        logger.info(
            f"Batch prediction completed: {len(results)} reviews, "
            f"{fake_count} fake ({fake_count/len(results)*100:.1f}%)"
        )
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in /predict/batch: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("  ReviewGuard API Server")
    print("="*70)
    print(f"  Starting server on {API_HOST}:{API_PORT}")
    print(f"  Debug mode: {API_DEBUG}")
    print("="*70 + "\n")
    
    app.run(
         host="127.0.0.1",
        port=5000,
        debug=False
    )