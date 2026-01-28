"""
Test script for ReviewGuard API
"""
import requests
import json

# API base URL
BASE_URL = "http://localhost:5000"


def test_health():
    """Test health check endpoint"""
    print("\n" + "="*70)
    print("Testing /health endpoint")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_single_prediction():
    """Test single review prediction"""
    print("\n" + "="*70)
    print("Testing /predict endpoint")
    print("="*70)
    
    # Test cases
    test_reviews = [
        {
            "name": "Suspicious Review (High rating, generic text)",
            "review_text": "This product is absolutely amazing! Best purchase ever! 5 stars!",
            "rating": 5
        },
        {
            "name": "Legitimate Review (Detailed, balanced)",
            "review_text": "The product quality is good and delivery was on time. "
                          "It works as described in the listing. The packaging could be better "
                          "but overall I'm satisfied with the purchase. Would recommend.",
            "rating": 4
        },
        {
            "name": "Suspicious Review (Extreme negative)",
            "review_text": "Terrible! Worst product ever! Complete scam! Don't buy!",
            "rating": 1
        }
    ]
    
    for test in test_reviews:
        print(f"\n{test['name']}")
        print("-" * 70)
        print(f"Text: {test['review_text'][:60]}...")
        print(f"Rating: {test['rating']}")
        
        payload = {
            "review_text": test['review_text'],
            "rating": test['rating']
        }
        
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"\nStatus Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Is Fake: {result['is_fake']}")
            print(f"Fake Probability: {result['fake_probability']:.3f}")
            print(f"Risk Level: {result['risk_level']}")
        else:
            print(f"Error: {response.json()}")


def test_batch_prediction():
    """Test batch prediction"""
    print("\n" + "="*70)
    print("Testing /predict/batch endpoint")
    print("="*70)
    
    payload = {
        "reviews": [
            {
                "review_text": "Great product! Highly recommend!",
                "rating": 5
            },
            {
                "review_text": "Good quality for the price. Works well.",
                "rating": 4
            },
            {
                "review_text": "Terrible quality! Broke after one day!",
                "rating": 1
            }
        ]
    }
    
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"\nSummary:")
        print(f"  Total: {result['summary']['total']}")
        print(f"  Fake: {result['summary']['fake']}")
        print(f"  Legitimate: {result['summary']['legitimate']}")
        print(f"  Fake %: {result['summary']['fake_percentage']:.1f}%")
        
        print(f"\nIndividual Results:")
        for i, res in enumerate(result['results'], 1):
            print(f"  {i}. {res['review_text'][:40]}...")
            print(f"     Rating: {res['rating']}, "
                  f"Fake: {res['is_fake']}, "
                  f"Risk: {res['risk_level']}")
    else:
        print(f"Error: {response.json()}")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("  REVIEWGUARD API TESTING")
    print("="*70)
    print("  Make sure the API server is running: python app.py")
    print("="*70)
    
    try:
        # Test endpoints
        if test_health():
            test_single_prediction()
            test_batch_prediction()
        else:
            print("\n❌ Health check failed. Is the API server running?")
        
        print("\n" + "="*70)
        print("✅ Testing completed")
        print("="*70)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to API server.")
        print("   Please start the server first: python app.py")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")


if __name__ == "__main__":
    main()