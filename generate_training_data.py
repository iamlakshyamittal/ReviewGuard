"""
Advanced Training Data Generator for ReviewGuard
Creates highly realistic fake and legitimate review patterns
"""
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# ===== FAKE REVIEW PATTERNS =====

FAKE_POSITIVE_TEMPLATES = [
    "AMAZING product!!! This is the BEST thing I've EVER bought!!! Must buy NOW!!!",
    "Incredible! Life changing! Can't believe how good this is! Five stars!!!",
    "Perfect in every way! Best purchase ever! Everyone needs this! Buy it!",
    "Outstanding quality! Exceeded all expectations! Highly recommend! A+++++",
    "Absolutely FANTASTIC! Worth every penny! Don't hesitate! Order today!",
    "WOW! Just WOW! This is incredible! Best product on the market!",
    "Excellent! Superior! Amazing! Perfect! Five stars all the way!",
    "Blown away by the quality! Superb! Magnificent! Get it now!",
    "This product is PERFECT! No flaws whatsoever! Must have item!",
    "Best thing ever created! Amazing quality! Super fast shipping! Love it!",
]

FAKE_NEGATIVE_TEMPLATES = [
    "TERRIBLE! Worst product EVER! Complete SCAM! Don't waste your money!",
    "Absolute garbage! Broke immediately! Total ripoff! Avoid at all costs!",
    "Horrible quality! Doesn't work! Waste of money! Save yourself!",
    "Awful! Cheap junk! Fell apart! Don't buy! Total disaster!",
    "Worst purchase ever! Defective! Scam! Demand refund immediately!",
    "Disgusting quality! Broken on arrival! Fraudulent seller! Stay away!",
    "Pathetic! Useless! Worst ever! Total waste! One star is too generous!",
    "Terrible experience! Product is junk! Complete failure! Avoid!",
    "Awful product! Doesn't work as advertised! Scam! Don't trust!",
    "Horrible! Cheap materials! Broke after one use! Total garbage!",
]

# ===== LEGITIMATE REVIEW PATTERNS =====

LEGIT_POSITIVE_TEMPLATES = [
    "Good product overall. Works as described in the listing. Delivery was on time and packaging was adequate. Would recommend for the price.",
    "Satisfied with this purchase. The quality meets my expectations and it functions properly. Setup was straightforward. Fair value for money.",
    "Decent item. Does what it's supposed to do. A few minor issues but nothing major. Generally happy with the purchase.",
    "Pretty good quality. Installation was easy enough. Works well for my needs. Reasonably priced compared to alternatives.",
    "Happy with this product. It arrived in good condition and works as expected. Customer service was helpful when I had questions.",
    "This meets my requirements. The build quality is solid and it performs reliably. Good value considering the features offered.",
    "Pleased with my purchase. It's well-made and functional. The instructions could be clearer but I figured it out. Worth the money.",
    "Good product for everyday use. Nothing fancy but it gets the job done. Shipping was prompt. Would buy again.",
]

LEGIT_NEUTRAL_TEMPLATES = [
    "It's okay. Does what it says on the box. Nothing special but nothing terrible either. Average product at an average price.",
    "Acceptable quality. Works fine for basic use. Some features could be better but it's functional. Fair for the cost.",
    "Decent but not amazing. It works adequately. A few design flaws but manageable. Okay value.",
    "Average product. Neither impressed nor disappointed. Functions as expected with minor quirks. Reasonable price point.",
    "It's alright. Gets the job done though not perfectly. Some room for improvement. Decent for the money.",
    "Mediocre quality but functional. Does the basics okay. Could be better engineered. Fair price.",
]

LEGIT_CRITICAL_TEMPLATES = [
    "Disappointed with this purchase. The quality isn't as good as advertised. It works but not as well as expected. Might return it.",
    "Not impressed. Several issues right out of the box. Functions but feels cheaply made. Expected more for the price.",
    "Below expectations. The product works but has some significant flaws. Customer service was slow to respond. Not sure I'd recommend.",
    "Quality is lacking. It does function but there are annoying problems. Instructions were unclear. Overpriced for what you get.",
    "Underwhelmed by this. Works intermittently. Build quality could be much better. Shipping was delayed. Two stars is generous.",
]

# ===== HELPER FUNCTIONS =====

def add_typos(text, num_typos=0):
    """Add realistic typos to text"""
    if num_typos == 0:
        return text
    
    words = text.split()
    typo_positions = random.sample(range(len(words)), min(num_typos, len(words)))
    
    for pos in typo_positions:
        word = words[pos]
        if len(word) > 3:
            # Swap two adjacent letters
            swap_pos = random.randint(0, len(word) - 2)
            word_list = list(word)
            word_list[swap_pos], word_list[swap_pos + 1] = word_list[swap_pos + 1], word_list[swap_pos]
            words[pos] = ''.join(word_list)
    
    return ' '.join(words)

def add_variations(text):
    """Add natural variations to text"""
    variations = [
        lambda t: t,  # No change
        lambda t: t.replace('!', '.'),  # Less enthusiasm
        lambda t: t + " Thanks!",
        lambda t: t + " Hope this helps.",
        lambda t: "Update: " + t,
        lambda t: t.replace('.', '...'),  # Ellipsis
    ]
    return random.choice(variations)(text)

# ===== FAKE REVIEW GENERATOR =====

def generate_fake_reviews(n=600):
    """Generate realistic fake review patterns"""
    reviews = []
    
    for i in range(n):
        # Decide if positive or negative fake
        is_positive = random.random() > 0.35  # 65% positive, 35% negative
        
        if is_positive:
            # Fake positive review
            base_text = random.choice(FAKE_POSITIVE_TEMPLATES)
            rating = 5
            
            # Fake positive reviewers characteristics
            total_reviews = random.randint(3, 20)  # Suspicious range
            
            # Very high average (always 5 stars)
            if random.random() > 0.3:
                avg_rating = random.uniform(4.85, 5.0)  # Almost always 5
            else:
                avg_rating = random.uniform(4.6, 4.85)  # Slightly varied
        else:
            # Fake negative review
            base_text = random.choice(FAKE_NEGATIVE_TEMPLATES)
            rating = 1
            
            # Fake negative reviewers characteristics
            total_reviews = random.randint(3, 18)
            
            # Very low average (always 1 star)
            if random.random() > 0.3:
                avg_rating = random.uniform(1.0, 1.2)
            else:
                avg_rating = random.uniform(1.2, 1.5)
        
        # Sometimes add minor variations
        if random.random() > 0.7:
            base_text = add_variations(base_text)
        
        reviews.append({
            'review_text': base_text,
            'rating': rating,
            'total_reviews': total_reviews,
            'avg_rating': avg_rating,
            'label': 1  # 1 = fake
        })
    
    return reviews

# ===== LEGITIMATE REVIEW GENERATOR =====

def generate_legit_reviews(n=600):
    """Generate realistic legitimate review patterns"""
    reviews = []
    
    for i in range(n):
        # Legitimate reviewers have varied ratings
        rating_weights = [0.05, 0.15, 0.25, 0.35, 0.20]  # 1-5 stars distribution
        rating = random.choices([1, 2, 3, 4, 5], weights=rating_weights)[0]
        
        # Select template based on rating
        if rating >= 4:
            base_text = random.choice(LEGIT_POSITIVE_TEMPLATES)
        elif rating == 3:
            base_text = random.choice(LEGIT_NEUTRAL_TEMPLATES)
        else:
            base_text = random.choice(LEGIT_CRITICAL_TEMPLATES)
        
        # Legitimate reviewers characteristics
        # More established accounts
        total_reviews = random.randint(15, 200)
        
        # Balanced average ratings
        avg_rating = random.uniform(3.0, 4.3)
        
        # Add natural variations
        if random.random() > 0.5:
            base_text = add_variations(base_text)
        
        # Occasionally add minor typos (humans make mistakes)
        if random.random() > 0.85:
            base_text = add_typos(base_text, num_typos=1)
        
        reviews.append({
            'review_text': base_text,
            'rating': rating,
            'total_reviews': total_reviews,
            'avg_rating': avg_rating,
            'label': 0  # 0 = legitimate
        })
    
    return reviews

# ===== EDGE CASES GENERATOR =====

def generate_edge_cases(n=100):
    """Generate tricky edge cases for better model robustness"""
    reviews = []
    
    edge_case_patterns = [
        # New legitimate users (low review count but legit)
        {
            'text': random.choice(LEGIT_POSITIVE_TEMPLATES),
            'rating': 4,
            'total_reviews': random.randint(1, 5),
            'avg_rating': random.uniform(3.5, 4.2),
            'label': 0
        },
        # Experienced fake reviewers (high count but fake patterns)
        {
            'text': random.choice(FAKE_POSITIVE_TEMPLATES),
            'rating': 5,
            'total_reviews': random.randint(25, 50),
            'avg_rating': random.uniform(4.8, 5.0),
            'label': 1
        },
        # Subtle fake (less obvious language)
        {
            'text': "Great product. Very satisfied. Highly recommend. Five stars.",
            'rating': 5,
            'total_reviews': random.randint(8, 15),
            'avg_rating': random.uniform(4.7, 4.95),
            'label': 1
        },
        # Critical but legitimate
        {
            'text': random.choice(LEGIT_CRITICAL_TEMPLATES),
            'rating': 2,
            'total_reviews': random.randint(30, 100),
            'avg_rating': random.uniform(3.2, 3.9),
            'label': 0
        },
    ]
    
    for i in range(n):
        pattern = random.choice(edge_case_patterns)
        reviews.append({
            'review_text': pattern['text'],
            'rating': pattern['rating'],
            'total_reviews': pattern['total_reviews'],
            'avg_rating': pattern['avg_rating'],
            'label': pattern['label']
        })
    
    return reviews

# ===== MAIN FUNCTION =====

def main():
    """Generate comprehensive training dataset"""
    print("\n" + "="*70)
    print("ADVANCED TRAINING DATA GENERATOR FOR REVIEWGUARD")
    print("="*70)
    print("\nGenerating realistic review patterns...")
    
    # Generate different types of reviews
    print("  → Generating fake reviews...")
    fake_reviews = generate_fake_reviews(600)
    
    print("  → Generating legitimate reviews...")
    legit_reviews = generate_legit_reviews(600)
    
    print("  → Generating edge cases...")
    edge_cases = generate_edge_cases(100)
    
    # Combine all reviews
    all_reviews = fake_reviews + legit_reviews + edge_cases
    df = pd.DataFrame(all_reviews)
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv('train_reviews.csv', index=False)
    
    # Print statistics
    print("\n" + "="*70)
    print("✅ DATASET CREATED SUCCESSFULLY")
    print("="*70)
    
    print(f"\nTotal Reviews: {len(df):,}")
    print(f"  • Fake: {df['label'].sum():,} ({df['label'].sum()/len(df)*100:.1f}%)")
    print(f"  • Legitimate: {(df['label']==0).sum():,} ({(df['label']==0).sum()/len(df)*100:.1f}%)")
    
    print("\nRating Distribution:")
    for rating in sorted(df['rating'].unique()):
        count = (df['rating'] == rating).sum()
        print(f"  • {rating}★: {count:,} ({count/len(df)*100:.1f}%)")
    
    print("\nReviewer Statistics:")
    print(f"  • Avg Total Reviews: {df['total_reviews'].mean():.1f}")
    print(f"  • Avg Rating: {df['avg_rating'].mean():.2f}")
    
    print("\n" + "="*70)
    print("SAMPLE REVIEWS")
    print("="*70)
    
    print("\n📌 Fake Review Example:")
    fake_sample = df[df['label'] == 1].iloc[0]
    print(f"  Text: {fake_sample['review_text'][:70]}...")
    print(f"  Rating: {fake_sample['rating']}★")
    print(f"  Total Reviews: {fake_sample['total_reviews']}")
    print(f"  Avg Rating: {fake_sample['avg_rating']:.2f}★")
    
    print("\n✅ Legitimate Review Example:")
    legit_sample = df[df['label'] == 0].iloc[0]
    print(f"  Text: {legit_sample['review_text'][:70]}...")
    print(f"  Rating: {legit_sample['rating']}★")
    print(f"  Total Reviews: {legit_sample['total_reviews']}")
    print(f"  Avg Rating: {legit_sample['avg_rating']:.2f}★")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("  1. Run: python train_model.py")
    print("  2. Run: python app.py")
    print("  3. Test in browser with index.html")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()