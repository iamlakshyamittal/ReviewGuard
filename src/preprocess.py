"""
Data preprocessing module for ReviewGuard
"""
import pandas as pd
import numpy as np
from config.settings import RAW_DATA_PATH, PROCESSED_DATA_PATH, MIN_REVIEW_LENGTH
from src.utils import setup_logger, print_section_header

logger = setup_logger(__name__)


def preprocess_reviews(input_path=RAW_DATA_PATH, output_path=PROCESSED_DATA_PATH):
    """
    Preprocess raw review data
    
    Steps:
    1. Load raw data
    2. Select required columns
    3. Clean and validate data types
    4. Remove invalid/missing data
    5. Filter out very short reviews
    6. Save cleaned data
    
    Args:
        input_path: Path to raw CSV file
        output_path: Path to save processed CSV
    
    Returns:
        pd.DataFrame: Processed reviews
    """
    print_section_header("PREPROCESSING RAW REVIEWS")
    logger.info(f"Loading raw data from {input_path}")
    
    try:
        # Read raw data
        df = pd.read_csv(input_path)
        initial_rows = len(df)
        logger.info(f"Loaded {initial_rows:,} raw reviews")
        
        # Select required columns
        required_cols = ["reviewerID", "productID", "reviewText", "rating", "reviewTime"]
        
        # Check if all required columns exist
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        df = df[required_cols].copy()
        logger.info(f"Selected {len(required_cols)} required columns")
        
        # Clean data types
        df["reviewText"] = df["reviewText"].astype(str)
        df["reviewerID"] = df["reviewerID"].astype(str)
        df["productID"] = df["productID"].astype(str)
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df["reviewTime"] = pd.to_datetime(df["reviewTime"], errors="coerce")
        
        # Remove rows with missing critical data
        df.dropna(subset=["reviewerID", "productID", "reviewText", "rating"], inplace=True)
        logger.info(f"Removed {initial_rows - len(df):,} rows with missing data")
        
        # Filter out very short reviews (likely spam or invalid)
        df = df[df["reviewText"].str.len() >= MIN_REVIEW_LENGTH].copy()
        logger.info(f"Removed reviews shorter than {MIN_REVIEW_LENGTH} characters")
        
        # Filter valid ratings (1-5)
        df = df[(df["rating"] >= 1) & (df["rating"] <= 5)].copy()
        
        # Remove duplicates
        duplicates = df.duplicated(subset=["reviewerID", "productID", "reviewText"]).sum()
        df.drop_duplicates(subset=["reviewerID", "productID", "reviewText"], inplace=True)
        logger.info(f"Removed {duplicates:,} duplicate reviews")
        
        # Sort by time
        df.sort_values("reviewTime", inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Save processed data
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df):,} processed reviews to {output_path}")
        
        # Print summary statistics
        print(f"\n✅ Preprocessing completed successfully")
        print(f"   Initial reviews: {initial_rows:,}")
        print(f"   Final reviews: {len(df):,}")
        print(f"   Removed: {initial_rows - len(df):,} ({(initial_rows - len(df))/initial_rows*100:.1f}%)")
        print(f"   Unique reviewers: {df['reviewerID'].nunique():,}")
        print(f"   Unique products: {df['productID'].nunique():,}")
        print(f"   Average rating: {df['rating'].mean():.2f}")
        print(f"   Date range: {df['reviewTime'].min()} to {df['reviewTime'].max()}")
        
        return df
        
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
        raise
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise


if __name__ == "__main__":
    preprocess_reviews()