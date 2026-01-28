"""
ReviewGuard - Main Pipeline
Orchestrates the complete fake review detection pipeline
"""
import sys
import argparse
from datetime import datetime
from src.preprocess import preprocess_reviews
from src.feature_engineering import create_features
from src.network_analysis import build_reviewer_network
from src.anomaly_detection import detect_fake_reviewers
from src.train_classifier import train_fake_review_detector
from src.utils import setup_logger, create_directories, print_section_header

logger = setup_logger(__name__)


def run_full_pipeline():
    """
    Run the complete ReviewGuard pipeline
    
    Steps:
    1. Preprocess raw reviews
    2. Engineer features
    3. Analyze reviewer network
    4. Detect anomalies
    5. Train supervised model
    """
    start_time = datetime.now()
    
    print("\n" + "="*70)
    print("  🚀 REVIEWGUARD - FAKE REVIEW DETECTION SYSTEM")
    print("="*70)
    print(f"  Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    try:
        # Create necessary directories
        create_directories()
        
        # Step 1: Preprocess
        logger.info("Starting pipeline step 1/5: Preprocessing")
        preprocess_reviews()
        
        # Step 2: Feature Engineering
        logger.info("Starting pipeline step 2/5: Feature Engineering")
        create_features()
        
        # Step 3: Network Analysis
        logger.info("Starting pipeline step 3/5: Network Analysis")
        build_reviewer_network()
        
        # Step 4: Anomaly Detection
        logger.info("Starting pipeline step 4/5: Anomaly Detection")
        detect_fake_reviewers()
        
        # Step 5: Train Classifier
        logger.info("Starting pipeline step 5/5: Train Classifier")
        train_fake_review_detector()
        
        # Complete
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print_section_header("PIPELINE COMPLETED SUCCESSFULLY")
        print(f"  Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print("="*70)
        
        print("\n✅ ReviewGuard pipeline completed successfully!")
        print("\n📊 Next steps:")
        print("  1. Review results in data/suspicious_reviewers.csv")
        print("  2. Check model performance in models/feature_importance.csv")
        print("  3. Start API server: python app.py")
        print("  4. Test predictions: python src/predict.py")
        
        return True
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        print("\n\n⚠️  Pipeline interrupted by user (Ctrl+C)")
        return False
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"\n\n❌ Pipeline failed with error: {e}")
        print("Check logs for details.")
        return False


def run_preprocessing_only():
    """Run only preprocessing step"""
    print_section_header("PREPROCESSING ONLY")
    try:
        preprocess_reviews()
        print("\n✅ Preprocessing completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return False


def run_training_only():
    """Run only training step (assumes features exist)"""
    print_section_header("TRAINING ONLY")
    try:
        train_fake_review_detector()
        print("\n✅ Training completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False


def main():
    """Main entry point with command-line arguments"""
    parser = argparse.ArgumentParser(
        description='ReviewGuard - Fake Review Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python main.py
  
  # Run only preprocessing
  python main.py --preprocess-only
  
  # Run only training (requires preprocessed data)
  python main.py --train-only
        """
    )
    
    parser.add_argument(
        '--preprocess-only',
        action='store_true',
        help='Run only the preprocessing step'
    )
    
    parser.add_argument(
        '--train-only',
        action='store_true',
        help='Run only the training step (requires preprocessed features)'
    )
    
    args = parser.parse_args()
    
    # Execute based on arguments
    if args.preprocess_only:
        success = run_preprocessing_only()
    elif args.train_only:
        success = run_training_only()
    else:
        success = run_full_pipeline()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()