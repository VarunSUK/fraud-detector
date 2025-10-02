#!/usr/bin/env python3
"""
Training script for fraud detection models.
Trains LightGBM and XGBoost models with class imbalance handling.
"""

import argparse
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_generator import TransactionGenerator
from models import FraudDetectionModel, train_ensemble_models
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train fraud detection models')
    parser.add_argument('--data-file', type=str, default='../creditcard.csv',
                       help='Path to training data CSV file')
    parser.add_argument('--generate-data', action='store_true',
                       help='Generate synthetic data if data file does not exist')
    parser.add_argument('--num-users', type=int, default=1000,
                       help='Number of users for synthetic data generation')
    parser.add_argument('--transactions-per-user', type=int, default=50,
                       help='Transactions per user for synthetic data generation')
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Directory to save trained models')
    parser.add_argument('--model-type', type=str, choices=['lightgbm', 'xgboost', 'ensemble'],
                       default='ensemble', help='Type of model to train')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Fraction of data to use for testing')
    parser.add_argument('--validation-size', type=float, default=0.2,
                       help='Fraction of training data to use for validation')
    parser.add_argument('--use-class-balance', action='store_true', default=True,
                       help='Use class balancing techniques')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Directory for training outputs and logs')
    parser.add_argument('--dataset-type', type=str, choices=['creditcard', 'synthetic'], 
                       default='creditcard', help='Type of dataset being used')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Sample size for training (useful for large datasets)')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Load or generate data
    if not os.path.exists(args.data_file) or args.generate_data:
        logger.info("Generating synthetic transaction data...")
        generator = TransactionGenerator(seed=42)
        df = generator.generate_transaction_stream(
            num_users=args.num_users,
            transactions_per_user=args.transactions_per_user
        )
        df.to_csv(args.data_file, index=False)
        logger.info(f"Generated {len(df)} transactions, saved to {args.data_file}")
        dataset_type = 'synthetic'
    else:
        logger.info(f"Loading data from {args.data_file}")
        df = pd.read_csv(args.data_file)
        
        # Sample data if specified (useful for large datasets)
        if args.sample_size and len(df) > args.sample_size:
            logger.info(f"Sampling {args.sample_size} rows from dataset")
            df = df.sample(n=args.sample_size, random_state=42)
        
        logger.info(f"Loaded {len(df)} transactions")
        dataset_type = args.dataset_type
    
    # Print data statistics
    if dataset_type == 'creditcard':
        fraud_rate = df['Class'].mean()
        logger.info(f"Fraud rate: {fraud_rate:.2%}")
        logger.info(f"Dataset info:")
        logger.info(f"  Total transactions: {len(df)}")
        logger.info(f"  Fraud transactions: {df['Class'].sum()}")
        logger.info(f"  Normal transactions: {len(df) - df['Class'].sum()}")
        
        # Prepare features and target for credit card dataset
        feature_cols = [f'V{i}' for i in range(1, 29)]  # V1-V28
        feature_cols.extend(['Time', 'Amount'])
        
        X = df[feature_cols].copy()
        y = df['Class'].copy()
        
    else:
        fraud_rate = df['is_fraud'].mean()
        logger.info(f"Fraud rate: {fraud_rate:.2%}")
        logger.info(f"Fraud types distribution:")
        fraud_types = df[df['is_fraud']]['fraud_type'].value_counts()
        for fraud_type, count in fraud_types.items():
            logger.info(f"  {fraud_type}: {count}")
        
        # Prepare features and target for synthetic dataset
        feature_cols = [
            'amount', 'merchant', 'card_type', 'hour', 'day_of_week', 'is_weekend',
            'previous_transactions', 'avg_amount', 'max_amount', 'location_country',
            'device_type'
        ]
        
        X = df[feature_cols].copy()
        y = df['is_fraud'].copy()
    
    logger.info(f"Feature columns: {feature_cols}")
    logger.info(f"Dataset shape: {X.shape}")
    
    # Train models
    training_start = datetime.now()
    
    if args.model_type == 'ensemble':
        logger.info("Training ensemble models (LightGBM + XGBoost)...")
        results = train_ensemble_models(
            X, y, 
            models_dir=args.models_dir,
            dataset_type=dataset_type
        )
        
        # Save training results
        results_file = os.path.join(args.output_dir, 'training_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Training results saved to {results_file}")
        
    else:
        logger.info(f"Training {args.model_type} model...")
        model = FraudDetectionModel(
            model_type=args.model_type,
            use_class_balance=args.use_class_balance,
            dataset_type=dataset_type
        )
        
        metrics = model.train(
            X, y,
            test_size=args.test_size,
            validation_size=args.validation_size
        )
        
        # Save model
        model_path = os.path.join(args.models_dir, args.model_type)
        model.save_model(model_path)
        
        # Save results
        results = {args.model_type: metrics}
        results_file = os.path.join(args.output_dir, 'training_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    training_end = datetime.now()
    training_duration = training_end - training_start
    
    logger.info(f"Training completed in {training_duration}")
    
    # Print summary
    logger.info("=== TRAINING SUMMARY ===")
    if args.model_type == 'ensemble':
        logger.info(f"LightGBM AUC: {results['lightgbm']['auc']:.4f}")
        logger.info(f"XGBoost AUC: {results['xgboost']['auc']:.4f}")
        logger.info(f"Ensemble AUC: {results['ensemble']['auc']:.4f}")
    else:
        logger.info(f"{args.model_type.upper()} AUC: {results[args.model_type]['auc']:.4f}")
        logger.info(f"{args.model_type.upper()} AP: {results[args.model_type]['average_precision']:.4f}")
    
    # Save training metadata
    metadata = {
        'training_start': training_start.isoformat(),
        'training_end': training_end.isoformat(),
        'training_duration_seconds': training_duration.total_seconds(),
        'data_file': args.data_file,
        'num_samples': len(df),
        'fraud_rate': fraud_rate,
        'model_type': args.model_type,
        'feature_columns': feature_cols,
        'args': vars(args)
    }
    
    metadata_file = os.path.join(args.output_dir, 'training_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Training metadata saved to {metadata_file}")
    
    # Feature importance analysis
    if args.model_type != 'ensemble':
        model = FraudDetectionModel(model_type=args.model_type)
        model.load_model(os.path.join(args.models_dir, args.model_type))
        
        if model.feature_importance:
            logger.info("Top 10 most important features:")
            sorted_features = sorted(
                model.feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            for i, (feature, importance) in enumerate(sorted_features[:10]):
                logger.info(f"  {i+1}. {feature}: {importance:.4f}")
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()


