"""
ML models for fraud detection with LightGBM and XGBoost.
Includes class imbalance handling and feature engineering.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import json
import os
from typing import Dict, List, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for fraud detection."""
    
    def __init__(self, dataset_type: str = 'creditcard'):
        self.dataset_type = dataset_type
        self.label_encoders = {}
        self.scalers = {}
        self.feature_names = []
        
    def encode_categorical_features(self, df: pd.DataFrame, 
                                  categorical_cols: List[str],
                                  fit: bool = True) -> pd.DataFrame:
        """Encode categorical features using label encoding."""
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if fit:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                df_encoded[f"{col}_encoded"] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    unique_values = set(df_encoded[col].astype(str).unique())
                    known_values = set(self.label_encoders[col].classes_)
                    unknown_values = unique_values - known_values
                    
                    if unknown_values:
                        logger.warning(f"Unknown categories in {col}: {unknown_values}")
                        # Map unknown categories to a default value
                        df_encoded[col] = df_encoded[col].astype(str).replace(
                            dict.fromkeys(unknown_values, 'unknown')
                        )
                    
                    df_encoded[f"{col}_encoded"] = self.label_encoders[col].transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features for better fraud detection."""
        df_features = df.copy()
        
        if self.dataset_type == 'creditcard':
            # For credit card dataset (PCA features + Time + Amount)
            # Amount-based features
            df_features['amount_log'] = np.log1p(df_features['Amount'])
            df_features['amount_sqrt'] = np.sqrt(df_features['Amount'])
            df_features['amount_percentile'] = df_features['Amount'].rank(pct=True)
            
            # Time-based features (convert seconds to hours and days)
            df_features['hour'] = (df_features['Time'] / 3600) % 24
            df_features['day'] = df_features['Time'] / (3600 * 24)
            df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
            df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
            df_features['day_sin'] = np.sin(2 * np.pi * df_features['day'] / 7)
            df_features['day_cos'] = np.cos(2 * np.pi * df_features['day'] / 7)
            df_features['is_night'] = (df_features['hour'] < 6) | (df_features['hour'] > 22)
            df_features['is_business_hours'] = (df_features['hour'] >= 9) & (df_features['hour'] <= 17)
            
            # PCA feature combinations
            pca_features = [f'V{i}' for i in range(1, 29)]
            df_features['pca_sum'] = df_features[pca_features].sum(axis=1)
            df_features['pca_mean'] = df_features[pca_features].mean(axis=1)
            df_features['pca_std'] = df_features[pca_features].std(axis=1)
            df_features['pca_max'] = df_features[pca_features].max(axis=1)
            df_features['pca_min'] = df_features[pca_features].min(axis=1)
            
            # Amount-PCA interactions
            df_features['amount_x_pca_sum'] = df_features['Amount'] * df_features['pca_sum']
            df_features['amount_x_pca_mean'] = df_features['Amount'] * df_features['pca_mean']
            
            # Statistical features
            df_features['amount_zscore'] = (df_features['Amount'] - df_features['Amount'].mean()) / df_features['Amount'].std()
            
        else:
            # For synthetic dataset
            # Amount-based features
            df_features['amount_log'] = np.log1p(df_features['amount'])
            df_features['amount_sqrt'] = np.sqrt(df_features['amount'])
            df_features['amount_to_avg_ratio'] = df_features['amount'] / (df_features['avg_amount'] + 1)
            df_features['amount_to_max_ratio'] = df_features['amount'] / (df_features['max_amount'] + 1)
            df_features['amount_percentile'] = df_features['amount'].rank(pct=True)
            
            # Time-based features
            df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
            df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
            df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
            df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
            df_features['is_night'] = (df_features['hour'] < 6) | (df_features['hour'] > 22)
            df_features['is_business_hours'] = (df_features['hour'] >= 9) & (df_features['hour'] <= 17)
            
            # Interaction features
            df_features['amount_x_hour'] = df_features['amount'] * df_features['hour']
            df_features['amount_x_weekend'] = df_features['amount'] * df_features['is_weekend']
            df_features['avg_amount_x_hour'] = df_features['avg_amount'] * df_features['hour']
            
            # Statistical features
            df_features['amount_zscore'] = (df_features['amount'] - df_features['amount'].mean()) / df_features['amount'].std()
            df_features['previous_txn_rate'] = df_features['previous_transactions'] / 100  # Normalize
        
        return df_features
    
    def prepare_features(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Prepare features for model training/inference."""
        
        # Create derived features
        df_features = self.create_derived_features(df)
        
        if self.dataset_type == 'creditcard':
            # For credit card dataset
            # Select all PCA features + derived features
            feature_cols = [f'V{i}' for i in range(1, 29)]  # V1-V28
            feature_cols.extend([
                'Time', 'Amount', 'amount_log', 'amount_sqrt', 'amount_percentile',
                'hour', 'day', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                'is_night', 'is_business_hours', 'pca_sum', 'pca_mean', 'pca_std',
                'pca_max', 'pca_min', 'amount_x_pca_sum', 'amount_x_pca_mean', 'amount_zscore'
            ])
        else:
            # For synthetic dataset
            categorical_cols = ['merchant', 'card_type', 'location_country', 'device_type']
            
            # Encode categorical features
            df_features = self.encode_categorical_features(df_features, categorical_cols, fit)
            
            # Select features for model
            feature_cols = [
                'amount', 'amount_log', 'amount_sqrt', 'amount_to_avg_ratio', 'amount_to_max_ratio',
                'amount_percentile', 'hour', 'hour_sin', 'hour_cos', 'day_of_week', 'day_sin', 'day_cos',
                'is_weekend', 'is_night', 'is_business_hours', 'previous_transactions', 'avg_amount',
                'max_amount', 'amount_x_hour', 'amount_x_weekend', 'avg_amount_x_hour',
                'amount_zscore', 'previous_txn_rate'
            ]
            
            # Add encoded categorical features
            for col in categorical_cols:
                feature_cols.append(f"{col}_encoded")
        
        # Store feature names
        if fit:
            self.feature_names = feature_cols
        
        # Ensure all features exist
        missing_features = set(feature_cols) - set(df_features.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            for feature in missing_features:
                df_features[feature] = 0
        
        # Select and return features
        X = df_features[feature_cols].fillna(0).values
        
        # Scale numerical features if needed
        if fit:
            self.scalers['standard'] = StandardScaler()
            X = self.scalers['standard'].fit_transform(X)
        else:
            if 'standard' in self.scalers:
                X = self.scalers['standard'].transform(X)
        
        return X
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for model interpretation."""
        return self.feature_names
    
    def save(self, filepath: str):
        """Save feature engineer to disk."""
        joblib.dump({
            'label_encoders': self.label_encoders,
            'scalers': self.scalers,
            'feature_names': self.feature_names
        }, filepath)
    
    def load(self, filepath: str):
        """Load feature engineer from disk."""
        data = joblib.load(filepath)
        self.label_encoders = data['label_encoders']
        self.scalers = data['scalers']
        self.feature_names = data['feature_names']


class FraudDetectionModel:
    """Fraud detection model with LightGBM and XGBoost."""
    
    def __init__(self, model_type: str = 'lightgbm', use_class_balance: bool = True, dataset_type: str = 'creditcard'):
        self.model_type = model_type
        self.use_class_balance = use_class_balance
        self.dataset_type = dataset_type
        self.model = None
        self.feature_engineer = FeatureEngineer(dataset_type=dataset_type)
        self.feature_importance = None
        self.metrics = {}
        
    def _get_model(self, **params):
        """Get model instance based on type."""
        if self.model_type == 'lightgbm':
            return lgb.LGBMClassifier(**params)
        elif self.model_type == 'xgboost':
            return xgb.XGBClassifier(**params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _get_default_params(self):
        """Get default parameters for the model."""
        if self.model_type == 'lightgbm':
            return {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42,
                'class_weight': 'balanced' if self.use_class_balance else None,
                'n_estimators': 1000,
                'early_stopping_rounds': 50
            }
        elif self.model_type == 'xgboost':
            return {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'scale_pos_weight': None,  # Will be calculated if needed
                'n_estimators': 1000,
                'early_stopping_rounds': 50
            }
    
    def _calculate_class_weights(self, y: np.ndarray) -> Dict:
        """Calculate class weights for imbalanced data."""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        class_weights = compute_class_weight(
            'balanced',
            classes=classes,
            y=y
        )
        return dict(zip(classes, class_weights))
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              test_size: float = 0.2, validation_size: float = 0.2) -> Dict:
        """Train the fraud detection model."""
        
        logger.info(f"Training {self.model_type} model...")
        logger.info(f"Dataset shape: {X.shape}")
        logger.info(f"Fraud rate: {y.mean():.2%}")
        
        # Prepare features
        X_processed = self.feature_engineer.prepare_features(X, fit=True)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=validation_size, random_state=42, stratify=y_temp
        )
        
        logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Handle class imbalance
        if self.use_class_balance and self.model_type == 'xgboost':
            class_weights = self._calculate_class_weights(y_train)
            scale_pos_weight = class_weights[1] / class_weights[0]
            
            # Update params with calculated weight
            params = self._get_default_params()
            params['scale_pos_weight'] = scale_pos_weight
            
            logger.info(f"Using scale_pos_weight: {scale_pos_weight:.2f}")
        else:
            params = self._get_default_params()
        
        # Create model
        self.model = self._get_model(**params)
        
        # Train with early stopping
        if self.model_type == 'lightgbm':
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
        elif self.model_type == 'xgboost':
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        
        # Get feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                self.feature_engineer.get_feature_names(),
                self.model.feature_importances_
            ))
        
        # Evaluate on test set
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        self.metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        logger.info(f"Test AUC: {self.metrics['auc']:.4f}")
        logger.info(f"Test AP: {self.metrics['average_precision']:.4f}")
        
        return self.metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_pred_proba: np.ndarray) -> Dict:
        """Calculate comprehensive metrics."""
        metrics = {
            'auc': roc_auc_score(y_true, y_pred_proba),
            'average_precision': average_precision_score(y_true, y_pred_proba),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = {
            'tn': int(cm[0, 0]), 'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]), 'tp': int(cm[1, 1])
        }
        
        # Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
        metrics['precision_recall_curve'] = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': pr_thresholds.tolist()
        }
        
        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
        metrics['roc_curve'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': roc_thresholds.tolist()
        }
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_processed = self.feature_engineer.prepare_features(X, fit=False)
        
        y_pred_proba = self.model.predict_proba(X_processed)[:, 1]
        y_pred = self.model.predict(X_processed)
        
        return y_pred, y_pred_proba
    
    def explain_prediction(self, X: pd.DataFrame, instance_idx: int = 0) -> Dict:
        """Explain prediction for a specific instance."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_processed = self.feature_engineer.prepare_features(X, fit=False)
        
        # Get prediction
        y_pred_proba = self.model.predict_proba(X_processed[instance_idx:instance_idx+1])[0, 1]
        
        # Get feature importance
        if self.feature_importance is None:
            return {"prediction": y_pred_proba, "explanation": "Feature importance not available"}
        
        # Create explanation
        feature_names = self.feature_engineer.get_feature_names()
        feature_values = X_processed[instance_idx]
        
        explanation = {
            "prediction": float(y_pred_proba),
            "feature_contributions": []
        }
        
        # Simple feature contribution (feature importance * normalized feature value)
        for i, (name, importance) in enumerate(self.feature_importance.items()):
            if i < len(feature_values):
                contribution = importance * abs(feature_values[i])
                explanation["feature_contributions"].append({
                    "feature": name,
                    "value": float(feature_values[i]),
                    "importance": float(importance),
                    "contribution": float(contribution)
                })
        
        # Sort by contribution
        explanation["feature_contributions"].sort(
            key=lambda x: x["contribution"], reverse=True
        )
        
        return explanation
    
    def save_model(self, filepath: str):
        """Save model and feature engineer to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        joblib.dump(self.model, f"{filepath}_model.joblib")
        
        # Save feature engineer
        self.feature_engineer.save(f"{filepath}_features.joblib")
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'use_class_balance': self.use_class_balance,
            'feature_importance': self.feature_importance,
            'metrics': self.metrics,
            'feature_names': self.feature_engineer.get_feature_names()
        }
        
        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model and feature engineer from disk."""
        # Load model
        self.model = joblib.load(f"{filepath}_model.joblib")
        
        # Load feature engineer
        self.feature_engineer.load(f"{filepath}_features.joblib")
        
        # Load metadata
        with open(f"{filepath}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.model_type = metadata['model_type']
        self.use_class_balance = metadata['use_class_balance']
        self.feature_importance = metadata['feature_importance']
        self.metrics = metadata['metrics']
        
        logger.info(f"Model loaded from {filepath}")


def train_ensemble_models(X: pd.DataFrame, y: pd.Series, 
                         models_dir: str = "models", dataset_type: str = 'creditcard') -> Dict:
    """Train ensemble of LightGBM and XGBoost models."""
    
    os.makedirs(models_dir, exist_ok=True)
    
    results = {}
    
    # Train LightGBM model
    logger.info("Training LightGBM model...")
    lgb_model = FraudDetectionModel(model_type='lightgbm', use_class_balance=True, dataset_type=dataset_type)
    lgb_metrics = lgb_model.train(X, y)
    lgb_model.save_model(f"{models_dir}/lightgbm")
    results['lightgbm'] = lgb_metrics
    
    # Train XGBoost model
    logger.info("Training XGBoost model...")
    xgb_model = FraudDetectionModel(model_type='xgboost', use_class_balance=True, dataset_type=dataset_type)
    xgb_metrics = xgb_model.train(X, y)
    xgb_model.save_model(f"{models_dir}/xgboost")
    results['xgboost'] = xgb_metrics
    
    # Create ensemble predictions
    logger.info("Creating ensemble predictions...")
    lgb_pred, lgb_proba = lgb_model.predict(X)
    xgb_pred, xgb_proba = xgb_model.predict(X)
    
    # Simple ensemble (average probabilities)
    ensemble_proba = (lgb_proba + xgb_proba) / 2
    ensemble_pred = (ensemble_proba > 0.5).astype(int)
    
    # Calculate ensemble metrics
    from sklearn.metrics import roc_auc_score, average_precision_score
    ensemble_auc = roc_auc_score(y, ensemble_proba)
    ensemble_ap = average_precision_score(y, ensemble_proba)
    
    results['ensemble'] = {
        'auc': ensemble_auc,
        'average_precision': ensemble_ap
    }
    
    logger.info(f"Ensemble AUC: {ensemble_auc:.4f}")
    logger.info(f"Ensemble AP: {ensemble_ap:.4f}")
    
    # Save ensemble results
    with open(f"{models_dir}/ensemble_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


if __name__ == "__main__":
    # Example usage
    from data_generator import TransactionGenerator
    
    # Generate synthetic data
    generator = TransactionGenerator()
    df = generator.generate_transaction_stream(num_users=500, transactions_per_user=30)
    
    # Prepare features and target
    X = df.drop(['is_fraud', 'fraud_type', 'transaction_id', 'user_id', 'timestamp', 'ip_address'], axis=1)
    y = df['is_fraud']
    
    # Train ensemble models
    results = train_ensemble_models(X, y)
    
    print("Training completed!")
    print(f"LightGBM AUC: {results['lightgbm']['auc']:.4f}")
    print(f"XGBoost AUC: {results['xgboost']['auc']:.4f}")
    print(f"Ensemble AUC: {results['ensemble']['auc']:.4f}")


