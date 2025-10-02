"""
Synthetic transaction data generator for fraud detection training.
Creates realistic transaction patterns with fraud scenarios.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple
import json
from dataclasses import dataclass, asdict


@dataclass
class Transaction:
    """Transaction data structure."""
    transaction_id: str
    timestamp: datetime
    amount: float
    merchant: str
    card_type: str
    hour: int
    day_of_week: int
    is_weekend: bool
    previous_transactions: int
    avg_amount: float
    max_amount: float
    is_fraud: bool
    fraud_type: str = "none"
    location_country: str = "US"
    ip_address: str = "192.168.1.1"
    device_type: str = "mobile"


class FraudPattern:
    """Defines fraud patterns for synthetic data generation."""
    
    @staticmethod
    def velocity_fraud(transactions: List[Transaction]) -> bool:
        """High velocity fraud - many transactions in short time."""
        if len(transactions) < 5:
            return False
        
        recent = transactions[-5:]
        time_span = (recent[-1].timestamp - recent[0].timestamp).total_seconds()
        return time_span < 300  # 5 minutes
    
    @staticmethod
    def amount_anomaly(transaction: Transaction) -> bool:
        """Unusually high amount compared to user's history."""
        return transaction.amount > transaction.avg_amount * 10
    
    @staticmethod
    def time_anomaly(transaction: Transaction) -> bool:
        """Transaction at unusual time."""
        return transaction.hour < 6 or transaction.hour > 23
    
    @staticmethod
    def merchant_anomaly(transaction: Transaction) -> bool:
        """Transaction at suspicious merchant."""
        suspicious_merchants = [
            "crypto_exchange", "offshore_bank", "gambling_site",
            "prepaid_cards", "money_transfer", "high_risk_merchant"
        ]
        return transaction.merchant in suspicious_merchants


class TransactionGenerator:
    """Generates synthetic transaction data with fraud patterns."""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        self.merchants = [
            "grocery_store", "gas_station", "restaurant", "pharmacy",
            "clothing_store", "electronics_store", "bookstore", "gym",
            "hotel", "airline", "supermarket", "coffee_shop",
            "crypto_exchange", "offshore_bank", "gambling_site",  # Fraud merchants
            "prepaid_cards", "money_transfer", "high_risk_merchant"
        ]
        
        self.card_types = ["debit", "credit", "prepaid"]
        self.countries = ["US", "CA", "MX", "GB", "DE", "FR", "IT", "ES"]
        self.device_types = ["mobile", "desktop", "tablet"]
        
        self.fraud_patterns = [
            FraudPattern.velocity_fraud,
            FraudPattern.amount_anomaly,
            FraudPattern.time_anomaly,
            FraudPattern.merchant_anomaly
        ]
    
    def generate_user_profile(self) -> Dict:
        """Generate a user profile with transaction patterns."""
        return {
            "user_id": f"user_{random.randint(10000, 99999)}",
            "avg_amount": random.uniform(20, 500),
            "max_amount": random.uniform(200, 2000),
            "preferred_merchants": random.sample(self.merchants, 5),
            "usual_hours": random.sample(range(24), 12),
            "fraud_probability": random.uniform(0.001, 0.02)  # 0.1-2% fraud rate
        }
    
    def generate_transaction(self, user_profile: Dict, timestamp: datetime) -> Transaction:
        """Generate a single transaction for a user."""
        
        # Determine if this will be fraud
        is_fraud = random.random() < user_profile["fraud_probability"]
        
        if is_fraud:
            # Generate fraud transaction
            amount = random.uniform(user_profile["max_amount"] * 2, user_profile["max_amount"] * 20)
            merchant = random.choice(self.merchants[-6:])  # Suspicious merchants
            hour = random.choice([random.randint(0, 5), random.randint(23, 23)])
            fraud_type = random.choice(["velocity", "amount", "time", "merchant"])
        else:
            # Generate normal transaction
            amount = random.uniform(5, user_profile["avg_amount"] * 3)
            merchant = random.choice(user_profile["preferred_merchants"])
            hour = random.choice(user_profile["usual_hours"])
            fraud_type = "none"
        
        # Generate previous transaction count (simulate user history)
        previous_transactions = random.randint(0, 100)
        
        return Transaction(
            transaction_id=f"txn_{random.randint(100000, 999999)}",
            timestamp=timestamp,
            amount=round(amount, 2),
            merchant=merchant,
            card_type=random.choice(self.card_types),
            hour=hour,
            day_of_week=timestamp.weekday(),
            is_weekend=timestamp.weekday() >= 5,
            previous_transactions=previous_transactions,
            avg_amount=user_profile["avg_amount"],
            max_amount=user_profile["max_amount"],
            is_fraud=is_fraud,
            fraud_type=fraud_type,
            location_country=random.choice(self.countries),
            ip_address=f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}",
            device_type=random.choice(self.device_types)
        )
    
    def generate_transaction_stream(self, 
                                  num_users: int = 1000,
                                  transactions_per_user: int = 50,
                                  start_date: datetime = None) -> pd.DataFrame:
        """Generate a stream of transactions for multiple users."""
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        
        all_transactions = []
        user_profiles = {}
        
        for user_idx in range(num_users):
            user_profile = self.generate_user_profile()
            user_profiles[user_profile["user_id"]] = user_profile
            
            user_transactions = []
            
            for txn_idx in range(transactions_per_user):
                # Generate timestamp (spread over time)
                days_offset = random.randint(0, 30)
                hours_offset = random.randint(0, 23)
                minutes_offset = random.randint(0, 59)
                
                timestamp = start_date + timedelta(
                    days=days_offset,
                    hours=hours_offset,
                    minutes=minutes_offset
                )
                
                # Check for velocity fraud
                if len(user_transactions) >= 5:
                    recent_transactions = user_transactions[-5:]
                    if FraudPattern.velocity_fraud(recent_transactions):
                        # Force fraud for velocity pattern
                        user_profile["fraud_probability"] = 1.0
                
                transaction = self.generate_transaction(user_profile, timestamp)
                transaction.user_id = user_profile["user_id"]
                
                user_transactions.append(transaction)
                all_transactions.append(transaction)
                
                # Reset fraud probability after velocity fraud
                if user_profile["fraud_probability"] == 1.0:
                    user_profile["fraud_probability"] = random.uniform(0.001, 0.02)
        
        # Convert to DataFrame
        df = pd.DataFrame([asdict(txn) for txn in all_transactions])
        
        # Add derived features
        df['amount_log'] = np.log1p(df['amount'])
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['amount_to_avg_ratio'] = df['amount'] / (df['avg_amount'] + 1)
        df['amount_to_max_ratio'] = df['amount'] / (df['max_amount'] + 1)
        
        return df
    
    def save_to_kafka(self, df: pd.DataFrame, kafka_config: Dict):
        """Save transactions to Kafka topic."""
        try:
            from kafka import KafkaProducer
            import json
            
            producer = KafkaProducer(
                bootstrap_servers=kafka_config['bootstrap_servers'],
                value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                key_serializer=lambda x: x.encode('utf-8') if x else None
            )
            
            for _, row in df.iterrows():
                message = row.to_dict()
                message['timestamp'] = message['timestamp'].isoformat()
                
                producer.send(
                    kafka_config['topic'],
                    key=row['user_id'],
                    value=message
                )
            
            producer.flush()
            producer.close()
            print(f"Sent {len(df)} transactions to Kafka topic {kafka_config['topic']}")
            
        except ImportError:
            print("kafka-python not installed. Skipping Kafka export.")
    
    def save_to_redis(self, df: pd.DataFrame, redis_config: Dict):
        """Save transactions to Redis stream."""
        try:
            import redis
            import json
            
            r = redis.Redis(
                host=redis_config['host'],
                port=redis_config['port'],
                decode_responses=True
            )
            
            for _, row in df.iterrows():
                message = row.to_dict()
                message['timestamp'] = message['timestamp'].isoformat()
                
                r.xadd(
                    redis_config['stream'],
                    message,
                    maxlen=redis_config.get('maxlen', 10000)
                )
            
            print(f"Sent {len(df)} transactions to Redis stream {redis_config['stream']}")
            
        except ImportError:
            print("redis-py not installed. Skipping Redis export.")


def main():
    """Generate and save synthetic transaction data."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Generate synthetic transaction data')
    parser.add_argument('--num-users', type=int, default=1000, help='Number of users')
    parser.add_argument('--transactions-per-user', type=int, default=50, help='Transactions per user')
    parser.add_argument('--output', type=str, default='data/transactions.csv', help='Output file')
    parser.add_argument('--kafka', action='store_true', help='Send to Kafka')
    parser.add_argument('--redis', action='store_true', help='Send to Redis')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Generate data
    generator = TransactionGenerator()
    df = generator.generate_transaction_stream(
        num_users=args.num_users,
        transactions_per_user=args.transactions_per_user
    )
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"Generated {len(df)} transactions, saved to {args.output}")
    
    # Print fraud statistics
    fraud_rate = df['is_fraud'].mean()
    print(f"Fraud rate: {fraud_rate:.2%}")
    print(f"Fraud types distribution:")
    print(df[df['is_fraud']]['fraud_type'].value_counts())
    
    # Send to streaming systems if requested
    if args.kafka:
        kafka_config = {
            'bootstrap_servers': ['localhost:9092'],
            'topic': 'transactions'
        }
        generator.save_to_kafka(df, kafka_config)
    
    if args.redis:
        redis_config = {
            'host': 'localhost',
            'port': 6379,
            'stream': 'transactions',
            'maxlen': 10000
        }
        generator.save_to_redis(df, redis_config)


if __name__ == "__main__":
    main()



