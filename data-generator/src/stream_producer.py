"""
Stream producer for sending credit card transactions to Kafka and Redis.
"""

import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from dataclasses import asdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamProducer:
    """Produces transaction streams to Kafka and Redis."""
    
    def __init__(self, kafka_config: Dict = None, redis_config: Dict = None):
        self.kafka_config = kafka_config or {
            'bootstrap_servers': ['localhost:9092'],
            'topic': 'transactions'
        }
        self.redis_config = redis_config or {
            'host': 'localhost',
            'port': 6379,
            'stream': 'transactions',
            'maxlen': 10000
        }
        
        self.kafka_producer = None
        self.redis_client = None
        
        # Initialize connections
        self._init_kafka()
        self._init_redis()
    
    def _init_kafka(self):
        """Initialize Kafka producer."""
        try:
            from kafka import KafkaProducer
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=self.kafka_config['bootstrap_servers'],
                value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                key_serializer=lambda x: x.encode('utf-8') if x else None,
                acks='all',
                retries=3,
                batch_size=16384,
                linger_ms=10
            )
            logger.info("Kafka producer initialized successfully")
        except ImportError:
            logger.warning("kafka-python not installed. Kafka streaming disabled.")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
    
    def _init_redis(self):
        """Initialize Redis client."""
        try:
            import redis
            self.redis_client = redis.Redis(
                host=self.redis_config['host'],
                port=self.redis_config['port'],
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis client initialized successfully")
        except ImportError:
            logger.warning("redis-py not installed. Redis streaming disabled.")
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
    
    def send_to_kafka(self, transaction: Dict) -> bool:
        """Send transaction to Kafka."""
        if not self.kafka_producer:
            return False
        
        try:
            # Add timestamp if not present
            if 'timestamp' not in transaction:
                transaction['timestamp'] = datetime.now().isoformat()
            
            # Send to Kafka
            future = self.kafka_producer.send(
                self.kafka_config['topic'],
                key=transaction.get('transaction_id', 'unknown'),
                value=transaction
            )
            
            # Wait for confirmation (optional, for reliability)
            # record_metadata = future.get(timeout=10)
            
            return True
        except Exception as e:
            logger.error(f"Failed to send to Kafka: {e}")
            return False
    
    def send_to_redis(self, transaction: Dict) -> bool:
        """Send transaction to Redis stream."""
        if not self.redis_client:
            return False
        
        try:
            # Add timestamp if not present
            if 'timestamp' not in transaction:
                transaction['timestamp'] = datetime.now().isoformat()
            
            # Send to Redis stream
            self.redis_client.xadd(
                self.redis_config['stream'],
                transaction,
                maxlen=self.redis_config.get('maxlen', 10000)
            )
            
            return True
        except Exception as e:
            logger.error(f"Failed to send to Redis: {e}")
            return False
    
    def send_transaction(self, transaction: Dict) -> Dict[str, bool]:
        """Send transaction to both Kafka and Redis."""
        results = {
            'kafka': self.send_to_kafka(transaction),
            'redis': self.send_to_redis(transaction)
        }
        
        if any(results.values()):
            logger.debug(f"Transaction sent: {transaction.get('transaction_id', 'unknown')}")
        
        return results
    
    def stream_from_dataframe(self, df: pd.DataFrame, 
                             batch_size: int = 100, 
                             delay: float = 0.1,
                             max_transactions: Optional[int] = None) -> Dict[str, int]:
        """Stream transactions from a DataFrame."""
        
        if max_transactions:
            df = df.head(max_transactions)
        
        total_transactions = len(df)
        logger.info(f"Starting to stream {total_transactions} transactions")
        
        kafka_success = 0
        redis_success = 0
        
        for i in range(0, total_transactions, batch_size):
            batch = df.iloc[i:i+batch_size]
            
            for _, row in batch.iterrows():
                # Convert row to transaction dict
                transaction = self._row_to_transaction(row)
                
                # Send transaction
                results = self.send_transaction(transaction)
                
                if results['kafka']:
                    kafka_success += 1
                if results['redis']:
                    redis_success += 1
            
            # Delay between batches
            if delay > 0:
                time.sleep(delay)
            
            # Progress logging
            if (i + batch_size) % 1000 == 0 or i + batch_size >= total_transactions:
                logger.info(f"Streamed {min(i + batch_size, total_transactions)}/{total_transactions} transactions")
        
        logger.info(f"Streaming completed. Kafka: {kafka_success}, Redis: {redis_success}")
        
        return {
            'kafka_success': kafka_success,
            'redis_success': redis_success,
            'total_attempted': total_transactions
        }
    
    def _row_to_transaction(self, row: pd.Series) -> Dict:
        """Convert DataFrame row to transaction dictionary."""
        transaction = row.to_dict()
        
        # Ensure transaction_id exists
        if 'transaction_id' not in transaction:
            transaction['transaction_id'] = f"txn_{int(time.time() * 1000000)}"
        
        # Convert numpy types to native Python types
        for key, value in transaction.items():
            if isinstance(value, np.integer):
                transaction[key] = int(value)
            elif isinstance(value, np.floating):
                transaction[key] = float(value)
            elif isinstance(value, np.bool_):
                transaction[key] = bool(value)
            elif pd.isna(value):
                transaction[key] = None
        
        return transaction
    
    def close(self):
        """Close connections."""
        if self.kafka_producer:
            self.kafka_producer.flush()
            self.kafka_producer.close()
            logger.info("Kafka producer closed")
        
        if self.redis_client:
            self.redis_client.close()
            logger.info("Redis client closed")


class CreditCardStreamProducer:
    """Specialized producer for credit card dataset streaming."""
    
    def __init__(self, kafka_config: Dict = None, redis_config: Dict = None):
        self.producer = StreamProducer(kafka_config, redis_config)
    
    def stream_credit_card_data(self, csv_file: str, 
                               sample_rate: float = 1.0,
                               batch_size: int = 100,
                               delay: float = 0.01,
                               shuffle: bool = True) -> Dict[str, int]:
        """Stream credit card transactions from CSV file."""
        
        logger.info(f"Loading credit card data from {csv_file}")
        df = pd.read_csv(csv_file)
        
        # Sample data if requested
        if sample_rate < 1.0:
            df = df.sample(frac=sample_rate, random_state=42)
            logger.info(f"Sampled {len(df)} transactions ({sample_rate:.1%} of original)")
        
        # Shuffle if requested
        if shuffle:
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Add transaction IDs
        df['transaction_id'] = [f"cc_txn_{i:08d}" for i in range(len(df))]
        
        # Add timestamp (simulate real-time)
        base_time = datetime.now() - timedelta(days=1)
        df['stream_timestamp'] = [
            base_time + timedelta(seconds=i) for i in range(len(df))
        ]
        
        logger.info(f"Starting to stream {len(df)} credit card transactions")
        
        # Stream the data
        return self.producer.stream_from_dataframe(
            df, 
            batch_size=batch_size, 
            delay=delay
        )
    
    def stream_real_time_simulation(self, csv_file: str, 
                                   duration_minutes: int = 60,
                                   transactions_per_minute: int = 1000) -> Dict[str, int]:
        """Simulate real-time streaming of credit card transactions."""
        
        logger.info(f"Loading credit card data from {csv_file}")
        df = pd.read_csv(csv_file)
        
        # Calculate how many transactions to send
        total_transactions = duration_minutes * transactions_per_minute
        
        if len(df) > total_transactions:
            # Sample if we have more data than needed
            df = df.sample(n=total_transactions, random_state=42)
        else:
            # Repeat data if we need more transactions
            repeat_times = (total_transactions // len(df)) + 1
            df = pd.concat([df] * repeat_times, ignore_index=True)
            df = df.head(total_transactions)
        
        # Shuffle the data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Add transaction IDs and timestamps
        df['transaction_id'] = [f"rt_txn_{i:08d}" for i in range(len(df))]
        
        # Calculate delay between transactions
        delay_per_transaction = 60.0 / transactions_per_minute  # seconds
        
        logger.info(f"Simulating real-time streaming:")
        logger.info(f"  Duration: {duration_minutes} minutes")
        logger.info(f"  Rate: {transactions_per_minute} transactions/minute")
        logger.info(f"  Delay: {delay_per_transaction:.3f} seconds per transaction")
        
        # Stream with calculated delay
        return self.producer.stream_from_dataframe(
            df, 
            batch_size=1, 
            delay=delay_per_transaction
        )
    
    def close(self):
        """Close the producer."""
        self.producer.close()


def main():
    """Main function for streaming credit card data."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stream credit card transactions')
    parser.add_argument('--csv-file', type=str, required=True,
                       help='Path to credit card CSV file')
    parser.add_argument('--kafka', action='store_true',
                       help='Enable Kafka streaming')
    parser.add_argument('--redis', action='store_true',
                       help='Enable Redis streaming')
    parser.add_argument('--sample-rate', type=float, default=0.1,
                       help='Sample rate for data (0.1 = 10%)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for streaming')
    parser.add_argument('--delay', type=float, default=0.01,
                       help='Delay between batches (seconds)')
    parser.add_argument('--real-time', action='store_true',
                       help='Simulate real-time streaming')
    parser.add_argument('--duration', type=int, default=60,
                       help='Duration for real-time simulation (minutes)')
    parser.add_argument('--rate', type=int, default=1000,
                       help='Transactions per minute for real-time simulation')
    
    args = parser.parse_args()
    
    # Configure streaming targets
    kafka_config = None
    redis_config = None
    
    if args.kafka:
        kafka_config = {
            'bootstrap_servers': ['localhost:9092'],
            'topic': 'credit_card_transactions'
        }
    
    if args.redis:
        redis_config = {
            'host': 'localhost',
            'port': 6379,
            'stream': 'credit_card_transactions',
            'maxlen': 50000
        }
    
    # Create producer
    producer = CreditCardStreamProducer(kafka_config, redis_config)
    
    try:
        if args.real_time:
            # Real-time simulation
            results = producer.stream_real_time_simulation(
                args.csv_file,
                duration_minutes=args.duration,
                transactions_per_minute=args.rate
            )
        else:
            # Batch streaming
            results = producer.stream_credit_card_data(
                args.csv_file,
                sample_rate=args.sample_rate,
                batch_size=args.batch_size,
                delay=args.delay
            )
        
        # Print results
        print("\n=== STREAMING RESULTS ===")
        print(f"Kafka successful: {results['kafka_success']}")
        print(f"Redis successful: {results['redis_success']}")
        print(f"Total attempted: {results['total_attempted']}")
        
    except KeyboardInterrupt:
        print("\nStreaming interrupted by user")
    except Exception as e:
        print(f"Error during streaming: {e}")
    finally:
        producer.close()


if __name__ == "__main__":
    main()
