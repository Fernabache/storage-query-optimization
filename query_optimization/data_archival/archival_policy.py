import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple
import os
import json

class StorageTierManager:
    def __init__(self, config: Dict = None):
        """Initialize Storage Tier Manager with configuration."""
        self.config = config or {
            'hot_threshold': 90,  # days
            'warm_threshold': 180,  # days
            'access_frequency_threshold': 10,  # accesses
            'size_threshold': 100,  # MB
        }
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('StorageTierManager')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def classify_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Classify data into hot, warm, and cold tiers based on multiple criteria:
        - Last access time
        - Access frequency
        - Data size
        - Data importance
        """
        try:
            # Create a copy to avoid modifying original data
            df = data.copy()
            
            # Convert date columns to datetime
            df["last_accessed"] = pd.to_datetime(df["last_accessed"])
            df["created_date"] = pd.to_datetime(df["created_date"])
            
            # Calculate days since last access
            current_date = datetime.now()
            df["days_since_access"] = (current_date - df["last_accessed"]).dt.days
            
            # Apply classification rules
            df["tier"] = df.apply(self._apply_tier_rules, axis=1)
            
            # Calculate storage metrics
            df["storage_savings"] = df.apply(self._calculate_storage_savings, axis=1)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in classify_data: {str(e)}")
            raise

    def _apply_tier_rules(self, row: pd.Series) -> str:
        """Apply tiering rules to a single data record."""
        # Hot tier criteria
        if (row["days_since_access"] <= self.config["hot_threshold"] and 
            row["access_frequency"] >= self.config["access_frequency_threshold"]):
            return "hot"
        
        # Warm tier criteria
        elif (self.config["hot_threshold"] < row["days_since_access"] <= self.config["warm_threshold"] or 
              (row["access_frequency"] >= self.config["access_frequency_threshold"] // 2)):
            return "warm"
        
        # Cold tier criteria
        else:
            return "cold"

    def _calculate_storage_savings(self, row: pd.Series) -> float:
        """Calculate potential storage savings for each record."""
        savings_multipliers = {
            "hot": 1.0,  # No savings for hot tier
            "warm": 0.5,  # 50% savings for warm tier
            "cold": 0.2   # 80% savings for cold tier
        }
        return row["size_mb"] * (1 - savings_multipliers[row["tier"]])

    def generate_optimization_report(self, classified_data: pd.DataFrame) -> Dict:
        """Generate a detailed optimization report."""
        try:
            report = {
                "summary": {
                    "total_records": len(classified_data),
                    "total_size_gb": classified_data["size_mb"].sum() / 1024,
                    "potential_savings_gb": classified_data["storage_savings"].sum() / 1024
                },
                "tier_distribution": classified_data["tier"].value_counts().to_dict(),
                "size_by_tier": classified_data.groupby("tier")["size_mb"].sum().to_dict(),
                "recommendations": self._generate_recommendations(classified_data)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error in generate_optimization_report: {str(e)}")
            raise

    def _generate_recommendations(self, data: pd.DataFrame) -> List[str]:
        """Generate specific optimization recommendations."""
        recommendations = []
        
        # Analyze cold data
        cold_data_size = data[data["tier"] == "cold"]["size_mb"].sum()
        if cold_data_size > self.config["size_threshold"]:
            recommendations.append(
                f"Consider archiving {cold_data_size:.2f}MB of cold data to lower-cost storage"
            )

        # Analyze access patterns
        frequent_cold = data[
            (data["tier"] == "cold") & 
            (data["access_frequency"] > self.config["access_frequency_threshold"])
        ]
        if not frequent_cold.empty:
            recommendations.append(
                f"Review {len(frequent_cold)} frequently accessed items in cold storage for potential tier adjustment"
            )

        return recommendations

    def export_results(self, classified_data: pd.DataFrame, report: Dict, output_dir: str):
        """Export classification results and report."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Export classified data
            classified_data.to_csv(
                os.path.join(output_dir, "classified_data.csv"), 
                index=False
            )
            
            # Export optimization report
            with open(os.path.join(output_dir, "optimization_report.json"), "w") as f:
                json.dump(report, f, indent=4)
                
            self.logger.info(f"Results exported to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error in export_results: {str(e)}")
            raise

def generate_sample_data(num_records: int = 1000) -> pd.DataFrame:
    """Generate sample data for testing."""
    current_date = datetime.now()
    
    data = {
        "file_id": range(num_records),
        "created_date": [
            current_date - timedelta(days=np.random.randint(0, 365))
            for _ in range(num_records)
        ],
        "last_accessed": [
            current_date - timedelta(days=np.random.randint(0, 180))
            for _ in range(num_records)
        ],
        "access_frequency": np.random.randint(1, 100, num_records),
        "size_mb": np.random.uniform(1, 1000, num_records),
        "importance_score": np.random.uniform(0, 1, num_records)
    }
    
    return pd.DataFrame(data)

def main():
    # Create output directory
    output_dir = "storage_optimization_results"
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Generate or load data
        try:
            data = pd.read_csv("../data/sample_data.csv")
            print("Loaded existing data file")
        except FileNotFoundError:
            print("Generating sample data")
            data = generate_sample_data()
            data.to_csv("../data/sample_data.csv", index=False)

        # Initialize storage tier manager
        manager = StorageTierManager()

        # Classify data
        print("Classifying data...")
        classified_data = manager.classify_data(data)

        # Generate optimization report
        print("Generating optimization report...")
        report = manager.generate_optimization_report(classified_data)

        # Export results
        print("Exporting results...")
        manager.export_results(classified_data, report, output_dir)

        # Print summary
        print("\n=== Storage Optimization Summary ===")
        print(f"Total Records: {report['summary']['total_records']}")
        print(f"Total Size: {report['summary']['total_size_gb']:.2f} GB")
        print(f"Potential Savings: {report['summary']['potential_savings_gb']:.2f} GB")
        print("\nTier Distribution:")
        for tier, count in report['tier_distribution'].items():
            print(f"{tier}: {count} records")
        print("\nRecommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()