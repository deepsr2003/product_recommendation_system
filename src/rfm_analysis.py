import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def calculate_rfm_features(transactions_df, analysis_date=None):
    """
    Calculate RFM (Recency, Frequency, Monetary) features for each customer
    """
    if analysis_date is None:
        analysis_date = transactions_df["transaction_date"].max() + timedelta(days=1)

    # Convert to datetime if not already
    transactions_df["transaction_date"] = pd.to_datetime(
        transactions_df["transaction_date"]
    )

    # Calculate RFM metrics
    rfm = (
        transactions_df.groupby("customer_id")
        .agg(
            {
                "transaction_date": [
                    lambda x: (analysis_date - x.max()).days,  # Recency
                    "nunique",  # Frequency
                ],
                "amount": "sum",  # Monetary
            }
        )
        .round(2)
    )

    # Flatten column names
    rfm.columns = ["recency", "frequency", "monetary"]
    rfm = rfm.reset_index()

    # Add RFM scores (1-5 scale)
    rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1]).astype(
        int
    )
    rfm["frequency_score"] = pd.qcut(
        rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
    ).astype(int)
    rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(
        int
    )

    # Calculate RFM segment
    rfm["rfm_segment"] = (
        rfm["recency_score"].astype(str)
        + rfm["frequency_score"].astype(str)
        + rfm["monetary_score"].astype(str)
    )

    # Add customer type based on monetary value (median split)
    median_monetary = rfm["monetary"].median()
    rfm["is_high_value"] = (rfm["monetary"] >= median_monetary).astype(int)

    return rfm


def analyze_rfm_segments(rfm_df):
    """
    Analyze and categorize RFM segments
    """

    def categorize_segment(row):
        recency, frequency, monetary = (
            row["recency_score"],
            row["frequency_score"],
            row["monetary_score"],
        )

        if recency >= 4 and frequency >= 4 and monetary >= 4:
            return "Champions"
        elif recency >= 4 and frequency >= 3:
            return "Loyal Customers"
        elif recency >= 3 and monetary >= 4:
            return "Potential Loyalists"
        elif recency >= 4 and frequency <= 2:
            return "New Customers"
        elif recency <= 2 and frequency >= 3:
            return "At Risk"
        elif recency <= 2 and frequency <= 2:
            return "Lost"
        else:
            return "Others"

    rfm_df["segment_category"] = rfm_df.apply(categorize_segment, axis=1)

    return rfm_df


if __name__ == "__main__":
    # Load transaction data
    transactions = pd.read_csv("data/transactions.csv")
    transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])

    # Calculate RFM features
    rfm_features = calculate_rfm_features(transactions)

    # Analyze segments
    rfm_analyzed = analyze_rfm_segments(rfm_features)

    # Save results
    rfm_analyzed.to_csv("data/rfm_features.csv", index=False)

    print(f"RFM features calculated for {len(rfm_analyzed)} customers")
    print(
        f"High-value customers: {rfm_analyzed['is_high_value'].sum()} ({rfm_analyzed['is_high_value'].mean() * 100:.1f}%)"
    )
    print(f"Median monetary value: ${rfm_analyzed['monetary'].median():.2f}")

    # Display segment distribution
    print("\nSegment Distribution:")
    print(rfm_analyzed["segment_category"].value_counts())
