import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_customer_data(
    n_customers=4339, start_date="2023-01-01", end_date="2024-11-26"
):
    """
    Generate synthetic customer transaction data for RFM analysis
    """
    np.random.seed(42)
    random.seed(42)

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    date_range = (end_dt - start_dt).days

    customers = []
    transactions = []

    for customer_id in range(1, n_customers + 1):
        # Generate customer profile
        customer_type = np.random.choice(
            ["regular", "premium", "vip"], p=[0.7, 0.25, 0.05]
        )

        if customer_type == "regular":
            n_transactions = np.random.poisson(5) + 1
            avg_value = np.random.normal(50, 15)
        elif customer_type == "premium":
            n_transactions = np.random.poisson(15) + 5
            avg_value = np.random.normal(150, 30)
        else:  # vip
            n_transactions = np.random.poisson(30) + 10
            avg_value = np.random.normal(500, 100)

        # Generate transactions for this customer
        for _ in range(n_transactions):
            # Random date within range
            days_ago = np.random.randint(0, date_range)
            transaction_date = end_dt - timedelta(days=days_ago)

            # Transaction amount with some variation
            amount = max(10, np.random.normal(avg_value, avg_value * 0.3))

            transactions.append(
                {
                    "customer_id": customer_id,
                    "transaction_date": transaction_date,
                    "amount": amount,
                    "customer_type": customer_type,
                }
            )

    # Create DataFrames
    df_transactions = pd.DataFrame(transactions)
    df_transactions["transaction_date"] = pd.to_datetime(
        df_transactions["transaction_date"]
    )

    # Add customer information
    customer_info = (
        df_transactions.groupby("customer_id")
        .agg(
            {
                "customer_type": "first",
                "transaction_date": ["min", "max", "count"],
                "amount": ["sum", "mean"],
            }
        )
        .round(2)
    )

    customer_info.columns = [
        "customer_type",
        "first_transaction",
        "last_transaction",
        "transaction_count",
        "total_amount",
        "avg_amount",
    ]
    customer_info = customer_info.reset_index()

    return df_transactions, customer_info


if __name__ == "__main__":
    # Generate the data
    transactions, customers = generate_customer_data()

    # Save to files
    transactions.to_csv("data/transactions.csv", index=False)
    customers.to_csv("data/customers.csv", index=False)

    print(f"Generated {len(customers)} customers and {len(transactions)} transactions")
    print(
        f"Date range: {transactions['transaction_date'].min()} to {transactions['transaction_date'].max()}"
    )
    print(f"Total revenue: ${transactions['amount'].sum():,.2f}")
