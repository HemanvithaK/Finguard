import sys
import pandas as pd
sys.path.append("../data_gen")

from entities import generate_users
from features import engineer_features
from train_baseline import time_based_split
from train_transformer import train_transformer

def main():
    df = pd.read_csv("../../data/raw/transactions.csv", parse_dates=["timestamp"])
    users = generate_users(n_users=5000)
    users_df = pd.DataFrame(users)

    df = engineer_features(df, users_df)
    train_df, test_df = time_based_split(df, split_date="2026-03-15")

    model, probs = train_transformer(train_df, test_df, epochs=25)  # Increased epochs

    # Save model state dict and test results
    import joblib
    joblib.dump({"model": model.state_dict(), 
                 "test_probs": probs, 
                 "test_labels": test_df["is_fraud"].values,
                 "test_fraud_types": test_df["fraud_type"].values}, 
                "transformer_results.pkl")
    print("\nTest results saved to transformer_results.pkl")

    return model, test_df, probs

if __name__ == "__main__":
    main()