# src/data_gen/build_dataset.py
import pandas as pd
from datetime import datetime
from entities import generate_users, generate_merchants
from transactions import generate_normal_stream
from fraud_typologies import inject_card_testing, inject_account_takeover, inject_velocity_abuse

def build_dataset(n_users=5000, n_merchants=800, days=90, fraud_rate=0.05):
    users = generate_users(n_users)
    merchants = generate_merchants(n_merchants)
    start_date = datetime(2026, 1, 1)

    all_txns = []
    for user in users:
        all_txns.extend(generate_normal_stream(user, merchants, start_date, days=days))

    # Pick a subset of users to victimize — real fraud hits a small % of accounts
    n_fraud_users = int(n_users * fraud_rate)
    fraud_users = random.sample(users, k=n_fraud_users)

    for user in fraud_users:
        # Random day within the window for the attack to occur
        attack_day = random.randint(0, days - 1)
        attack_time = start_date + timedelta(days=attack_day, hours=random.randint(0, 23))

        typology = random.choice(["card_testing", "account_takeover", "velocity_abuse"])
        if typology == "card_testing":
            all_txns.extend(inject_card_testing(user, merchants, attack_time))
        elif typology == "account_takeover":
            all_txns.append(inject_account_takeover(user, merchants, attack_time))
        else:
            all_txns.extend(inject_velocity_abuse(user, merchants, attack_time))

    df = pd.DataFrame(all_txns)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["transaction_id"] = [f"T{i:07d}" for i in range(len(df))]
    return df

if __name__ == "__main__":
    import random
    from datetime import timedelta
    df = build_dataset()
    print(df["is_fraud"].value_counts(normalize=True))
    print(df["fraud_type"].value_counts())
    df.to_csv("../../data/raw/transactions.csv", index=False)