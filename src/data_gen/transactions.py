# src/data_gen/transactions.py
import numpy as np
from datetime import datetime, timedelta
import random

def make_normal_transaction(user, merchants, timestamp):
    """
    Generates one legitimate transaction for `user` at `timestamp`.
    Picks a merchant from the user's preferred categories,
    a location close to the user's home, and an amount near
    their historical average.
    """
    # 1. Pick a merchant matching the user's typical spending categories
    candidate_merchants = [m for m in merchants if m["category"] in user["preferred_categories"]]
    merchant = random.choice(candidate_merchants)

    # 2. Amount: normal variation around the user's average, never negative
    amount = max(1.0, np.random.normal(loc=user["avg_txn_amount"], scale=user["avg_txn_amount"] * 0.3))

    # 3. Location: small jitter around home (a few km), NOT the merchant's actual location —
    #    this is a simplification; card-present location ≈ user's current location
    lat = float(user["home_lat"]) + np.random.normal(0, 0.05)
    lon = float(user["home_lon"]) + np.random.normal(0, 0.05)

    return {
        "user_id": user["user_id"],
        "merchant_id": merchant["merchant_id"],
        "amount": round(amount, 2),
        "timestamp": timestamp,
        "lat": lat,
        "lon": lon,
        "device_id": f"D_{user['user_id']}_primary",  # user's usual device
        "is_fraud": 0,
        "fraud_type": None,
    }


#generating a time sequence of transactions for a user

def generate_normal_stream(user, merchants, start_date, days=90, avg_txns_per_day=0.8):
    """
    Simulates a user's normal transaction history over `days`.
    avg_txns_per_day=0.8 means most users don't transact daily —
    some days zero, some days one, rarely more.
    """
    transactions = []
    current_time = start_date

    for day in range(days):
        # Poisson distribution: models "count of independent events per time unit"
        # the standard way to model how many purchases happen in a day
        n_txns_today = np.random.poisson(avg_txns_per_day)

        for _ in range(n_txns_today):
            # Spread transactions across realistic waking hours (7am-11pm),
            # weighted toward lunchtime/evening — real spending isn't uniform across 24h
            hour = int(np.clip(np.random.normal(loc=15, scale=4), 7, 23))
            minute = random.randint(0, 59)
            txn_time = start_date + timedelta(days=day, hours=hour, minutes=minute)

            txn = make_normal_transaction(user, merchants, txn_time) #One thing to note: this function calls make_normal_transaction in a loop, and for ~5000 users × 90 days that's already tens of thousands of transactions
            transactions.append(txn)

    return transactions
