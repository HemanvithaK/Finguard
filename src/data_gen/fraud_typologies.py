# src/data_gen/fraud_typologies.py
import numpy as np
import random
from datetime import timedelta


def _sample_device(user):
    """
    Shared helper: ~12% of fraud transactions occur on the user's known
    device (session hijack, stolen physical card before the owner
    notices) rather than always showing an unknown device. This keeps
    is_known_device a strong but imperfect signal, forcing the model
    to combine it with velocity/geo/amount features rather than
    shortcut on it alone.
    """
    if random.random() < 0.12:
        return f"D_{user['user_id']}_primary"
    return f"D_UNKNOWN_{random.randint(1000, 9999)}"


def inject_card_testing(user, merchants, start_time, n_probes=None):
    """
    Simulates a card-testing attack: a burst of tiny transactions
    across many DIFFERENT merchants, seconds to minutes apart.
    Location stays near the user's home (attacker often has the
    card number but not necessarily far-off physical possession —
    many card-testing attacks are card-not-present/online).
    """
    if n_probes is None:
        n_probes = random.randint(5, 15)  # a "burst"

    transactions = []
    current_time = start_time

    # Sample merchants WITHOUT replacement — real card testing hits
    # many distinct merchants, not the same one repeatedly
    probe_merchants = random.sample(merchants, k=min(n_probes, len(merchants)))

    for merchant in probe_merchants:
        amount = round(np.random.uniform(0.50, 3.00), 2)  # tiny "is this card alive" amounts

        transactions.append({
            "user_id": user["user_id"],
            "merchant_id": merchant["merchant_id"],
            "amount": amount,
            "timestamp": current_time,
            "lat": float(user["home_lat"]) + np.random.normal(0, 0.05),
            "lon": float(user["home_lon"]) + np.random.normal(0, 0.05),
            "device_id": _sample_device(user),
            "is_fraud": 1,
            "fraud_type": "card_testing",
        })

        # Gap between probes: seconds to a couple minutes — this speed
        # is the core signal that separates it from normal spending
        current_time += timedelta(seconds=random.randint(5, 120))

    return transactions


def inject_account_takeover(user, merchants, timestamp):
    """
    One large transaction, usually geographically distant, on an
    unrecognized device — but with realistic noise: some ATO happens
    closer to home (attacker in the same region), and some occurs on
    a known device (session hijack, stolen physical card).
    """
    merchant = random.choice(merchants)  # NOT filtered by user's preferred category —
                                          # attacker doesn't know/care about user's habits

    # Distance: usually a big jump, but sometimes closer (same-region attacker) —
    # 30% chance of a smaller 1-5 degree jump, 70% chance of the original 5-30 range
    distance_degrees = np.random.choice(
        [np.random.uniform(1, 5), np.random.uniform(5, 30)], p=[0.3, 0.7]
    )
    lat = float(user["home_lat"]) + random.choice([-1, 1]) * distance_degrees
    lon = float(user["home_lon"]) + random.choice([-1, 1]) * distance_degrees

    # Go big: several multiples of the user's normal average
    amount = round(user["avg_txn_amount"] * np.random.uniform(4, 10), 2)

    return {
        "user_id": user["user_id"],
        "merchant_id": merchant["merchant_id"],
        "amount": amount,
        "timestamp": timestamp,
        "lat": lat,
        "lon": lon,
        "device_id": _sample_device(user),
        "is_fraud": 1,
        "fraud_type": "account_takeover",
    }


def inject_velocity_abuse(user, merchants, start_time, n_txns=None):
    """
    Rapid burst of normal-looking transactions at UNFAMILIAR merchants —
    a stolen card being used for real purchases quickly, before it gets
    cancelled.
    """
    if n_txns is None:
        n_txns = random.randint(4, 8)

    # Merchants OUTSIDE the user's preferred categories — key differentiator
    unfamiliar_merchants = [m for m in merchants if m["category"] not in user["preferred_categories"]]
    chosen = random.sample(unfamiliar_merchants, k=min(n_txns, len(unfamiliar_merchants)))

    transactions = []
    current_time = start_time

    for merchant in chosen:
        amount = round(np.random.normal(loc=user["avg_txn_amount"], scale=user["avg_txn_amount"] * 0.4), 2)
        amount = max(5.0, amount)  # keep it realistic, not a probe amount

        transactions.append({
            "user_id": user["user_id"],
            "merchant_id": merchant["merchant_id"],
            "amount": amount,
            "timestamp": current_time,
            "lat": float(user["home_lat"]) + np.random.normal(0, 0.1),  # slightly wider than normal jitter
            "lon": float(user["home_lon"]) + np.random.normal(0, 0.1),
            "device_id": _sample_device(user),
            "is_fraud": 1,
            "fraud_type": "velocity_abuse",
        })
        # Faster than normal user behavior, but slower than card-testing probes —
        # a distinct pace signature; overlaps with the normal shopping-trip
        # burst's 5-25 min gaps on purpose, so velocity alone isn't a perfect tell
        current_time += timedelta(minutes=random.randint(2, 20))

    return transactions