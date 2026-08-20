# src/models/features.py
import pandas as pd
import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):
    """Distance in km between two lat/lon points — standard formula
    for distance on a sphere, used because raw lat/lon degree
    differences aren't linear distances."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def engineer_features(df, users_df):
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # 1. Time since this user's previous transaction (seconds)
    #    — directly captures "velocity abuse" and "card testing" pace
    df["time_since_prev_txn"] = (
        df.groupby("user_id")["timestamp"].diff().dt.total_seconds()
    )
    df["time_since_prev_txn"] = df["time_since_prev_txn"].fillna(999999)  # first txn ever: no history

    # 2. Rolling transaction count in the last 10 minutes per user
    #    — this is the actual "velocity" feature, not just a single gap
    df = df.set_index("timestamp")
    df["txns_last_10min"] = (
        df.groupby("user_id")["transaction_id"]
        .rolling("10min").count()
        .reset_index(level=0, drop=True)
    )
    df = df.reset_index()

    # 3. Distance from user's home location — catches account takeover directly
    df = df.merge(users_df[["user_id", "home_lat", "home_lon", "avg_txn_amount"]], on="user_id")
    df["dist_from_home_km"] = haversine_distance(
        df["lat"], df["lon"], df["home_lat"].astype(float), df["home_lon"].astype(float)
    )

    # 4. Amount relative to the user's own historical average
    #    — a $150 charge means different things for different users
    df["amount_vs_avg_ratio"] = df["amount"] / df["avg_txn_amount"]

    # 5. Is this the user's known device? (binary)
    df["is_known_device"] = (df["device_id"] == "D_" + df["user_id"] + "_primary").astype(int)

    # 6. Distinct merchants visited by this user in the last hour
    #    — catches card testing's "many different merchants fast" signature
    #    NOTE: rolling().apply() requires numeric dtype internally, so we
    #    convert merchant_id to a numeric category code first
    df["merchant_code"] = df["merchant_id"].astype("category").cat.codes.astype(float)

    df = df.set_index("timestamp")
    df["distinct_merchants_1hr"] = (
        df.groupby("user_id")["merchant_code"]
        .rolling("1h")
        .apply(lambda x: pd.Series(x).nunique(), raw=True)
        .reset_index(level=0, drop=True)
    )
    df = df.reset_index()

    # 7. Hour of day — normal behavior clusters around 7am-11pm (Phase 1, Piece 3)
    df["hour_of_day"] = df["timestamp"].dt.hour

    return df