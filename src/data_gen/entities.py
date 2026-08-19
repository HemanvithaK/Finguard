# src/data_gen/entities.py
from faker import Faker
import numpy as np
import random

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

def generate_users(n_users=5000):
    """Each user has a 'home' behavioral profile: typical geo, 
    typical spend range, typical merchant categories."""
    users = []
    for i in range(n_users):
        users.append({
            "user_id": f"U{i:05d}",
            "home_lat": fake.latitude(),
            "home_lon": fake.longitude(),
            "avg_txn_amount": round(np.random.lognormal(mean=3.5, sigma=1.0), 2), #Why lognormal for spend amounts? Real transaction amounts aren't normally distributed — most are small, a few are large (right-skewed). Lognormal captures that.
            "preferred_categories": random.sample(
                ["grocery", "restaurant", "gas", "retail", "electronics", "travel"], 
                k=random.randint(2, 4)
            ),
        })
    return users

def generate_merchants(n_merchants=800):
    categories = ["grocery", "restaurant", "gas", "retail", "electronics", "travel"]
    merchants = []
    for i in range(n_merchants):
        merchants.append({
            "merchant_id": f"M{i:04d}",
            "name": fake.company(),
            "category": random.choice(categories),
            "lat": fake.latitude(),
            "lon": fake.longitude(),
        })
    return merchants