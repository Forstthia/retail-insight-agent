from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path


def generate_sample_data(output_path: str | Path, seed: int = 42) -> Path:
    """Generate deterministic retail data with a recent UK revenue decline."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    start = date(2026, 1, 1)
    end = date(2026, 6, 30)
    countries = ["United Kingdom", "France", "Germany"]
    products = [("Desk Lamp", 28.0), ("Storage Box", 18.0), ("Travel Mug", 12.0)]

    rows: list[dict[str, object]] = []
    order_number = 1
    current = start
    while current <= end:
        for country_index, country in enumerate(countries):
            product, base_price = products[(current.day + country_index) % len(products)]
            recent_uk = country == "United Kingdom" and current >= date(2026, 6, 1)
            quantity = rng.randint(1, 2) if recent_uk else rng.randint(2, 5)
            returned = int(recent_uk and current.day % 4 == 0)
            if not recent_uk:
                returned = int(rng.random() < 0.04)
            rows.append(
                {
                    "order_id": f"ORD-{order_number:05d}",
                    "date": current.isoformat(),
                    "country": country,
                    "product": product,
                    "customer_id": f"CUS-{(current.day * 7 + country_index * 19) % 83:03d}",
                    "quantity": quantity,
                    "unit_price": f"{base_price + rng.choice([-1.0, 0.0, 1.0]):.2f}",
                    "returned": returned,
                }
            )
            order_number += 1
        current += timedelta(days=1)

    # These two rows exercise duplicate and invalid-row handling in clean_data.
    rows.append(dict(rows[10]))
    rows.append(
        {
            "order_id": "ORD-BAD",
            "date": "2026-06-30",
            "country": "",
            "product": "Desk Lamp",
            "customer_id": "CUS-999",
            "quantity": "not-a-number",
            "unit_price": "28.00",
            "returned": 0,
        }
    )

    fields = [
        "order_id",
        "date",
        "country",
        "product",
        "customer_id",
        "quantity",
        "unit_price",
        "returned",
    ]
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output

