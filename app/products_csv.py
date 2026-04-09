"""Load product CSV with pandas only — no LangChain or Google AI imports."""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
products_csv_path = BASE_DIR / "data" / "skincare_products.csv"


def load_products_df() -> pd.DataFrame:
    products_df = pd.read_csv(products_csv_path)
    products_df = products_df.dropna(axis=1, how="all")

    product_text_cols = [
        "product_name",
        "product_type",
        "brand",
        "notable_effects",
        "skintype",
        "product_href",
        "picture_src",
    ]
    for col in product_text_cols:
        if col in products_df.columns:
            products_df[col] = products_df[col].fillna("Unknown")

    skin_cols = ["Sensitive", "Combination", "Oily", "Dry", "Normal"]
    present_skin = [c for c in skin_cols if c in products_df.columns]
    for col in present_skin:
        products_df[col] = products_df[col].fillna(0)

    if present_skin:
        products_df["skin_types_str"] = products_df[present_skin].apply(
            lambda row: ", ".join([col for col in present_skin if row[col] == 1]),
            axis=1,
        )
        products_df["skin_types_str"] = products_df["skin_types_str"].replace("", "Unknown")
    else:
        products_df["skin_types_str"] = "Unknown"

    if "notable_effects" not in products_df.columns:
        products_df["notable_effects"] = "Unknown"

    return products_df
