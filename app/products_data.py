"""Process shared product dataframe for ranking and tools (no Gemini/Chroma)."""

from app.products_csv import load_products_df

products_df = load_products_df()
if "product_href" in products_df.columns:
    _bh_href = products_df["product_href"].astype(str).str.contains(
        "beautyhaul", case=False, na=False
    )
    products_df = products_df.copy()
    products_df.loc[_bh_href, "product_href"] = ""
