""" Builds the vector store once in the beginning. This prevents memory overload when deploying """

from pathlib import Path
import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
ingredients_csv_path = BASE_DIR / "data" / "skincare_ingredients.csv"
products_csv_path = BASE_DIR / "data" / "skincare_products.csv"
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"

def load_ingredients_df():
    ingredients_df = pd.read_csv(ingredients_csv_path)
    ingredients_df = ingredients_df.dropna(axis=1, how="all")

    ingredient_text_cols = [
        "name",
        "short_description",
        "what_is_it",
        "what_does_it_do",
        "who_is_it_good_for",
        "who_should_avoid",
        "url",
    ]
    for col in ingredient_text_cols:
        if col in ingredients_df.columns:
            ingredients_df[col] = ingredients_df[col].fillna("Unknown")

    return ingredients_df

def load_products_df():
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
    for col in skin_cols:
        if col in products_df.columns:
            products_df[col] = products_df[col].fillna(0)

    products_df["skin_types_str"] = products_df[skin_cols].apply(
        lambda row: ", ".join([col for col in skin_cols if row[col] == 1]),
        axis=1,
    )
    products_df["skin_types_str"] = products_df["skin_types_str"].replace("", "Unknown")

    return products_df

def format_ingredient_row(row):
    return (
        f"Ingredient: {row['name']}\n"
        f"Short Description: {row['short_description']}\n"
        f"What It Is: {row['what_is_it']}\n"
        f"What It Does: {row['what_does_it_do']}\n"
        f"Who It Is Good For: {row['who_is_it_good_for']}\n"
        f"Who Should Avoid: {row['who_should_avoid']}\n"
        f"More Info: {row['url']}\n"
        f"Source: GlowAgent ingredient dataset"
    )

def format_product_row(row):
    return (
        f"Product: {row['product_name']}\n"
        f"Brand: {row['brand']}\n"
        f"Category: {row['product_type']}\n"
        f"Effects: {row['notable_effects']}\n"
        f"Skin Types: {row['skin_types_str']}\n"
        f"More Info: {row['product_href']}\n"
        f"Image: {row['picture_src']}\n"
        f"Source: GlowAgent product dataset"
    )

def build_vectorstore():
    ingredients_df = load_ingredients_df()
    products_df = load_products_df()

    ingredient_chunks = [format_ingredient_row(row) for _, row in ingredients_df.iterrows()]
    product_chunks = [format_product_row(row) for _, row in products_df.iterrows()]
    all_chunks = product_chunks + ingredient_chunks

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    Chroma.from_texts(
        texts=all_chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_PERSIST_DIR),
        collection_name="glowagent_skincare",
    )

    print(f"Built Chroma DB at: {CHROMA_PERSIST_DIR}")

if __name__ == "__main__":
    build_vectorstore()