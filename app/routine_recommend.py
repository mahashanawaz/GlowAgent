"""
Build AM/PM routine slots from the product dataset + vector retrieval,
using the same scoring signals as product_ranking_tool (no LLM call).
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from app.glow_agent import (
    _google_image_search,
    _google_product_search,
    _is_bad_catalog_or_geo_host,
    _score_product_for_ranking,
    products_df,
    vectorstore,
)

# (step_name, product_type hint for scorer, semantic query boost)
AM_STEP_SPECS = [
    ("Cleanser", "cleanser", "gentle face wash cleanser morning"),
    ("Toner", "toner", "hydrating balancing toner"),
    ("Serum", "serum", "day serum antioxidant vitamin c niacinamide"),
    ("Moisturizer", "moisturizer", "lightweight daytime moisturizer"),
    ("Sunscreen", "sunscreen", "broad spectrum sunscreen SPF face"),
]

PM_STEP_SPECS = [
    ("Cleanser", "cleanser", "cleansing oil balm evening cleanser"),
    ("Exfoliant", "toner", "exfoliating toner AHA BHA glycolic salicylic"),
    ("Treatment", "serum", "night treatment serum retinol acne dark spots"),
    ("Moisturizer", "moisturizer", "night cream rich moisturizer"),
    ("Eye Cream", "moisturizer", "eye cream eye treatment"),
]


def _allergy_filter(candidates: pd.DataFrame, allergies: list[str]) -> pd.DataFrame:
    out = candidates
    for term in allergies:
        t = term.strip().lower()
        if not t:
            continue
        mask = out["product_name"].str.lower().str.contains(t, na=False) | out[
            "notable_effects"
        ].str.lower().str.contains(t, na=False)
        out = out[~mask]
    return out


def _pick_top_row(
    pool: pd.DataFrame,
    skin_type: str,
    concerns: str,
    product_type: str,
    semantic_query: str,
) -> Optional[pd.Series]:
    if pool.empty:
        return None
    scored = pool.copy()
    scored["_rank_score"] = scored.apply(
        lambda row: _score_product_for_ranking(row, skin_type, concerns, product_type),
        axis=1,
    )
    if semantic_query.strip():
        try:
            docs = vectorstore.similarity_search(semantic_query, k=24)
            retrieved = set()
            for doc in docs:
                for line in doc.page_content.split("\n"):
                    if line.startswith("Product:"):
                        n = line.replace("Product:", "").strip()
                        if n:
                            retrieved.add(n)

            def boost(row):
                base = row["_rank_score"]
                return base + 25 if row["product_name"] in retrieved else base

            scored["_rank_score"] = scored.apply(boost, axis=1)
        except Exception:
            pass
    top = scored.nlargest(1, "_rank_score")
    if top.empty:
        return None
    return top.iloc[0]


def _row_to_slot(step: str, row: pd.Series) -> dict[str, Any]:
    name = str(row.get("product_name", "") or "").strip()
    brand = str(row.get("brand", "") or "").strip()
    label = f"{brand} {name}".strip() or name
    # Google search links only — avoids many slow Tavily/reachability calls per routine load.
    link = _google_product_search(label)
    pic = row.get("picture_src", "")
    image: Optional[str] = None
    if pic and str(pic) not in ("Unknown", "nan") and str(pic).startswith("http"):
        ps = str(pic).strip()
        if _is_bad_catalog_or_geo_host(ps):
            image = _google_image_search(label)
        else:
            image = ps
    return {
        "step": step,
        "product": name or None,
        "brand": brand or None,
        "price": None,
        "link": link,
        "image": image,
    }


def _empty_slot(step_name: str) -> dict[str, Any]:
    return {
        "step": step_name,
        "product": None,
        "brand": None,
        "price": None,
        "link": None,
        "image": None,
    }


def _fill_steps(
    specs: list[tuple[str, str, str]],
    pool: pd.DataFrame,
    skin_type: str,
    concerns_str: str,
    used_names: set[str],
) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    work = pool.copy()
    for step_name, pt, query in specs:
        step_pool = work
        if step_name == "Eye Cream":
            eye_pool = work[
                work["product_name"].str.lower().str.contains("eye", na=False)
            ]
            if len(eye_pool) >= 2:
                step_pool = eye_pool
        step_pool = step_pool[~step_pool["product_name"].isin(used_names)]
        row = _pick_top_row(step_pool, skin_type, concerns_str, pt, query)
        if row is not None:
            pname = str(row.get("product_name", "") or "")
            used_names.add(pname)
            work = work[work["product_name"] != pname]
            slots.append(_row_to_slot(step_name, row))
        else:
            slots.append(_empty_slot(step_name))
    return slots


def recommend_routine_from_profile(
    skin_type: str = "",
    concerns: Optional[list[str]] = None,
    allergies: Optional[list[str]] = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Returns { "am": [...], "pm": [...] } with one product dict per step,
    keys: step, product, brand, price, link, image.
    """
    concerns_str = ", ".join(concerns or [])
    allergy_list = [a for a in (allergies or []) if str(a).strip()]
    base = _allergy_filter(products_df.copy(), allergy_list)

    used: set[str] = set()
    am_slots = _fill_steps(AM_STEP_SPECS, base, skin_type, concerns_str, used)
    pm_slots = _fill_steps(PM_STEP_SPECS, base, skin_type, concerns_str, used)

    return {"am": am_slots, "pm": pm_slots}
