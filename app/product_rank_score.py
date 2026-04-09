"""Pandas-only scoring for product ranking (shared by routine fill and agent tools)."""

PRODUCT_TYPE_ALIASES = {
    "cleanser": ["Face Wash", "Cleanser"],
    "face wash": ["Face Wash"],
    "moisturizer": ["Moisturizer", "Face Cream", "Lotion"],
    "serum": ["Serum", "Essence"],
    "sunscreen": ["Sunscreen", "SPF"],
    "toner": ["Toner"],
    "treatment": ["Serum", "Treatment"],
}


def _score_product_for_ranking(row, skin_type: str, concerns: str, product_type_filter: str) -> float:
    score = 0.0

    if skin_type:
        skin_lower = skin_type.lower().strip()
        skin_cols = ["Sensitive", "Combination", "Oily", "Dry", "Normal"]
        for col in skin_cols:
            if col.lower() in skin_lower and col in row.index:
                if row.get(col, 0) == 1:
                    score += 15
                break
        skintype_str = str(row.get("skintype", "")).lower()
        if skintype_str and any(s in skintype_str for s in skin_lower.split()):
            score += 10

    if concerns:
        effects = str(row.get("notable_effects", "")).lower()
        for concern in concerns.lower().split(","):
            concern = concern.strip()
            if not concern:
                continue
            concern_map = {
                "acne": ["acne", "pore", "purifying", "clarifying"],
                "pores": ["pore", "pore-care", "refining"],
                "dryness": ["moisturizing", "hydration", "hydrating", "dry"],
                "sensitivity": ["soothing", "sensitive", "calming", "gentle"],
                "dark spots": ["brightening", "hyperpigmentation", "even"],
                "aging": ["anti-aging", "wrinkle", "firming"],
                "oil": ["oil-control", "balancing", "matifying"],
            }
            keywords = concern_map.get(concern, [concern])
            if any(kw in effects for kw in keywords):
                score += 12

    if product_type_filter:
        pt_lower = product_type_filter.lower().strip()
        row_pt = str(row.get("product_type", "")).lower()
        matched = False
        for alias, types in PRODUCT_TYPE_ALIASES.items():
            if alias in pt_lower:
                if any(t.lower() in row_pt for t in types):
                    score += 20
                matched = True
                break
        if not matched and pt_lower in row_pt:
            score += 20

    return max(0, min(100, score + 5))
