"""Google search URL helpers for routine UI — no API calls."""

import urllib.parse


def _is_bad_catalog_or_geo_host(url: str) -> bool:
    try:
        host = urllib.parse.urlparse(url).netloc.lower()
        path = urllib.parse.urlparse(url).path.lower()
    except Exception:
        return True
    if "beautyhaul.com" in host:
        return True
    if "innisfree.com" in host and "/id/" in path:
        return True
    if host == "shopee.co.id" or host.endswith(".shopee.co.id"):
        return True
    if "thebodyshop.co.id" in host:
        return True
    return False


def _google_image_search(product_label: str) -> str:
    pl = (product_label or "").strip()
    if not pl:
        return "https://www.google.com/search?tbm=isch&q=skincare+product"
    return (
        "https://www.google.com/search?tbm=isch&q="
        + urllib.parse.quote_plus(f"{pl} skincare product")
    )


def _google_product_search(product_label: str) -> str:
    pl = (product_label or "").strip()
    if not pl:
        return "https://www.google.com/search?q=skincare"
    return "https://www.google.com/search?q=" + urllib.parse.quote_plus(f"{pl} skincare buy")
