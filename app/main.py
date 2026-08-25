from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi import Request as FastAPIRequest
from fastapi.responses import FileResponse

logging.basicConfig(
    level=os.getenv("LABEL_LENS_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("label_lens")

app = FastAPI(title="Label Lens", docs_url=None, redoc_url=None)
BASE_DIR = Path(__file__).resolve().parents[1]

# Search-a-licious: world.openfoodfacts.org/cgi/search.pl is deprecated and
# frequently returns 503, which surfaced to users as a 502. This endpoint is
# the maintained replacement. ponytail: single provider, no failover host.
OFF_SEARCH = "https://search.openfoodfacts.org/search"
OFF_PRODUCT = "https://world.openfoodfacts.org/api/v3/product"
USER_AGENT = "LabelLens/0.1 (buildathon demo)"
MAX_SCAN_BYTES = 8 * 1024 * 1024
# The :free Gemma 3 slug was withdrawn from OpenRouter (404); this is the
# current free vision successor. Override with LABEL_LENS_OCR_MODEL (:free only).
FREE_OCR_MODEL = "google/gemma-4-31b-it:free"


load_dotenv(BASE_DIR / ".env")
OCR_MODEL = (
    os.getenv("LABEL_LENS_OCR_MODEL", FREE_OCR_MODEL)
    if os.getenv("LABEL_LENS_OCR_MODEL", FREE_OCR_MODEL).endswith(":free")
    else FREE_OCR_MODEL
)
COUNTRIES = {
    "au": "australia",
    "ca": "canada",
    "de": "germany",
    "in": "india",
    "gb": "united-kingdom",
    "sg": "singapore",
    "us": "united-states",
}
SAFETY_FIELDS = {
    "Allergen declaration": ("allergen", "contains", "may contain"),
    "Best-before or use-by date": ("best before", "use by", "expiry", "expiration"),
    "Lot or batch number": ("lot", "batch"),
    "Storage instructions": ("storage", "store", "keep in"),
    "Preparation instructions": ("preparation", "cook", "boil", "directions"),
    "Manufacturer or importer": ("manufacturer", "manufactured by", "importer", "address"),
}
INGREDIENT_FLAGS = (
    (re.compile(r"\b(sugar|glucose syrup|corn syrup|fructose|jaggery|honey)\b", re.I),
     "Sweetener", "Compare with the sugars line; the ingredient list does not show quantity."),
    (re.compile(r"\b(palm oil|palm fat|hydrogenated|shortening)\b", re.I),
     "Added fat", "Worth checking alongside saturated fat and the serving basis."),
    (re.compile(r"\b(artificial|synthetic)\s+(flavou?r|colou?r)\b", re.I),
     "Additive wording", "Check the package for the additive name and local labelling context."),
    (
        re.compile(r"\b(preservative|emulsifier|stabilizer|flavou?r enhancer|sweetener)\b", re.I),
        "Additive",
        "The name alone is not a safety verdict; verify the exact additive and amount "
        "if important.",
    ),
    (
        re.compile(
            r"\b(wheat|gluten|peanut|almond?s?|milk|soy|soya|sesame|egg|fish|shellfish)\b",
            re.I,
        ),
     "Potential allergen", "Confirm the package's Contains / May contain statement."),
)

OCR_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "raw_text": {"type": "string"},
        "captured": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "value": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "evidence": {"type": "string"},
                },
                "required": ["name", "value", "confidence", "evidence"],
            },
        },
        "missing": {"type": "array", "items": {"type": "string"}},
        "suggestions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["raw_text", "captured", "missing", "suggestions"],
}

DEMO_PRODUCTS = {
    "demo-muesli": {
        "code": "demo-muesli",
        "product_name": "Morning Crunch Muesli",
        "brands": "Label Lens demo",
        "quantity": "400 g",
        "image_front_small_url": "",
        "nutriments": {
            "energy-kcal_100g": 382,
            "fat_100g": 6.8,
            "saturated-fat_100g": 1.2,
            "carbohydrates_100g": 68,
            "sugars_100g": 18,
            "fiber_100g": 8,
            "proteins_100g": 10,
            "salt_100g": 0.28,
        },
        "ingredients_text": "Whole grain oats, raisins, almonds, sunflower seeds, jaggery.",
        "front_claims": ["Source of fibre"],
        "quantity_full": "400 g",
        "serving_size": "50 g",
        "serving_quantity": 50,
        "data_quality": "demo data",
    },
    "demo-crackers": {
        "code": "demo-crackers",
        "product_name": "Everyday Seed Crackers",
        "brands": "Label Lens demo",
        "quantity": "200 g",
        "image_front_small_url": "",
        "nutriments": {
            "energy-kcal_100g": 454,
            "fat_100g": 18,
            "saturated-fat_100g": 4.2,
            "carbohydrates_100g": 62,
            "sugars_100g": 3.5,
            "fiber_100g": 6,
            "proteins_100g": 11,
            "salt_100g": 1.3,
        },
        "ingredients_text": "Whole wheat flour, sesame seeds, sunflower oil, salt, cumin.",
        "front_claims": ["Baked, not fried"],
        "quantity_full": "200 g",
        "serving_size": "25 g",
        "serving_quantity": 25,
        "data_quality": "demo data",
    },
}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "public/index.html")


@app.get("/tokens.css", include_in_schema=False)
def tokens() -> FileResponse:
    return FileResponse(BASE_DIR / "public/tokens.css", media_type="text/css")


def _request_json(url: str, attempts: int = 3) -> dict:
    last_error = None
    for attempt in range(1, attempts + 1):
        request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        try:
            with urlopen(request, timeout=8) as response:
                return json.load(response)
        except HTTPError as error:
            last_error = error
            log.warning("OFF %s -> HTTP %s (attempt %s/%s)", url, error.code, attempt, attempts)
            if error.code < 500 and error.code != 429:
                break  # a real client error won't fix itself on retry
        except (OSError, URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            log.warning("OFF %s -> %s (attempt %s/%s)", url, error, attempt, attempts)
        if attempt < attempts:
            time.sleep(0.4 * attempt)
    log.error("OFF request failed after %s attempts: %s (%s)", attempts, url, last_error)
    raise HTTPException(
        status_code=502, detail="Product registry is temporarily unavailable"
    ) from last_error


def _as_text(value) -> str | None:
    # search-a-licious returns some fields (e.g. brands) as lists; legacy sent strings.
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) or None
    return value or None


def _summary(product: dict) -> dict:
    nutrients = product.get("nutriments") or {}
    return {
        "code": product.get("code", ""),
        "name": _as_text(product.get("product_name")) or "Unnamed product",
        "brand": _as_text(product.get("brands")) or "Unknown brand",
        "quantity": product.get("quantity") or "Pack size unavailable",
        "image": product.get("image_front_small_url") or product.get("image_front_url") or "",
        "sugar": nutrients.get("sugars_100g"),
        "protein": nutrients.get("proteins_100g"),
        "source": "Demo label" if product.get("data_quality") else "Open Food Facts",
    }


def _grams(value: str | None) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*g\b", value or "", re.IGNORECASE)
    return float(match.group(1)) if match else None


def _claim_checks(product: dict) -> list[dict]:
    return [
        {
            "claim": claim,
            "status": "Needs evidence",
            "note": "A label claim needs its definition and supporting evidence to be verified.",
        }
        for claim in product.get("front_claims", [])
    ]


def _ingredient_items(text: str) -> list[dict]:
    items = []
    current = []
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            value = "".join(current).strip()
            if value:
                items.append(value)
            current = []
        else:
            current.append(char)
    value = "".join(current).strip()
    if value:
        items.append(value)
    return [
        {
            "order": index,
            "text": value,
            "flags": [
                {"label": label, "note": note}
                for pattern, label, note in INGREDIENT_FLAGS
                if pattern.search(value)
            ],
        }
        for index, value in enumerate(items, 1)
    ]


def _completeness(product: dict, nutrients: dict) -> dict:
    captured = []
    missing = []
    if product.get("product_name"):
        captured.append("Product name")
    else:
        missing.append("Product name")
    if product.get("ingredients_text"):
        captured.append("Ingredients")
    else:
        missing.append("Ingredients")
    if product.get("quantity"):
        captured.append("Net quantity")
    else:
        missing.append("Net quantity")
    if product.get("serving_size") or product.get("serving_quantity"):
        captured.append("Serving size")
    else:
        missing.append("Serving size")
    if any(value is not None for value in nutrients.values()):
        captured.append("Part of nutrition panel")
    else:
        missing.append("Nutrition panel")
    safety_values = {
        "Allergen declaration": product.get("allergens") or product.get("allergens_tags"),
        "Best-before or use-by date": product.get("expiration_date"),
        "Lot or batch number": product.get("lot_number"),
        "Storage instructions": product.get("conservation_conditions")
        or product.get("storage_conditions"),
        "Preparation instructions": product.get("preparation")
        or product.get("instructions_for_use"),
        "Manufacturer or importer": product.get("manufacturing_places")
        or product.get("packaging_text"),
    }
    for label, value in safety_values.items():
        (captured if value else missing).append(label)
    return {
        "captured": captured,
        "missing": missing,
        "unverified": [
            "Current package versus registry record",
            "Marketing-claim evidence",
            "Country-specific label compliance",
        ],
    }


def _safety_missing(captured: list[dict]) -> list[str]:
    names = " ".join(str(field.get("name", "")) for field in captured).lower()
    return [
        label
        for label, aliases in SAFETY_FIELDS.items()
        if not any(alias in names for alias in aliases)
    ]


def _detail(product: dict) -> dict:
    nutrients = product.get("nutriments") or {}
    serving = product.get("serving_quantity")
    serving_sugar = nutrients.get("sugars_serving")
    if serving_sugar is None and serving and nutrients.get("sugars_100g") is not None:
        serving_sugar = nutrients["sugars_100g"] * float(serving) / 100

    nutrition = [
        ("Energy", nutrients.get("energy-kcal_100g"), "kcal"),
        ("Sugar", nutrients.get("sugars_100g"), "g"),
        ("Protein", nutrients.get("proteins_100g"), "g"),
        ("Fibre", nutrients.get("fiber_100g"), "g"),
        ("Saturated fat", nutrients.get("saturated-fat_100g"), "g"),
        ("Salt", nutrients.get("salt_100g"), "g"),
    ]
    pack_grams = _grams(product.get("quantity"))
    pack_sugar = (
        nutrients["sugars_100g"] * pack_grams / 100
        if pack_grams and nutrients.get("sugars_100g") is not None
        else None
    )
    servings_per_pack = pack_grams / float(serving) if pack_grams and serving else None
    ingredient_items = _ingredient_items(product.get("ingredients_text") or "")
    return {
        **_summary(product),
        "ingredients": product.get("ingredients_text") or "Ingredients not available",
        "serving_size": product.get("serving_size") or "Serving size unavailable",
        "serving_sugar": round(serving_sugar, 1) if serving_sugar is not None else None,
        "pack_sugar": round(pack_sugar, 1) if pack_sugar is not None else None,
        "servings_per_pack": round(servings_per_pack, 1) if servings_per_pack else None,
        "nutrition": [
            {"name": name, "value": value, "unit": unit} for name, value, unit in nutrition
        ],
        "source_url": f"https://world.openfoodfacts.org/product/{product.get('code', '')}",
        "updated": product.get("last_modified_t"),
        "claim_checks": _claim_checks(product),
        "completeness": _completeness(product, nutrients),
        "ingredients_list": ingredient_items,
        "ingredient_preview": [item["text"] for item in ingredient_items],
        "lessons": [
            {
                "title": "Serving size is a lens",
                "body": (
                    "Nutrition values are often shown per serving and per 100 g. "
                    "Use the same basis when comparing products."
                ),
            },
            {
                "title": "Order tells you quantity",
                "body": (
                    "Ingredients are displayed in the order provided by the product record. "
                    "The first few deserve your attention."
                ),
            },
            {
                "title": "Claims need context",
                "body": (
                    "Words on the front of a pack are not the same as the complete nutrition "
                    "panel. This report flags claims it cannot verify."
                ),
            },
        ],
        "limitations": [
            "Registry data may be incomplete or outdated.",
            "Check the physical package for the latest label.",
            "This is educational information, not medical advice.",
        ],
    }


def _country_from_request(request: FastAPIRequest) -> str:
    value = (
        request.headers.get("x-vercel-ip-country")
        or request.headers.get("cf-ipcountry")
        or request.headers.get("x-country-code")
        or ""
    ).lower()
    return COUNTRIES.get(value, "")


@app.get("/api/region")
def get_region(request: FastAPIRequest) -> dict:
    return {"country": _country_from_request(request) or "world"}


@app.get("/api/products")
def search_products(
    request: FastAPIRequest,
    q: str = Query(min_length=2, max_length=80),
    country: str = Query(default="", max_length=48),
) -> dict:
    demo_matches = [
        _summary(product)
        for product in DEMO_PRODUCTS.values()
        if q.lower() in json.dumps(product).lower()
    ]
    if demo_matches:
        return {"products": demo_matches, "source": "demo"}

    selected_country = country.lower().strip() or _country_from_request(request)
    if selected_country and not re.fullmatch(r"[a-z-]+", selected_country):
        raise HTTPException(status_code=400, detail="Invalid country filter")
    params = {
        "q": q,
        "fields": "code,product_name,brands,quantity,image_front_small_url,nutriments",
        "page_size": 8,
    }
    if selected_country:
        params["countries_tags_en"] = selected_country
    result = _request_json(f"{OFF_SEARCH}?{urlencode(params)}")
    products = result.get("hits", result.get("products", []))  # search-a-licious uses "hits"
    log.info("search q=%r country=%r -> %s products", q, selected_country or "world", len(products))
    return {
        "products": [_summary(product) for product in products],
        "source": "Open Food Facts",
        "country": selected_country or "world",
    }


@app.get("/api/products/{code}")
def get_product(code: str) -> dict:
    if code in DEMO_PRODUCTS:
        return _detail(DEMO_PRODUCTS[code])
    if not code.isalnum() or len(code) > 32:
        raise HTTPException(status_code=400, detail="Invalid product code")
    return _detail(_request_json(f"{OFF_PRODUCT}/{quote(code)}").get("product", {}))


def _ocr_text(response: dict) -> str:
    if response.get("choices"):
        content = response["choices"][0].get("message", {}).get("content", "")
        if isinstance(content, list):
            return "".join(item.get("text", "") for item in content if isinstance(item, dict))
        return content
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                return content.get("text", "")
    return ""


def _ocr_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("OCR response was not an object")
    return parsed


def _ocr_label(image: bytes, content_type: str) -> dict:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return {
            "status": "unavailable",
            "feedback": [
                "The photo was received, but OCR is not configured on this deployment.",
                "Set OPENROUTER_API_KEY to enable Gemma label reading.",
                "No image was saved.",
            ],
            "safety_missing": list(SAFETY_FIELDS),
        }
    prompt = (
        "Read this packaged-food label as carefully as possible. Transcribe visible text exactly "
        "in raw_text. Extract only information visibly supported by the image: product/brand text, "
        "serving size, nutrition values and basis, ingredients in order, allergens, storage, "
        "dates, and front/back claims. For every captured field, include a short evidence quote "
        "and confidence. "
        "List important label fields not visible or unreadable in missing. Suggestions must be "
        "label-reading actions only, such as retake the photo, compare per 100 g, inspect an "
        "unreadable allergen line, or verify a marketing claim. Never diagnose, prescribe, rank a "
        "product, or infer a health outcome. "
        "Do not fill gaps from general knowledge."
    )
    payload = {
        "model": OCR_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{content_type};base64,{base64.b64encode(image).decode()}"
                    },
                },
            ],
        }],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "label_scan", "strict": True, "schema": OCR_SCHEMA},
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:8000"),
        "X-Title": "Label Lens",
    }
    def post(body: dict, attempts: int = 3) -> dict:
        request = Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        for attempt in range(1, attempts + 1):
            try:
                with urlopen(request, timeout=45) as response:
                    return json.load(response)
            except HTTPError as error:
                detail = error.read()[:300].decode(errors="replace")
                log.warning(
                    "OCR model=%s -> HTTP %s (attempt %s/%s): %s",
                    OCR_MODEL, error.code, attempt, attempts, detail,
                )
                # 400/404/422 = payload issue (handled by caller's fallback); don't retry.
                # 429/5xx = free-tier rate limit or upstream blip; back off and retry.
                if error.code not in {429} and error.code < 500:
                    raise
                if attempt == attempts:
                    raise
                time.sleep(0.8 * attempt)
        raise RuntimeError("unreachable")

    try:
        try:
            result = post(payload)
        except HTTPError as error:
            if error.code not in {400, 404, 422}:
                raise
            log.info("OCR retrying %s without structured output (HTTP %s)", OCR_MODEL, error.code)
            result = post({k: v for k, v in payload.items() if k != "response_format"})
        parsed = _ocr_json(_ocr_text(result))
    except (OSError, URLError, TimeoutError, TypeError, ValueError, json.JSONDecodeError) as error:
        log.error("OCR failed for model=%s: %s", OCR_MODEL, error)
        raise HTTPException(
            status_code=502, detail="OCR could not read this photo right now"
        ) from error
    parsed["safety_missing"] = _safety_missing(parsed.get("captured", []))
    log.info("OCR ok model=%s captured=%s", OCR_MODEL, len(parsed.get("captured", [])))
    return {"status": "complete", "model": OCR_MODEL, **parsed}


@app.post("/api/scan-label")
async def scan_label(request: FastAPIRequest) -> dict:
    """Read a temporary camera photo and return only evidence visible in it."""
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload a label photo (JPEG, PNG, or HEIC)")
    image = await request.body()
    if not image:
        raise HTTPException(status_code=400, detail="The label photo was empty")
    if len(image) > MAX_SCAN_BYTES:
        raise HTTPException(status_code=413, detail="Keep the label photo under 8 MB")
    return _ocr_label(image, content_type)
