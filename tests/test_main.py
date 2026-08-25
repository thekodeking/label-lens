from fastapi.testclient import TestClient

from app.main import DEMO_PRODUCTS, FREE_OCR_MODEL, OCR_MODEL, _detail, app


def test_demo_product_calculates_serving_sugar():
    result = _detail(DEMO_PRODUCTS["demo-muesli"])
    assert result["serving_sugar"] == 9.0
    assert result["pack_sugar"] == 72.0
    assert result["servings_per_pack"] == 8.0
    assert result["nutrition"][1] == {"name": "Sugar", "value": 18, "unit": "g"}
    assert result["claim_checks"][0]["status"] == "Needs evidence"
    assert "Ingredients" in result["completeness"]["captured"]
    assert "Allergen declaration" in result["completeness"]["missing"]
    assert "Current package versus registry record" in result["completeness"]["unverified"]
    assert len(result["lessons"]) == 3


def test_scan_label_explains_missing_ocr_configuration(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    response = TestClient(app).post(
        "/api/scan-label", content=b"fake-image", headers={"content-type": "image/jpeg"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "unavailable"
    assert "OCR is not configured" in response.json()["feedback"][0]


def test_region_uses_platform_country_header():
    response = TestClient(app).get("/api/region", headers={"x-vercel-ip-country": "IN"})
    assert response.json() == {"country": "india"}


def test_product_search_applies_selected_country(monkeypatch):
    requested = {}

    def fake_request(url):
        requested["url"] = url
        return {"products": []}

    monkeypatch.setattr("app.main._request_json", fake_request)
    response = TestClient(app).get("/api/products?q=oreo&country=india")
    assert response.status_code == 200
    assert "countries_tags_en=india" in requested["url"]


def test_ocr_model_is_pinned_to_free_variant():
    assert OCR_MODEL == FREE_OCR_MODEL
    assert OCR_MODEL.endswith(":free")
