from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi import Request as FastAPIRequest
from fastapi.responses import FileResponse
from loguru import logger as log

LOG_LEVEL = os.getenv("LABEL_LENS_LOG_LEVEL", "DEBUG").upper()
log.remove()
log.add(
    sys.stderr,
    level=LOG_LEVEL,
    backtrace=True,
    diagnose=True,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    ),
)


# Route stdlib logging (uvicorn, fastapi, urllib) through loguru so everything
# shares one verbose sink. ponytail: standard loguru intercept recipe.
class _InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = log.level(record.levelname).name
        except ValueError:
            level = record.levelno
        log.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())


logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
    lg = logging.getLogger(name)
    lg.handlers = [_InterceptHandler()]
    lg.propagate = False

app = FastAPI(title="Label Lens", docs_url=None, redoc_url=None)
BASE_DIR = Path(__file__).resolve().parents[1]


@app.middleware("http")
async def log_requests(request: FastAPIRequest, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed = (time.perf_counter() - start) * 1000
        log.exception("{} {} -> 500 ({:.0f}ms)", request.method, request.url.path, elapsed)
        raise
    elapsed = (time.perf_counter() - start) * 1000
    level = "WARNING" if response.status_code >= 500 else "INFO"
    log.log(level, "{} {} -> {} ({:.0f}ms)", request.method, request.url.path,
            response.status_code, elapsed)
    return response

MAX_SCAN_BYTES = 8 * 1024 * 1024
# Google AI Studio free tier via its OpenAI-compatible endpoint.
# Gemma (open, multimodal) has far higher free limits than the Gemini flash
# models — ~30 RPM / 14.4k RPD vs 20/day on gemini-3.6-flash — which is what a
# public launch needs. It's weaker at clean JSON, so _ocr_label's plain-prompt
# fallback does more work. Override the exact id from AI Studio via env if it
# differs. Must be a MULTIMODAL Gemma variant (accepts image input).
FREE_OCR_MODEL = "gemma-4-31b-it"  # 30 RPM / 16K TPM / 14.4K RPD (26B id: gemma-4-26b-a4b-it)
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


load_dotenv(BASE_DIR / ".env")
# Ordered fallback chain: try the stronger Gemini flash models first (better JSON,
# faster), then fall to the higher-limit Gemma models for volume once flash's small
# free budget 429s. Each model has its own free daily budget, so chaining multiplies
# the effective RPD. A runtime 429 rolls to the next id. Override with a comma list.
OCR_MODELS = [
    m.strip()
    for m in os.getenv(
        "LABEL_LENS_OCR_MODELS",
        os.getenv(
            # gemini-2.5-flash is fast + reliable (verified ~1.7s), then Gemma for
            # daily volume. The newest 3.7/3.6-flash are currently 503 "high demand"
            # (~70s hangs) — add them back via LABEL_LENS_OCR_MODELS once that eases.
            "LABEL_LENS_OCR_MODEL",
            "gemini-2.5-flash,gemma-4-31b-it,gemma-4-26b-a4b-it",
        ),
    ).split(",")
    if m.strip()
] or [FREE_OCR_MODEL]


# Thinking stays ON by default: with reasoning_effort=none the flash models hung
# to the 90s timeout on a small image. Set OCR_DISABLE_THINKING=1 to try turning
# it off again (only affects gemini flash; Gemma has no thinking mode).
OCR_DISABLE_THINKING = os.getenv("OCR_DISABLE_THINKING", "0") == "1"


def _thinking_off(model: str) -> bool:
    m = model.lower()
    return OCR_DISABLE_THINKING and m.startswith("gemini") and "flash" in m

# Global OCR rate gate. The free tier is a shared per-key limit, so the
# constraint is global across all visitors, not per-user. This token bucket
# serialises calls to OCR_RPM/min; a per-model daily counter caps each model at
# OCR_DAILY_CAP and rolls the chain to the next model when one is spent.
# ponytail: in-process only — one uvicorn worker. Multi-instance/serverless
# needs a shared store (e.g. Upstash Redis) for a single global queue + counters.
OCR_RPM = max(1, int(os.getenv("OCR_RPM", "30")))  # Gemma free tier: 30 req/min
_OCR_INTERVAL = 60.0 / OCR_RPM
OCR_MAX_WAIT = float(os.getenv("OCR_MAX_WAIT", "100"))   # never hold a request longer than this
OCR_MAX_QUEUE = int(os.getenv("OCR_MAX_QUEUE", "10"))    # reject once this many are already waiting
OCR_DAILY_CAP = int(os.getenv("OCR_DAILY_CAP", "14000"))  # per model, under the 14.4k RPD
OCR_HTTP_TIMEOUT = float(os.getenv("OCR_HTTP_TIMEOUT", "60"))  # abort a hung/overloaded model and roll to the next
_ocr_gate_lock = asyncio.Lock()
_ocr_next_slot = 0.0
_ocr_waiting = 0
_ocr_day = ""             # current UTC date; per-model counters reset on rollover
_ocr_counts: dict[str, int] = {}


class _RateLimited(Exception):
    """A model returned 429 at call time — mark it spent and fall through the chain."""


class _Overloaded(Exception):
    """A model is transiently unavailable (503 high-demand) or timed out — roll to
    the next model for THIS request, but keep it in the chain for future scans."""


def _roll_day() -> None:
    global _ocr_day, _ocr_counts
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if today != _ocr_day:
        _ocr_day, _ocr_counts = today, {}


def _remaining(model: str) -> int:
    return OCR_DAILY_CAP - _ocr_counts.get(model, 0)


def _total_remaining() -> int:
    _roll_day()
    return sum(max(0, _remaining(m)) for m in OCR_MODELS)


def _mark_spent(model: str) -> None:
    _ocr_counts[model] = OCR_DAILY_CAP


async def _ocr_gate(exclude: set[str]) -> str:
    """Wait for a rate slot and return the first model with daily budget; 503 if none."""
    global _ocr_next_slot, _ocr_waiting
    async with _ocr_gate_lock:
        _roll_day()
        model = next((m for m in OCR_MODELS if m not in exclude and _remaining(m) > 0), None)
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="We've reached today's free scan limit — please come back tomorrow.",
            )
        now = time.monotonic()
        run_at = max(now, _ocr_next_slot)
        wait = run_at - now
        if _ocr_waiting >= OCR_MAX_QUEUE or wait > OCR_MAX_WAIT:
            raise HTTPException(
                status_code=503,
                detail="High demand on our free OCR right now — please try again in a minute.",
            )
        _ocr_next_slot = run_at + _OCR_INTERVAL
        _ocr_waiting += 1
        _ocr_counts[model] = _ocr_counts.get(model, 0) + 1  # count against this model's budget
    try:
        if wait > 0:
            log.info("OCR queued: waiting {:.1f}s ({} ahead) on {}", wait, _ocr_waiting - 1, model)
            await asyncio.sleep(wait)
    finally:
        async with _ocr_gate_lock:
            _ocr_waiting -= 1
    return model
SAFETY_FIELDS = {
    "Allergen declaration": ("allergen", "contains", "may contain"),
    "Best-before or use-by date": ("best before", "use by", "expiry", "expiration"),
    "Lot or batch number": ("lot", "batch"),
    "Storage instructions": ("storage", "store", "keep in"),
    "Preparation instructions": ("preparation", "cook", "boil", "directions"),
    "Manufacturer or importer": ("manufacturer", "manufactured by", "importer", "address"),
}
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

@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "public/index.html")


@app.get("/healthz", include_in_schema=False)
def healthz() -> dict:
    return {"ok": True}


@app.get("/tokens.css", include_in_schema=False)
def tokens() -> FileResponse:
    return FileResponse(BASE_DIR / "public/tokens.css", media_type="text/css")


def _safety_missing(text: str) -> list[str]:
    t = text.lower()
    return [
        label
        for label, aliases in SAFETY_FIELDS.items()
        if not any(alias in t for alias in aliases)
    ]


def _num(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value or ""))
    return float(match.group()) if match else None


def _split_ingredients(text: str) -> list[str]:
    """Split an ingredients line on top-level commas (keeps '... (338)' intact)."""
    out, cur, depth = [], [], 0
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            if cur and "".join(cur).strip():
                out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur).strip().rstrip("."))
    return out


def _normalize(p: dict) -> dict:
    """Repair Gemma's loose output into the flat shape the frontend renders,
    filling gaps from raw_text so a missing/garbled key never blanks the UI."""
    raw = p.get("raw_text") or ""
    text = "\n".join([raw] + [str(v) for v in p.values() if isinstance(v, str)])

    def filled(key, pattern):
        n = _num(p.get(key))
        if n is None:
            m = re.search(pattern, text, re.I)
            n = float(m.group(1)) if m else None
        return n

    p["sugar_g"] = filled("sugar_g", r"\bsugars?\b[^\d\n]{0,20}?(\d+(?:\.\d+)?)\s*g\b")
    p["energy_kcal"] = filled("energy_kcal", r"\benergy\b[^\d\n]{0,20}?(\d+(?:\.\d+)?)\s*k?cal\b")
    p["protein_g"] = filled("protein_g", r"\bprotein\b[^\d\n]{0,20}?(\d+(?:\.\d+)?)\s*g\b")
    p["total_fat_g"] = filled("total_fat_g", r"\b(?:total\s+)?fat\b[^\d\n]{0,20}?(\d+(?:\.\d+)?)\s*g\b")
    p["saturated_fat_g"] = filled("saturated_fat_g", r"\bsaturat\w*[^\d\n]{0,20}?(\d+(?:\.\d+)?)\s*g\b")
    p["trans_fat_g"] = filled("trans_fat_g", r"\btrans[^\d\n]{0,20}?(\d+(?:\.\d+)?)\s*g\b")
    p["carbohydrate_g"] = filled("carbohydrate_g", r"\bcarbohydrate?s?\b[^\d\n]{0,20}?(\d+(?:\.\d+)?)\s*g\b")
    p["fiber_g"] = filled("fiber_g", r"\b(?:dietary\s+)?fib(?:re|er)\b[^\d\n]{0,20}?(\d+(?:\.\d+)?)\s*g\b")
    # sodium: keep the printed mg, or convert a g value / a salt figure (sodium ≈ salt × 0.4)
    if _num(p.get("sodium_mg")) is None:
        m = re.search(r"\bsodium\b[^\d\n]{0,20}?(\d+(?:\.\d+)?)\s*(mg|g)\b", text, re.I)
        if m:
            p["sodium_mg"] = float(m.group(1)) * (1000 if m.group(2).lower() == "g" else 1)
        else:
            s = re.search(r"\bsalt\b[^\d\n]{0,20}?(\d+(?:\.\d+)?)\s*g\b", text, re.I)
            p["sodium_mg"] = round(float(s.group(1)) * 400) if s else None
    else:
        p["sodium_mg"] = _num(p.get("sodium_mg"))

    ing = p.get("ingredients")
    if not isinstance(ing, list) or not ing:
        m = re.search(r"ingredients?\s*[:\-]([^\n]+)", text, re.I)
        p["ingredients"] = _split_ingredients(m.group(1)) if m else []
    else:
        p["ingredients"] = [str(x).strip() for x in ing if str(x).strip()]

    if not p.get("nutrition_basis"):
        m = re.search(r"per\s*(\d+)\s*(ml|g)\b", text, re.I)
        p["nutrition_basis"] = f"{m.group(1)} {m.group(2)}" if m else None
    if not p.get("net_quantity"):
        m = re.search(r"net\s*(?:quantity|weight|content|vol\w*)\b[^\d\n]{0,20}?(\d+(?:\.\d+)?)\s*(ml|g|l|kg)\b", text, re.I)
        p["net_quantity"] = f"{m.group(1)} {m.group(2)}" if m else None
    if not p.get("best_before"):
        m = re.search(r"best\s*before[^\n]{0,60}", text, re.I)
        p["best_before"] = m.group(0).strip() if m else None
    if not p.get("serving_size"):
        m = re.search(r"serving\s*size\s*[:\-]?\s*([\d.]+\s*(?:g|ml|kg|l)\b[^\n.;]*)", text, re.I)
        p["serving_size"] = m.group(1).strip() if m else None

    p["safety_missing"] = _safety_missing(text)
    p.setdefault("missing", [])
    p.setdefault("suggestions", [])
    return p




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
    """Parse the model's reply into a dict — tolerant of markdown fences and
    prose around the JSON. If it's not JSON at all, degrade to raw_text so
    _normalize can still recover fields by regex (never fatal)."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    # Try the whole thing first; then scan for every balanced {...} object and
    # keep the richest dict. A naive first-{-to-last-} slice breaks when the model
    # wraps its reply in prose that itself contains braces (e.g. "{Rolled Oats,...}").
    best = None
    for candidate in [cleaned, *_brace_objects(cleaned)]:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and (best is None or len(parsed) > len(best)):
            best = parsed
            if "raw_text" in parsed or "ingredients" in parsed:
                break
    if best is not None:
        return best
    log.warning("OCR reply was not JSON — degrading to raw_text ({} chars)", len(text or ""))
    return {"raw_text": text or ""}


def _brace_objects(s: str):
    """Yield each balanced {...} substring (string-aware, so braces inside JSON
    strings don't throw off the depth count)."""
    depth = start = 0
    in_str = esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                yield s[start:i + 1]


def _ocr_label(image: bytes, content_type: str, model: str) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {
            "status": "unavailable",
            "feedback": [
                "The photo was received, but OCR is not configured on this deployment.",
                "Set GEMINI_API_KEY to enable label reading.",
                "No image was saved.",
            ],
            "safety_missing": list(SAFETY_FIELDS),
        }
    log.info("OCR request: model={} content_type={} bytes={}", model, content_type, len(image))
    # Gemma ignores response_format, so we drive the shape from the prompt and
    # ask for exactly the flat keys the frontend renders. raw_text stays the
    # safety net the client re-parses when a key is missing/garbled.
    prompt = (
        "You are reading a photo of a packaged-food label. Reply with ONE JSON object and "
        "nothing else — no markdown fences, no commentary. Use only what is visibly supported "
        "by the image; use null (or [] for lists) when something isn't visible. Never diagnose, "
        "rank, or infer health outcomes, and never fill gaps from general knowledge.\n"
        "Keys:\n"
        '- raw_text: the full label text, transcribed verbatim.\n'
        '- product: product or brand name.\n'
        '- ingredients: array of ingredient strings in the printed order (split the ingredients line).\n'
        '- sugar_g: number — sugars per the nutrition basis.\n'
        '- energy_kcal: number.\n'
        '- protein_g: number.\n'
        '- total_fat_g: number — total fat per the nutrition basis.\n'
        '- saturated_fat_g: number.\n'
        '- trans_fat_g: number.\n'
        '- carbohydrate_g: number — total carbohydrate.\n'
        '- fiber_g: number — dietary fibre.\n'
        '- sodium_mg: number in milligrams (if only salt is printed, sodium_mg ≈ salt_g × 400).\n'
        '- nutrition_basis: e.g. "100 ml", "100 g", or "per serving".\n'
        '- net_quantity: e.g. "300 ml".\n'
        '- serving_size.\n'
        '- best_before: exactly as printed (may be relative, e.g. "8 months from manufacture").\n'
        '- dates_location: if date/batch/MRP are printed elsewhere, where (e.g. "bottom of can"); else null.\n'
        '- mrp, manufacturer, fssai.\n'
        '- veg_nonveg: "veg", "non-veg", or null.\n'
        '- missing: array of important label fields not visible or unreadable.\n'
        '- suggestions: array of short label-reading next steps.'
    )
    payload = {
        "model": model,
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
    }
    if _thinking_off(model):
        payload["reasoning_effort"] = "none"  # Google maps none → thinking_budget 0
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    def post(body: dict, attempts: int = 3) -> dict:
        request = Request(
            GEMINI_URL,
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        for attempt in range(1, attempts + 1):
            try:
                with urlopen(request, timeout=OCR_HTTP_TIMEOUT) as response:
                    data = json.load(response)
                    log.info(
                        "OCR response: model={} status={} choices={} usage={}",
                        model, response.status, len(data.get("choices", [])),
                        data.get("usage", {}),
                    )
                    return data
            except HTTPError as error:
                detail = error.read()[:300].decode(errors="replace")
                log.warning(
                    "OCR model={} -> HTTP {} (attempt {}/{}): {}",
                    model, error.code, attempt, attempts, detail,
                )
                # 503 UNAVAILABLE = overloaded ("high demand"); 404 = bad/unsupported
                # model id. Neither is fixable by retrying THIS model — roll to the next
                # so one overloaded or mistyped model can't take down the whole chain.
                if error.code in {404, 503}:
                    raise _Overloaded(model) from error
                # 400/404/422 = payload issue (handled by caller's fallback); don't retry.
                # 429/other-5xx = rate limit or upstream blip; back off and retry.
                if error.code not in {429} and error.code < 500:
                    raise
                if attempt == attempts:
                    raise
                time.sleep(0.8 * attempt)
        raise RuntimeError("unreachable")

    try:
        parsed = _ocr_json(_ocr_text(post(payload)))
    except HTTPError as error:
        if error.code == 429:
            raise _RateLimited(model) from error  # let the caller fall to the next model
        log.error("OCR failed model={}: HTTP {}", model, error.code)
        raise HTTPException(
            status_code=502,
            detail="The free AI reader hit an error on this photo — please try again.",
        ) from error
    except (TimeoutError, socket.timeout) as error:
        # A hung model shouldn't 504 the whole scan — roll to the next in the chain.
        log.warning("OCR timed out model={} ({:.0f}s) — rolling to next model: {}",
                    model, OCR_HTTP_TIMEOUT, error)
        raise _Overloaded(model) from error
    except (OSError, URLError, TypeError, ValueError, json.JSONDecodeError) as error:
        log.error("OCR failed model={}: {}", model, error)
        raise HTTPException(
            status_code=502,
            detail="Couldn't read this photo — try a tighter crop or better light, then scan again.",
        ) from error
    parsed = _normalize(parsed)
    log.info(
        "OCR ok model={} sugar={} ingredients={}",
        model, parsed.get("sugar_g"), len(parsed.get("ingredients") or []),
    )
    return {"status": "complete", **parsed}


@app.get("/api/scan-status")
async def scan_status() -> dict:
    """Rough wait estimate so the UI can show an ETA before a scan is submitted."""
    wait = max(0.0, _ocr_next_slot - time.monotonic())
    return {
        "wait_seconds": round(wait),
        "ahead": _ocr_waiting,
        "rpm": OCR_RPM,
        "daily_remaining": _total_remaining(),
    }


@app.post("/api/scan-label")
async def scan_label(request: FastAPIRequest) -> dict:
    """Read a temporary camera photo and return only evidence visible in it."""
    content_type = request.headers.get("content-type", "")
    log.info(
        "scan-label: ip={} ua={!r} content_type={!r} length={}",
        request.client.host if request.client else "?",
        request.headers.get("user-agent", "")[:80],
        content_type,
        request.headers.get("content-length", "?"),
    )
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload a label photo (JPEG, PNG, or HEIC)")
    image = await request.body()
    if not image:
        raise HTTPException(status_code=400, detail="The label photo was empty")
    if len(image) > MAX_SCAN_BYTES:
        raise HTTPException(status_code=413, detail="Keep the label photo under 8 MB")
    # Try each model in the chain: a runtime 429 marks that model spent for the
    # day and falls through to the next. _ocr_label uses blocking urlopen, so run
    # it off the event loop to keep queued requests responsive.
    tried: set[str] = set()
    overloaded = False  # some model was busy (503/timeout), not out of daily budget
    while True:
        try:
            model = await _ocr_gate(tried)  # waits for a slot; 503 if every model is spent
        except HTTPException:
            # Every model is excluded. If any dropped out from transient overload
            # rather than a real 429, say "busy" instead of "daily limit reached".
            if overloaded:
                raise HTTPException(
                    status_code=503,
                    detail="Our free AI reader is busy right now — please try again in a little while.",
                ) from None
            raise
        try:
            return await asyncio.to_thread(_ocr_label, image, content_type, model)
        except _RateLimited:
            log.warning("model {} hit its rate limit — falling through the chain", model)
            async with _ocr_gate_lock:
                _mark_spent(model)
            tried.add(model)
        except _Overloaded:
            # Transient (503 / timeout): skip for THIS request only, don't mark spent.
            log.warning("model {} overloaded — falling through the chain", model)
            tried.add(model)
            overloaded = True
