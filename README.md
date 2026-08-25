# Label Lens

Evidence-first packaged food label explorer.

## Run locally

```bash
uv sync
uv run uvicorn api.index:app --reload
```

Open <http://127.0.0.1:8000/>.

To enable OCR for camera photos and uploaded label images, set
`OPENROUTER_API_KEY` before starting the server. Label Lens uses the free
`google/gemma-4-31b-it:free` vision model. Any `LABEL_LENS_OCR_MODEL` override
must also end in `:free`, otherwise Label Lens falls back to Gemma free. The
image is sent for one scan and is not saved by Label Lens.

Set `LABEL_LENS_LOG_LEVEL=DEBUG` for verbose request logging when diagnosing
Open Food Facts or OCR failures.

For Vercel, deploy this directory. Vercel detects `api/index.py` as the FastAPI function and serves `public/` as static assets.
