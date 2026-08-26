# Click Background Remover

A small Gradio web app for removing image backgrounds. Upload an image, click the subject you want to keep, preview the selected subject, and export a transparent PNG.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:7860.

## Render

This repository includes `render.yaml` configured for a free Render web service. The app binds to Render's `PORT` environment variable automatically.

The lightweight `u2netp` model is used by default to keep memory requirements lower on free CPU hosting.
