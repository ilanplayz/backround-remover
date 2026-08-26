# Simple Background Remover

A simple Gradio web app for automatic background removal. Upload a PNG, JPG/JPEG, WebP, BMP, or TIFF image and the app immediately shows a transparent PNG that can be downloaded at the original resolution.

## How it works

The app uses classical image processing rather than an AI model. OpenCV GrabCut assumes the main subject is near the center, separates it from the surrounding background, and creates a softly feathered transparency mask.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:7860.

## Render

This repository includes `render.yaml` configured for a free Render web service. The app binds to Render's `PORT` environment variable automatically.

No model is downloaded. Removing `rembg`, ONNX Runtime, and the image-matting dependency tree keeps startup fast and memory use comfortably below Render's free 512 MB limit.
