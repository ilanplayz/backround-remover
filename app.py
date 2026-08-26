import os
import tempfile
from functools import lru_cache

import cv2
import gradio as gr
import numpy as np
from PIL import Image
from rembg import new_session, remove

MAX_SIDE = 3000
MODEL_NAME = os.getenv("REMBG_MODEL", "u2netp")


@lru_cache(maxsize=1)
def get_session():
    return new_session(MODEL_NAME)


def normalize_image(image):
    if image is None:
        raise gr.Error("Upload an image first.")
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    image = image.astype(np.uint8)
    h, w = image.shape[:2]
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image


def ai_mask(image):
    rgb = normalize_image(image)
    pil = Image.fromarray(rgb, mode="RGB")
    result = remove(pil, session=get_session(), only_mask=True)
    return rgb, np.array(result.convert("L"), dtype=np.uint8)


def component_from_click(mask, x, y):
    binary = (mask > 20).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    if n <= 1:
        return binary * 255

    h, w = binary.shape
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    label = int(labels[y, x])

    if label == 0:
        best_label = None
        best_distance = float("inf")
        for i in range(1, n):
            ys, xs = np.where(labels == i)
            if len(xs) == 0:
                continue
            distance = np.min((xs - x) ** 2 + (ys - y) ** 2)
            if distance < best_distance:
                best_distance = distance
                best_label = i
        label = best_label or 1

    selected = np.where(labels == label, mask, 0).astype(np.uint8)
    return selected


def make_overlay(rgb, mask, x, y):
    preview = rgb.copy()
    tint = np.zeros_like(preview)
    tint[:, :, 1] = 220
    alpha = (mask.astype(np.float32) / 255.0 * 0.45)[:, :, None]
    preview = (preview * (1 - alpha) + tint * alpha).astype(np.uint8)
    cv2.circle(preview, (int(x), int(y)), 9, (255, 60, 60), -1)
    cv2.circle(preview, (int(x), int(y)), 12, (255, 255, 255), 2)
    return preview


def upload_image(image):
    rgb = normalize_image(image)
    return rgb, None, None, "Click the subject you want to keep."


def click_subject(image, evt: gr.SelectData):
    if image is None:
        raise gr.Error("Upload an image first.")
    if not evt or evt.index is None:
        raise gr.Error("I could not read that click. Try again.")

    x, y = evt.index
    rgb, mask = ai_mask(image)
    selected = component_from_click(mask, x, y)
    overlay = make_overlay(rgb, selected, x, y)
    state = {"x": int(x), "y": int(y)}
    return overlay, state, "Subject selected. If it looks right, click Remove Background."


def remove_background(image, point):
    if image is None:
        raise gr.Error("Upload an image first.")
    if not point:
        raise gr.Error("Click the subject you want to keep first.")

    rgb, mask = ai_mask(image)
    selected = component_from_click(mask, point["x"], point["y"])

    rgba = np.dstack([rgb, selected]).astype(np.uint8)
    output = Image.fromarray(rgba, mode="RGBA")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    output.save(tmp.name, format="PNG")
    return rgba, tmp.name, "Done — the background is transparent."


def reset_all():
    return None, None, None, None, "Upload an image to begin."


css = """
#app-wrap {max-width: 1050px; margin: 0 auto;}
#headline {text-align:center; margin-bottom: 0.2rem;}
#subhead {text-align:center; opacity:.75; margin-bottom:1rem;}
"""

with gr.Blocks(css=css, title="Click Background Remover") as demo:
    point_state = gr.State(None)

    with gr.Column(elem_id="app-wrap"):
        gr.Markdown("# Click Background Remover", elem_id="headline")
        gr.Markdown("Upload an image, then click the person or object you want to keep.", elem_id="subhead")

        with gr.Row():
            input_image = gr.Image(
                label="1. Upload & click subject",
                type="numpy",
                sources=["upload"],
                interactive=True,
                height=480,
            )
            preview_image = gr.Image(
                label="2. Selection preview",
                type="numpy",
                interactive=False,
                height=480,
            )

        status = gr.Markdown("Upload an image to begin.")

        with gr.Row():
            remove_btn = gr.Button("Remove Background", variant="primary")
            reset_btn = gr.Button("Start Over")

        with gr.Row():
            result_image = gr.Image(
                label="Transparent PNG",
                type="numpy",
                format="png",
                interactive=False,
                height=420,
            )
            download_file = gr.File(label="Download PNG")

    input_image.upload(
        fn=upload_image,
        inputs=input_image,
        outputs=[input_image, preview_image, point_state, status],
    )
    input_image.select(
        fn=click_subject,
        inputs=[input_image],
        outputs=[preview_image, point_state, status],
    )
    remove_btn.click(
        fn=remove_background,
        inputs=[input_image, point_state],
        outputs=[result_image, download_file, status],
    )
    reset_btn.click(
        fn=reset_all,
        outputs=[input_image, preview_image, result_image, point_state, status],
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
