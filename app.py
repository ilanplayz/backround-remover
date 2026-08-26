import os
import tempfile

import cv2
import gradio as gr
import numpy as np
from PIL import Image

WORK_MAX_SIDE = int(os.getenv("WORK_MAX_SIDE", "1200"))
SUPPORTED_FORMATS = "PNG, JPG/JPEG, WebP, BMP, and TIFF"


def _as_uint8_array(image):
    if image is None:
        raise gr.Error("Upload an image first.")
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.floating):
        if array.size and np.nanmax(array) <= 1:
            array = array * 255
        array = np.nan_to_num(array, nan=0, posinf=255, neginf=0)
    return np.clip(array, 0, 255).astype(np.uint8)


def normalize_image(image):
    array = _as_uint8_array(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.ndim != 3 or array.shape[2] < 1:
        raise gr.Error("This image format could not be read. Try PNG, JPG, WebP, BMP, or TIFF.")
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] >= 4:
        array = array[:, :, :3]
    elif array.shape[2] == 2:
        array = np.repeat(array[:, :, :1], 3, axis=2)
    return np.ascontiguousarray(array.astype(np.uint8))


def get_input_alpha(image):
    array = _as_uint8_array(image)
    if array.ndim == 3 and array.shape[2] >= 4:
        return np.ascontiguousarray(array[:, :, 3].astype(np.uint8))
    return None


def create_foreground_mask(rgb):
    """Separate a centered subject using OpenCV GrabCut (no AI model)."""
    original_height, original_width = rgb.shape[:2]
    scale = min(1.0, WORK_MAX_SIDE / max(original_height, original_width))
    if scale < 1.0:
        work = cv2.resize(
            rgb,
            (max(1, round(original_width * scale)), max(1, round(original_height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        work = rgb.copy()
    height, width = work.shape[:2]
    margin_x = max(1, round(width * 0.04))
    margin_y = max(1, round(height * 0.04))
    rectangle = (margin_x, margin_y, width - 2 * margin_x, height - 2 * margin_y)
    mask = np.zeros((height, width), dtype=np.uint8)
    background = np.zeros((1, 65), dtype=np.float64)
    foreground = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(work, mask, rectangle, background, foreground, 5, cv2.GC_INIT_WITH_RECT)
        binary = np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)
    except cv2.error as error:
        raise gr.Error("This image could not be processed. Try a clearer image.") from error
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count > 1:
        center_x, center_y = width // 2, height // 2
        center_label = int(labels[center_y, center_x])
        if center_label == 0:
            center_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        binary = (labels == center_label).astype(np.uint8)
    alpha = cv2.GaussianBlur(binary.astype(np.float32) * 255, (0, 0), 1.2)
    alpha = cv2.resize(alpha, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    return np.clip(alpha, 0, 255).astype(np.uint8)


def remove_background(image):
    rgb = normalize_image(image)
    alpha = create_foreground_mask(rgb)
    input_alpha = get_input_alpha(image)
    if input_alpha is not None:
        alpha = np.minimum(alpha, input_alpha)
    rgba = np.dstack([rgb, alpha]).astype(np.uint8)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temporary:
        output_path = temporary.name
    Image.fromarray(rgba).save(output_path, format="PNG")
    return rgba, output_path, "Background removed. Download your transparent PNG."


def reset_all():
    return None, None, None, "Upload an image to begin."


css = """
#app-wrap {max-width: 1000px; margin: 0 auto;}
#headline, #subhead {text-align: center;}
#subhead {opacity: .75; margin-bottom: 1rem;}
"""

with gr.Blocks(css=css, title="Simple Background Remover") as demo:
    with gr.Column(elem_id="app-wrap"):
        gr.Markdown("# Simple Background Remover", elem_id="headline")
        gr.Markdown(
            f"Upload {SUPPORTED_FORMATS}. The transparent result appears automatically.",
            elem_id="subhead",
        )
        with gr.Row():
            input_image = gr.Image(
                label="Upload Image",
                type="numpy",
                image_mode="RGBA",
                sources=["upload"],
                interactive=True,
                height=480,
            )
            result_image = gr.Image(
                label="Background Removed", type="numpy", format="png", interactive=False, height=480
            )
        status = gr.Markdown("Upload an image to begin.")
        with gr.Row():
            download_file = gr.File(label="Download Transparent PNG")
            reset_button = gr.Button("Start Over")
    input_image.upload(remove_background, input_image, [result_image, download_file, status])
    reset_button.click(reset_all, outputs=[input_image, result_image, download_file, status])


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name="0.0.0.0", server_port=port, show_error=True
    )
