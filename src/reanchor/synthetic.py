"""Generate a synthetic MNIST-like dataset using PIL-rendered digits."""

import random

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_PATH = "/usr/share/fonts/TTF/Hack-Regular.ttf"


def render_digit(label: int, rng: random.Random) -> np.ndarray:
    font_size = rng.randint(36, 44)
    font = ImageFont.truetype(FONT_PATH, font_size)

    canvas = Image.new("L", (64, 64), 0)
    draw = ImageDraw.Draw(canvas)
    text = str(label)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (64 - text_width) / 2 - bbox[0]
    y = (64 - text_height) / 2 - bbox[1]
    draw.text((x, y), text, fill=255, font=font)

    angle = rng.uniform(-15, 15)
    canvas = canvas.rotate(angle, fillcolor=0, resample=Image.BILINEAR)

    canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.8)))

    scale = rng.uniform(0.8, 1.15)
    new_size = int(64 * scale)
    canvas = canvas.resize((new_size, new_size), Image.BILINEAR)

    crop_x = rng.randint(0, new_size - 28)
    crop_y = rng.randint(0, new_size - 28)
    canvas = canvas.crop((crop_x, crop_y, crop_x + 28, crop_y + 28))

    pixels = np.array(canvas, dtype=np.float32) / 255.0
    noise = np.random.normal(0, 0.08, pixels.shape).astype(np.float32)
    pixels = np.clip(pixels + noise, 0.0, 1.0)
    return pixels


def generate_dataset(num_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    images = np.zeros((num_samples, 784), dtype=np.float32)
    labels = np.zeros(num_samples, dtype=np.int64)
    for i in range(num_samples):
        label = rng.randint(0, 9)
        images[i] = render_digit(label, rng).flatten()
        labels[i] = label
    return images, labels

