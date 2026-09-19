"""
src/generate_funky_textures.py

Generates funky, vibrant textures for the 3D golf simulator:
1. Funky Neon Cyberpunk / Synthwave Radar Golf Ball:
   - High-contrast geometric split quarters (hot magenta, electric cyan, luminous solar gold, neon violet)
   - Hexagonal dimple lattice & equatorial radar tracking crosshairs
2. Vibrant Stylized Emerald Golf Turf Texture.
"""

import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def create_funky_cyber_ball_texture(size=1024):
    """
    Creates a funky, luminous high-visibility golf ball texture with:
    - 4 vibrant quadrants (Neon Magenta, Cyber Cyan, Solar Gold, Electric Violet)
    - Hexagonal dimple array patterns
    - High-visibility radar tracking bands & alignment crosshairs
    """
    img = Image.new('RGBA', (size, size), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    half = size // 2

    # 1. Funky Quadrants
    quad_colors = [
        (255, 0, 128, 255),    # Top-Left: Neon Hot Magenta (#ff0080)
        (0, 240, 255, 255),    # Top-Right: Cyber Electric Cyan (#00f0ff)
        (138, 43, 226, 255),   # Bottom-Left: Electric Violet (#8a2be2)
        (255, 215, 0, 255),    # Bottom-Right: Luminous Solar Gold (#ffd700)
    ]

    draw.rectangle([0, 0, half, half], fill=quad_colors[0])
    draw.rectangle([half, 0, size, half], fill=quad_colors[1])
    draw.rectangle([0, half, half, size], fill=quad_colors[2])
    draw.rectangle([half, half, size, size], fill=quad_colors[3])

    # 2. Add Stylized Dimple Lattice Pattern
    dimple_overlay = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    ddraw = ImageDraw.Draw(dimple_overlay)

    spacing = size // 16
    radius = spacing // 3.2

    for row in range(18):
        y = int((row + 0.5) * spacing)
        x_offset = (spacing // 2) if (row % 2 == 1) else 0
        for col in range(18):
            x = int(col * spacing + x_offset) % size
            # Dark outer shadow
            ddraw.ellipse([x - radius, y - radius, x + radius, y + radius],
                          fill=(0, 0, 0, 45), outline=(255, 255, 255, 80), width=1)
            # Inner soft highlight
            ddraw.ellipse([x - radius * 0.5, y - radius * 0.5, x + radius * 0.3, y + radius * 0.3],
                          fill=(255, 255, 255, 90))

    img = Image.alpha_composite(img, dimple_overlay)
    draw = ImageDraw.Draw(img)

    # 3. Funky Chevron Equator Band (Radar Spin Tracking Alignment)
    band_y1 = half - size // 12
    band_y2 = half + size // 12
    draw.rectangle([0, band_y1, size, band_y2], fill=(15, 23, 42, 230), outline=(255, 255, 255, 255), width=3)

    # Chevron neon arrows in band
    n_chevrons = 12
    c_w = size // n_chevrons
    for i in range(n_chevrons):
        cx = i * c_w + c_w // 2
        cy = half
        pts = [
            (cx - c_w // 3, cy - size // 16),
            (cx + c_w // 3, cy),
            (cx - c_w // 3, cy + size // 16),
            (cx - c_w // 5, cy),
        ]
        draw.polygon(pts, fill=(0, 240, 255, 255) if i % 2 == 0 else (255, 0, 128, 255))

    # 4. Bold White Crosshairs & Center Target Core
    draw.line([(half, 0), (half, size)], fill=(255, 255, 255, 240), width=4)
    draw.line([(0, half), (size, half)], fill=(255, 255, 255, 240), width=4)

    # Center target bullseye
    draw.ellipse([half - 32, half - 32, half + 32, half + 32], fill=(255, 255, 255, 255), outline=(0, 0, 0, 255), width=3)
    draw.ellipse([half - 16, half - 16, half + 16, half + 16], fill=(0, 240, 255, 255), outline=(255, 0, 128, 255), width=2)
    draw.ellipse([half - 6, half - 6, half + 6, half + 6], fill=(255, 0, 128, 255))

    return img.convert('RGB')


def create_stylized_cartoon_grass(size=1024):
    """Creates a rich, lush emerald green turf texture."""
    img = Image.new('RGB', (size, size), '#1b461c')
    draw = ImageDraw.Draw(img)

    np.random.seed(42)
    num_cells = 45
    cell_centers = np.random.randint(0, size, (num_cells, 2))
    cell_colors = [
        (26, 68, 28),
        (34, 88, 38),
        (40, 102, 42),
        (22, 58, 24),
        (30, 78, 32)
    ]

    patch_img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    pdraw = ImageDraw.Draw(patch_img)
    for cx, cy in cell_centers:
        rad = np.random.randint(size // 8, size // 3)
        col = cell_colors[np.random.randint(len(cell_colors))]
        pdraw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=col + (140,))

    patch_img = patch_img.filter(ImageFilter.GaussianBlur(radius=size // 16))
    img.paste(patch_img, (0, 0), patch_img)
    return img


def generate_all_textures():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    tex_dir = os.path.join(project_root, 'visualizer', 'textures')
    os.makedirs(tex_dir, exist_ok=True)

    print("Generating funky stylized golf ball & turf textures...")

    textures = {
        'stylized_turf.png': create_stylized_cartoon_grass(1024),
        'ball_radar_pattern.png': create_funky_cyber_ball_texture(1024),
    }

    for filename, img in textures.items():
        out_path = os.path.join(tex_dir, filename)
        img.save(out_path)
        print(f"  [+] Saved {filename} ({img.size[0]}x{img.size[1]}) to {out_path}")

    print("\nFunky textures successfully generated!")


if __name__ == '__main__':
    generate_all_textures()
