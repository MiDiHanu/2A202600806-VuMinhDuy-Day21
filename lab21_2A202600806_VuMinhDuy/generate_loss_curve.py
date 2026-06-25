"""Generate loss_curve.png using stdlib only (no matplotlib/Pillow).

Draws a simple line plot with grid, axes labels, title, legend.
Output: results/loss_curve.png
"""

import os
import struct
import zlib

# ── Data ──
STEPS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
TRAIN_LOSS = [2.342, 1.982, 1.812, 1.701, 1.643, 1.598, 1.567,
              1.541, 1.523, 1.508, 1.498, 1.489, 1.516]
EVAL_LOSS = 1.5161

# ── Image dimensions ──
W, H = 1000, 600
MARGIN_L, MARGIN_R = 90, 50
MARGIN_T, MARGIN_B = 70, 80
PLOT_W = W - MARGIN_L - MARGIN_R
PLOT_H = H - MARGIN_T - MARGIN_B

# ── Data ranges ──
X_MIN, X_MAX = 0, 70
Y_MIN, Y_MAX = 1.4, 2.5

def x_to_px(x):
    return MARGIN_L + (x - X_MIN) / (X_MAX - X_MIN) * PLOT_W

def y_to_px(y):
    return MARGIN_T + (1 - (y - Y_MIN) / (Y_MAX - Y_MIN)) * PLOT_H

# ── Pixel buffer (RGB) ──
pixels = [[(255, 255, 255)] * W for _ in range(H)]

def set_px(x, y, color):
    if 0 <= int(x) < W and 0 <= int(y) < H:
        pixels[int(y)][int(x)] = color

def draw_line(x1, y1, x2, y2, color, thickness=1):
    """Bresenham's line algorithm with thickness."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    dx = abs(x2 - x1); sx = 1 if x1 < x2 else -1
    dy = -abs(y2 - y1); sy = 1 if y1 < y2 else -1
    err = dx + dy
    while True:
        for dx_off in range(-thickness // 2, thickness // 2 + 1):
            for dy_off in range(-thickness // 2, thickness // 2 + 1):
                set_px(x1 + dx_off, y1 + dy_off, color)
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy; x1 += sx
        if e2 <= dx:
            err += dx; y1 += sy

def draw_filled_circle(cx, cy, r, color):
    cx, cy, r = int(cx), int(cy), int(r)
    for y in range(cy - r, cy + r + 1):
        for x in range(cx - r, cx + r + 1):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                set_px(x, y, color)

def draw_rect(x1, y1, x2, y2, color):
    for y in range(int(y1), int(y2) + 1):
        for x in range(int(x1), int(x2) + 1):
            set_px(x, y, color)

# ── Background ──
draw_rect(0, 0, W - 1, H - 1, (255, 255, 255))

# ── Grid lines ──
for x in range(0, 71, 10):
    px = int(x_to_px(x))
    draw_line(px, MARGIN_T, px, MARGIN_T + PLOT_H, (230, 230, 230))
for y in [1.5, 1.7, 1.9, 2.1, 2.3]:
    py = int(y_to_px(y))
    draw_line(MARGIN_L, py, MARGIN_L + PLOT_W, py, (230, 230, 230))

# ── Axes ──
draw_line(MARGIN_L, MARGIN_T, MARGIN_L, MARGIN_T + PLOT_H, (60, 60, 60), 2)
draw_line(MARGIN_L, MARGIN_T + PLOT_H, MARGIN_L + PLOT_W, MARGIN_T + PLOT_H, (60, 60, 60), 2)

# ── Plot train loss ──
NAVY = (14, 42, 82)
RED = (200, 16, 46)

for i in range(len(STEPS) - 1):
    x1 = x_to_px(STEPS[i]); y1 = y_to_px(TRAIN_LOSS[i])
    x2 = x_to_px(STEPS[i + 1]); y2 = y_to_px(TRAIN_LOSS[i + 1])
    draw_line(x1, y1, x2, y2, NAVY, 3)

# Markers
for sx, sy in zip(STEPS, TRAIN_LOSS):
    draw_filled_circle(x_to_px(sx), y_to_px(sy), 5, NAVY)

# Eval loss horizontal line
y_eval = int(y_to_px(EVAL_LOSS))
draw_line(MARGIN_L + 1, y_eval, MARGIN_L + PLOT_W - 1, y_eval, RED, 2)

# ── 5x7 bitmap font for labels ──
FONT_5x7 = {
    ' ': ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    '0': ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    '1': ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    '2': ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    '3': ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    '4': ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    '5': ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    '6': ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    '7': ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    '8': ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    '9': ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    '.': ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    '-': ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    ':': ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    '/': ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
    '=': ["00000", "00000", "11111", "00000", "11111", "00000", "00000"],
    'T': ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    'i': ["00100", "00000", "01100", "00100", "00100", "00100", "01110"],
    'L': ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    'a': ["00000", "00000", "01110", "00001", "01111", "10001", "01111"],
    'b': ["10000", "10000", "11110", "10001", "10001", "10001", "11110"],
    'c': ["00000", "00000", "01111", "10000", "10000", "10000", "01111"],
    'd': ["00001", "00001", "01111", "10001", "10001", "10001", "01111"],
    'e': ["00000", "00000", "01110", "10001", "11111", "10000", "01110"],
    'f': ["00110", "01001", "01000", "11110", "01000", "01000", "01000"],
    'g': ["00000", "00000", "01111", "10001", "01111", "00001", "01110"],
    'h': ["10000", "10000", "11110", "10001", "10001", "10001", "10001"],
    'k': ["10000", "10000", "10010", "10100", "11000", "10100", "10010"],
    'l': ["01100", "00100", "00100", "00100", "00100", "00100", "01110"],
    'm': ["00000", "00000", "11010", "10101", "10101", "10101", "10101"],
    'n': ["00000", "00000", "11110", "10001", "10001", "10001", "10001"],
    'o': ["00000", "00000", "01110", "10001", "10001", "10001", "01110"],
    'p': ["00000", "00000", "11110", "10001", "11110", "10000", "10000"],
    'r': ["00000", "00000", "10110", "11001", "10000", "10000", "10000"],
    's': ["00000", "00000", "01111", "10000", "01110", "00001", "11110"],
    't': ["01000", "01000", "11110", "01000", "01000", "01001", "00110"],
    'u': ["00000", "00000", "10001", "10001", "10001", "10001", "01111"],
    'v': ["00000", "00000", "10001", "10001", "10001", "01010", "00100"],
    'w': ["00000", "00000", "10001", "10001", "10101", "10101", "01010"],
    'x': ["00000", "00000", "10001", "01010", "00100", "01010", "10001"],
    'y': ["00000", "00000", "10001", "10001", "01111", "00001", "01110"],
    'A': ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    'B': ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    'C': ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    'D': ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    'E': ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    'F': ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    'G': ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
    'H': ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    'I': ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    'K': ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    'L': ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    'M': ["10001", "11011", "10101", "10001", "10001", "10001", "10001"],
    'N': ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    'O': ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    'P': ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    'Q': ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    'R': ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    'S': ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    'T': ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    'U': ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    'V': ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    'W': ["10001", "10001", "10001", "10001", "10101", "10101", "01010"],
    'X': ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    'Y': ["10001", "10001", "10001", "01010", "00100", "00100", "00100"],
    'Z': ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    '_': ["00000", "00000", "00000", "00000", "00000", "00000", "11111"],
    '(': ["00010", "00100", "01000", "01000", "01000", "00100", "00010"],
    ')': ["01000", "00100", "00010", "00010", "00010", "00100", "01000"],
    ',': ["00000", "00000", "00000", "00000", "01100", "01100", "01000"],
    '+': ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
}

def draw_text(x, y, text, color=(0, 0, 0), scale=1):
    cx = int(x)
    for ch in text:
        glyph = FONT_5x7.get(ch, FONT_5x7[' '])
        for ry, row in enumerate(glyph):
            for rx, bit in enumerate(row):
                if bit == '1':
                    for dy in range(scale):
                        for dx in range(scale):
                            set_px(cx + rx * scale + dx, int(y) + ry * scale + dy, color)
        cx += 6 * scale

# ── Title ──
title = "Training Loss Curve  Qwen2.5-3B LoRA r=16  T4  3 epochs"
draw_text((W - len(title) * 6) // 2, 25, title, NAVY)

# ── Axis labels ──
x_label = "Step"
draw_text((W - len(x_label) * 6) // 2, H - 30, x_label, (0, 0, 0))

# Y label (rotated not implemented — just put on left)
y_label = "Loss"
draw_text(20, H // 2 - 10, y_label, (0, 0, 0))

# X-axis tick labels
for x in range(0, 71, 10):
    lbl = str(x)
    draw_text(x_to_px(x) - len(lbl) * 3, MARGIN_T + PLOT_H + 10, lbl, (0, 0, 0))

# Y-axis tick labels
for y in [1.5, 1.7, 1.9, 2.1, 2.3]:
    lbl = f"{y:.1f}"
    draw_text(MARGIN_L - len(lbl) * 6 - 6, y_to_px(y) - 3, lbl, (0, 0, 0))

# ── Legend ──
leg_x, leg_y = MARGIN_L + 20, MARGIN_T + 15
draw_rect(leg_x, leg_y, leg_x + 270, leg_y + 60, (252, 252, 252))
draw_rect(leg_x, leg_y, leg_x + 270, leg_y + 60, (200, 200, 200))
draw_line(leg_x + 8, leg_y + 15, leg_x + 40, leg_y + 15, NAVY, 3)
draw_filled_circle(leg_x + 24, leg_y + 15, 4, NAVY)
draw_text(leg_x + 50, leg_y + 10, "train loss (every 5 steps)", (0, 0, 0))
draw_line(leg_x + 8, leg_y + 38, leg_x + 40, leg_y + 38, RED, 2)
draw_text(leg_x + 50, leg_y + 33, f"final eval = 1.5161  PPL = 4.55", (0, 0, 0))

# ── PNG encode ──
def png(width, height, rgb_rows):
    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b""
    for row in rgb_rows:
        raw += b"\x00" + b"".join(struct.pack("BBB", *p) for p in row)
    idat = zlib.compress(raw, 9)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")

out_path = os.path.join(os.path.dirname(__file__), "results", "loss_curve.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "wb") as f:
    f.write(png(W, H, pixels))
print(f"✓ Saved {out_path}  ({W}x{H})")