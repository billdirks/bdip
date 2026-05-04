#!/usr/bin/env python3
"""
Generate ascii_char_map_16x16.png — a 128×128 texture representing a 16×16
grid of 8×8 character cells arranged by increasing ink density.

Each cell contains a hand-coded 8×8 bitmask representing one of 16 ASCII
characters, ordered from least dense (space) to most dense (@). The luminance
of a rendered cell's white pixels encodes how "filled" that character is, so
the GPU shader can pick the correct cell by sampling luma.

The 16 characters (density order, ascending):
  0: ' '  (space)           — 0 lit pixels
  1: '.'                    — 1 lit pixel
  2: ','                    — 2 lit pixels
  3: '-'                    — 3 lit pixels
  4: '~'                    — 4 lit pixels
  5: '!'                    — 5 lit pixels
  6: ':'                    — 5 lit pixels (colon)
  7: '+'                    — 6 lit pixels
  8: '='                    — 8 lit pixels
  9: 'i'                    — 8 lit pixels (lowercase i)
 10: 't'                    — 9 lit pixels
 11: 'n'                    — 10 lit pixels
 12: 'o'                    — 11 lit pixels
 13: 'x'                    — 12 lit pixels
 14: '#'                    — 14 lit pixels
 15: '@'                    — 16 lit pixels

Each row in the grid = one character cell row (8 px high).
Each column in the grid = one character column (8 px wide).
The texture is 16 columns × 16 rows = 128×128 pixels.
White pixels (255) = ink; Black pixels (0) = no ink.
"""

from PIL import Image

# Each character is an 8-row × 8-col bitmask (MSB = leftmost pixel).
# Rows are listed top-to-bottom.
CHARS = [
    # 0: ' ' (space) — completely empty
    [0b00000000,
     0b00000000,
     0b00000000,
     0b00000000,
     0b00000000,
     0b00000000,
     0b00000000,
     0b00000000],
    # 1: '.'
    [0b00000000,
     0b00000000,
     0b00000000,
     0b00000000,
     0b00000000,
     0b00000000,
     0b00011000,
     0b00000000],
    # 2: ','
    [0b00000000,
     0b00000000,
     0b00000000,
     0b00000000,
     0b00000000,
     0b00011000,
     0b00011000,
     0b00000000],
    # 3: '-'
    [0b00000000,
     0b00000000,
     0b00000000,
     0b00111100,
     0b00000000,
     0b00000000,
     0b00000000,
     0b00000000],
    # 4: '~'
    [0b00000000,
     0b00000000,
     0b00100100,
     0b01011010,
     0b00000000,
     0b00000000,
     0b00000000,
     0b00000000],
    # 5: '!'
    [0b00011000,
     0b00011000,
     0b00011000,
     0b00011000,
     0b00000000,
     0b00011000,
     0b00000000,
     0b00000000],
    # 6: ':' (colon)
    [0b00000000,
     0b00011000,
     0b00000000,
     0b00000000,
     0b00000000,
     0b00011000,
     0b00000000,
     0b00000000],
    # 7: '+'
    [0b00000000,
     0b00011000,
     0b00011000,
     0b01111110,
     0b00011000,
     0b00011000,
     0b00000000,
     0b00000000],
    # 8: '='
    [0b00000000,
     0b00000000,
     0b01111110,
     0b00000000,
     0b01111110,
     0b00000000,
     0b00000000,
     0b00000000],
    # 9: 'i' (lowercase)
    [0b00011000,
     0b00000000,
     0b00011000,
     0b00011000,
     0b00011000,
     0b00011000,
     0b00000000,
     0b00000000],
    # 10: 't'
    [0b00011000,
     0b01111110,
     0b00011000,
     0b00011000,
     0b00011000,
     0b00001100,
     0b00000000,
     0b00000000],
    # 11: 'n'
    [0b00000000,
     0b01100110,
     0b01110110,
     0b01101110,
     0b01100110,
     0b01100110,
     0b00000000,
     0b00000000],
    # 12: 'o'
    [0b00000000,
     0b00111100,
     0b01100110,
     0b01100110,
     0b01100110,
     0b00111100,
     0b00000000,
     0b00000000],
    # 13: 'x'
    [0b00000000,
     0b01100110,
     0b00111100,
     0b00011000,
     0b00111100,
     0b01100110,
     0b00000000,
     0b00000000],
    # 14: '#'
    [0b00100100,
     0b01111110,
     0b00100100,
     0b00100100,
     0b01111110,
     0b00100100,
     0b00000000,
     0b00000000],
    # 15: '@'
    [0b00111100,
     0b01100110,
     0b01101110,
     0b01101010,
     0b01101110,
     0b01100000,
     0b00111110,
     0b00000000],
]

assert len(CHARS) == 16, f"Expected 16 characters, got {len(CHARS)}"

CELL = 8           # pixels per character cell side
GRID = 16          # number of cells per row/column
SIZE = CELL * GRID # total image size (128×128)

img = Image.new("L", (SIZE, SIZE), 0)  # greyscale, black background
pixels = img.load()

for char_idx, rows in enumerate(CHARS):
    col = char_idx          # all 16 chars in a single row
    row = 0
    cell_x = col * CELL
    cell_y = row * CELL
    for py, row_bits in enumerate(rows):
        for px in range(CELL):
            bit = (row_bits >> (7 - px)) & 1
            pixels[cell_x + px, cell_y + py] = 255 if bit else 0

# Save as 8-bit grayscale PNG — the asset loader converts to f16 at upload time.
out_path = "ascii_char_map_16x16.png"
img.save(out_path)
print(f"Saved {out_path} ({SIZE}×{SIZE} px, 16 chars in one row)")
