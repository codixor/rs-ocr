"""Render synthetic printed OCR fixtures; no real account or financial data."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

for name, title, lines in [
    ("invoice", "SAMPLE INVOICE", ["MERCHANT: EXAMPLE STORE", "INVOICE NO: 0007", "DATE: 2026-10-29",
                                   "CURRENCY: AED", "SUBTOTAL: 110.00", "VAT: 5.50", "TOTAL: 115.50"]),
    ("cheque", "SAMPLE - NOT VALID FOR PAYMENT", ["BANK: EXAMPLE BANK", "CHEQUE NO: 000042",
                                                 "PAYEE: ALICE EXAMPLE", "DATE: 2026-10-29", "AMOUNT: 1234.56"]),
]:
    image = Image.new("RGB", (1250, 750), "white")
    draw = ImageDraw.Draw(image)
    draw.text((50, 35), title, font=ImageFont.truetype(FONT_BOLD, 43), fill="black")
    for index, text in enumerate(lines):
        draw.text((60, 120 + index * 76), text, font=ImageFont.truetype(FONT, 40), fill="black")
    output = ROOT / "tests/fixtures" / f"{name}.png"
    image.save(output)
    print(output)
