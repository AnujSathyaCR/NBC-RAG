"""
Extract every embedded image from a PDF into page-wise folders:

    output/images_v2/page_001/page_001_image_01.png
    output/images_v2/page_001/page_001_image_02.jpeg
    ...

Usage:
    python extract_images.py
"""

from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent.parent

PDF_PATH = ROOT / "output" / "books" / "NBC_2016_Volume_2.pdf"

OUTPUT_DIR = ROOT / "output" / "images_v2"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

doc = fitz.open(PDF_PATH)

saved = 0
pages_with_images = 0

for page_num in range(len(doc)):
    page = doc[page_num]
    images = page.get_images(full=True)

    if not images:
        continue

    pages_with_images += 1
    page_dir = OUTPUT_DIR / f"page_{page_num + 1:03d}"
    page_dir.mkdir(parents=True, exist_ok=True)

    seen_xrefs = set()
    img_idx = 0

    for img in images:
        xref = img[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)

        img_idx += 1
        base_image = doc.extract_image(xref)
        ext = base_image["ext"]
        filename = f"page_{page_num + 1:03d}_image_{img_idx:02d}.{ext}"
        out_path = page_dir / filename
        out_path.write_bytes(base_image["image"])

        print(f"Saved: {out_path.relative_to(ROOT)}")
        saved += 1

print()
print(f"Images saved: {saved}")
print(f"Pages with images: {pages_with_images}")
print(f"Output: {OUTPUT_DIR}")
