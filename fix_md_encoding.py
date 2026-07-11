from pathlib import Path

PAGES_DIR = Path("../output/pages")

fixed = 0

for md in PAGES_DIR.glob("page_*.md"):
    try:
        # Already UTF-8
        md.read_text(encoding="utf-8")

    except UnicodeDecodeError:
        try:
            # Read as Windows encoding
            text = md.read_text(encoding="cp1252")

            # Rewrite as UTF-8
            md.write_text(text, encoding="utf-8")

            print(f"Fixed: {md.name}")
            fixed += 1

        except Exception as e:
            print(f"Failed: {md.name} -> {e}")

print(f"\nDone. Fixed {fixed} files.")