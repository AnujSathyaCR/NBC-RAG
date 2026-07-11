import json
from collections import defaultdict

with open("output/nbc_full.json", encoding="utf-8") as f:
    data = json.load(f)

pages = data["pages"]
chunks = [json.loads(l) for l in open("output/chunks.jsonl", encoding="utf-8")]

chunked_pages = set(c["page"] for c in chunks)
all_pages = set(p["page"] for p in pages)
missing = sorted(all_pages - chunked_pages)

print(f"Total pages in JSON:  {len(all_pages)}")
print(f"Pages with chunks:    {len(chunked_pages)}")
print(f"Pages with NO chunks: {len(missing)}")
print(f"Sample missing pages: {missing[:10]}")

# Inspect a missing page
for p in pages:
    if p["page"] == missing[0]:
        print(f"\n--- Page {missing[0]} ---")
        print(f"  kind:        {p.get('kind')}")
        print(f"  page_type:   {p.get('page_type')}")
        print(f"  body length: {len(p.get('body', ''))}")
        print(f"  body[:300]:  {repr(p.get('body', '')[:300])}")
        break