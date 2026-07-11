# NBC 2016 — Page Extraction Prompt

Extract one PDF page into YAML frontmatter + Markdown body. Return nothing else — no preamble, commentary, or summaries. Null/empty fields: write `null` or `[]`. Never omit a field.

---

## YAML FRONTMATTER

```yaml
---
page: <integer>
kind: <list: clause|table|figure|drawing|formula|annex|note|continuation|mixed>
continues_from_prev: <true|false>
continues_to_next: <true|false>

parts_on_page: <list, e.g. ["PART 8"]>
sections_on_page: <list, e.g. ["Section 2"]>
clauses_on_page: <list in appearance order, e.g. ["B-9","B-9.2","B-9.2.2.1"]>

figures_on_page: <list, e.g. ["Fig. 74","Fig. 75"]>

figure_clause_map:           # [] if no figures
  - figure: "Fig. 74"
    parent_clause: "B-9.2.2" # closest preceding clause heading (not Part/Section)

figure_metadata:             # [] if no figures; id+caption must match body exactly
  - id: "Fig. 74"
    caption: "Accessible Toilet Room — Type A"  # null if absent

figure_assets:               # [] if no figures; image_index = order on page (1,2,3…), NOT figure number
  - figure: "Fig. 74"        # pipeline path: output/images/page_PPPP/page_PPPP_img_II.png
    image_index: 1           # do NOT include image_path here

xrefs: <list of all cross-references exactly as printed>
defined_terms: <list of formally defined terms on this page>

has_table: <true|false>
has_figure: <true|false>
has_drawing: <true|false>
has_equations: <true|false>

page_type: <standard|formula_heavy>
# formula_heavy: page dominated by equations, derivations, variable definitions,
#   structural/geotechnical/load calculations, formula annexes.
# standard: all other pages. A page with 1–2 inline equations remains standard.
formula_topic: <engineering topic string, e.g. "Foundation Settlement"; null if standard>

notes: <extractor observations only; null if none>
---
```

---

## BODY RULES

**Faithfulness:** Extract verbatim. Do not invent, summarise, paraphrase, or reorder. Do not merge clauses.

**Strip:** page numbers, running headers/footers, printer marks, copyright notices.

---

## HIERARCHY → HEADINGS

| Level | Example | Markdown |
|---|---|---|
| Part | `PART 8 — BUILDING SERVICES` | `#` |
| Section / Annex | `Section 2 —…` / `Annex B —…` | `##` |
| Top clause | `B-9 —…` | `###` |
| Sub-clause 1 | `B-9.2 —…` | `####` |
| Sub-clause 2 | `B-9.2.2 —…` | `#####` |
| Sub-clause 3+ | `B-9.2.2.1` | `######` / bold inline |

**Identifier rules (CRITICAL):** Preserve exactly as printed. Never renumber, strip prefixes (`B-`, `C-`), or normalise (`Section 2` → `2`). Heading text = `<identifier> — <title>` as printed.

**Continuations:** If page starts mid-clause, do not fabricate a heading. Set `continues_from_prev: true`. Use: `<!-- continuation of B-9.2.2 -->`

**Annex identifiers** (e.g. `B-1`, `B-9.2.2`) carry the annex-letter prefix — preserve it.

---

## FIGURES (regulatory content — never skip)

**Completeness rule:** A figure is NOT extracted unless at least one of labels / dimensions / legend / annotations / callouts is extracted. Heading alone is insufficient. If nothing can be extracted, flag in `notes`.

**OCR all text inside the figure:** labels, dimensions, room/equipment names, abbreviations, callouts, legend entries, scale annotations, north arrow, notes.

**Dual-location rule:** Every figure id and caption must appear in both YAML (`figure_metadata`, `figure_clause_map`) and Markdown body. Required for PDF image cropping and citation display.

**Position rule:** Place figures inline immediately after the clause paragraph that references them. Do not group at page end.

**Extract per figure in this order:**

1. Heading: `#### Fig. 74 — Title` (depth matching clause level)
2. Caption as `>` blockquote (if a separate caption line exists)
3. `**Figure labels:**` — ordered/unordered list
4. `**Dimensions:**` — bullet list; preserve units and spacing exactly (`1 500` not `1500`)
5. `**Legend:**` — table or list
6. `**Figure notes:**` — bullet list

**Drawings** (floor plans, fire layouts, schematics, stair/ramp, equipment, site plans): open with `**[DRAWING: Type — Fig. N]**`, then extract Labels, Dimensions, Legend, Notes in that order.

**If geometry cannot be encoded in text:** extract all text first, then add:
`<!-- figure present: graphical content not represented in text -->`
Set `has_drawing: true`.

---

## TABLES

- GFM pipe table for simple tables; HTML `<table>` only for merged cells.
- Preserve all rows, columns, cells (empty cell = `| |`), header rows, units, footnote markers.
- Caption: `**Table 12 — Title**` immediately above.
- Notes: `**Table notes:**` list immediately below. Place table inline after its governing clause.

---

## NOTES / CAUTIONS / WARNINGS

Render as blockquotes with bold label. Preserve label exactly (do not lowercase):

```
> **NOTE** …
> **NOTES** 1. … 2. …
> **CAUTION** …
> **WARNING** …
> **EXCEPTION** …
```

Never truncate. Preserve numbering (NOTE 1, NOTE 2).

---

## CROSS REFERENCES

Preserve exactly as printed. Collect all in `xrefs`. Render inline as plain text (no hyperlinks).

---

## DEFINED TERMS

Format: `**Term** — definition verbatim`. Add term to `defined_terms`.

---

## FORMULA-HEAVY PAGES

**Trigger:** `page_type: formula_heavy` when page is primarily: equations, structural/geotechnical/load calculations, derivations, variable definition tables, formula annexes. Mixed pages with majority equations qualify. Pages with only 1–2 inline equations remain `standard`.

**Hard constraints — do not:**
- Analyse, derive, prove, simplify, or interpret mathematics
- Perform calculations or validate equations
- Reason through derivations
- Rewrite formulas (except for Markdown formatting)
- Generate lengthy explanations

**Body format for formula_heavy (in order, omit empty sections):**

```
**Summary**
- Engineering purpose (1 sentence)
- Key concept/parameter
- Design context (if relevant)

**Equations**

**Eq. (label)**

    equation in 4-space indented code block

**Variables:**
- symbol = definition [units]

(variables shared with Eq. X if applicable)

**References**
- Clause / Annex / Table / IS standard exactly as printed

**Regulatory Rules**
[verbatim requirements, or: "No regulatory requirements identified on this page."]
```

**Equation notation:**

| Construct | Use |
|---|---|
| Subscript/superscript | `x_i`, `x^2` |
| Fraction | `a/b` |
| Square root | `sqrt(x)` |
| Greek | spell out: `gamma`, `phi`, `sigma`, etc. |
| Multiply | `·` or `×` |
| Summation/integral | `Σ(i=1 to n)`, `∫(a to b)` |
| Units | `[kN/m²]`, `[mm]` |

Preserve all coefficients, exponents, units exactly. For undefined variables: `<not defined on this page>`.

**YAML for formula_heavy:** include `formula` in `kind`; set `has_equations: true`; populate `formula_topic`; apply all figure/table fields normally.

---

## RETRIEVAL RULES

- Figures inline after their referencing clause (not page-end).
- Tables inline after their governing clause.
- `parent_clause` = closest preceding clause heading, not Part/Section.
- These rules enable single-chunk citation popups (clause + figure + table + image).

---

## CHECKLIST

- [ ] All YAML fields present; null/[] where empty
- [ ] `figure_clause_map`: one entry per figure; `parent_clause` is clause-level (not Part/Section)
- [ ] `figure_metadata`: `id` + `caption` match body exactly
- [ ] `figure_assets`: `image_index` = appearance order (not figure number); no `image_path`
- [ ] Figure id + caption in both YAML and body
- [ ] Each figure has ≥1 of: labels/dimensions/legend/annotations/callouts
- [ ] Figure OCR done (labels, dims, callouts, legend, scale, north arrow)
- [ ] Clause identifiers unaltered; no renumbering; prefixes preserved
- [ ] All table rows/columns/cells present
- [ ] NOTE/CAUTION/WARNING/EXCEPTION verbatim with exact label
- [ ] All xrefs collected
- [ ] Boilerplate stripped (page nos, headers, footers, marks)
- [ ] No invented, summarised, or paraphrased content
- [ ] `continues_from_prev`/`continues_to_next` correct
- [ ] Figures/tables inline (not page-end)
- [ ] `<!-- figure present: … -->` only after all text extracted; no `[EXTRACTION NOTE]` anywhere
- [ ] `formula_heavy`: body = Summary→Equations→References→Regulatory Rules; no derivation; `has_equations: true`; `formula_topic` set
- [ ] `standard`: `formula_topic: null`
- [ ] Output = frontmatter + body only