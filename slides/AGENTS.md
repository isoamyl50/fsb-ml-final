# AGENTS.md — AI Agent Guide for the Final Project Slidev Deck

This workspace holds the **Slidev** presentation deck for the AI & Machine Learning final course project, aligned with `../S10. Final Project.pptx.txt`.

## 1. Grading Rubric & Timing Constraints

- **Presentation duration:** **5–10 minutes** (target **~8 minutes** across the 12 main slides; Appendix slides sit outside the main timer for Q&A).
- **Rubric weights (`../S10. Final Project.pptx.txt`):**
  1. Data exploration and preprocessing — **20%** (`pages/02-dataset.md`)
  2. Model selection and implementation — **30%** (`pages/03-models.md`)
  3. Performance analysis and visualization — **20%** (`pages/04-results.md`)
  4. Comparison and insights — **20%** (`pages/05-insights.md` + `pages/06-conclusion.md`)
  5. Presentation and clarity — **10%** (overall structure, visuals, delivery)

## 2. Directory Map

```text
final-project/slides/
├── slides.md                 # Slidev headmatter, Cover (Slide 1), and src: imports
├── pages/
│   ├── 01-intro.md           # Slide 2: Problem, objective, ML project lifecycle
│   ├── 02-dataset.md         # Slides 3-4: Dataset overview + preprocessing pipeline
│   ├── 03-models.md          # Slides 5-6: Model selection + training & tuning setup
│   ├── 04-results.md         # Slides 7-8: Loss/acc curves + test metrics & confusion matrices
│   ├── 05-insights.md        # Slides 9-10: Model comparison/errors + preprocessing impact & challenges
│   ├── 06-conclusion.md      # Slides 11-12: Course reflections/conclusions + Thank You/Q&A
│   └── 07-appendix.md        # Slides 13-14: Detailed per-class report + experiment log (Q&A)
├── components/
│   ├── FigureSlot.vue        # Renders PNG when `src` is passed, clean placeholder otherwise
│   └── MetricCard.vue        # Compact tabular-nums KPI card
├── data/
│   └── results.json          # Structured handoff contract populated from the Jupyter Notebook
├── public/
│   └── figures/              # High-DPI (300 DPI) light-mode PNG plots exported from the notebook
├── scripts/
│   └── slide_export.py       # Python helper to save figures & log metrics into results.json
├── slide-bottom.vue          # Per-slide footer (hidden on cover/end)
├── style.css                 # Scoped `.slidev-layout` light theme styles
└── exports/                  # PDF export output directory (gitignored)
```

## 3. Language Policy

- **Slides, figure captions, table headers/cells, and speaker notes MUST be in clear academic Vietnamese.**
- Keep standard ML terms in English where natural in Vietnamese CS coursework (e.g., `Accuracy`, `Precision`, `Recall`, `F1-score`, `Overfitting`, `Underfitting`, `Data Augmentation`, `Early Stopping`, `CNN`, `Transformer`, `Dropout`, `Transfer Learning`).
- Keep code, scripts, comments, and `AGENTS.md` in English.
- Keep the instructor's name exactly as written in `../S10. Final Project.pptx.txt` (`PhD. Truc Thi Kim, Nguyen`) unless instructed otherwise.

## 4. Data Integrity Rule (Strict)

- **Populate numbers, dataset facts, hyperparameters, and findings ONLY from verified project artifacts:**
  - `data/results.json`
  - PNG plots present in `public/figures/`
  - The team's Jupyter Notebook (`.ipynb`) or training scripts/logs
- **NEVER invent, extrapolate, or guess** sample counts, split percentages, accuracy/F1 scores, or error patterns.
- If an item is missing from the notebook or `results.json`, leave its `[TODO: ...]` marker intact and report what is still missing.

## 5. Step-by-Step Workflow to Populate the Deck

1. **Inspect artifacts first:**
   - Read `data/results.json` and check `ls public/figures/`.
   - If `data/results.json` still has `[TODO]` placeholders, inspect the team's `.ipynb` notebook directly.
2. **Replace `[TODO: ...]` placeholders:**
   - Update `slides.md` (title, dataset name, team members) and `pages/01-intro.md` through `pages/07-appendix.md` in Vietnamese.
   - Remove `<span class="todo-pill">...</span>` wrappers once real content replaces the placeholder.
3. **Wire generated plots into `<FigureSlot>`:**
   - For each PNG present in `public/figures/`, add the matching `src="/figures/<filename>.png"` prop on `<FigureSlot>` (e.g., `src="/figures/class_distribution.png"`).
   - Never set `src` to a file that does not exist yet in `public/figures/` (leaving `src` omitted prevents 404 errors and renders the placeholder card).
4. **Update speaker notes & timing:**
   - Update the `<!-- [MM:SS – MM:SS | ~XXs] ... -->` HTML comment at the bottom of each slide with concrete numbers and talking points in Vietnamese.
   - Keep the total budget for Slides 1–12 between **7.5 and 8.5 minutes** (well within the 5–10 minute limit).
5. **Verify layout, build, and PDF export:**
   - Run `bun run build && bun run export` inside `final-project/slides/`.
   - Confirm zero build/export errors and verify that the exported PDF (`exports/final-project-slides.pdf`) has the expected slide count and no clipped tables/text.

## 6. Slidev Layout & Authoring Hygiene

- **Canvas size:** `980x551` (`16:9`). Keep slides scannable: max **4–5 concise bullets** per card/column and short table cells.
- **Scoped CSS:** Always scope custom styles under `.slidev-layout` in `style.css` so Slidev Presenter Mode is unaffected.
- **Imports (`slides.md`):** Keep an HTML comment line between adjacent `src:` frontmatter blocks so Slidev never merges fences.
- **Markdown inside HTML containers:** When placing Markdown tables or fenced blocks (e.g. `mermaid`) inside a `<div>`, never indent closing `</div>` tags by 4+ spaces after a blank line (CommonMark treats 4-space indentation as a code block). Keep indentation to 0–2 spaces.
- **Mermaid diagrams:** Wrap node labels containing parentheses, slashes, or special punctuation in quotes (`A["Text (info)"]`). Prefer horizontal full-width `flowchart LR` for Mermaid diagrams (and use HTML/Tailwind step cards inside narrow vertical grid columns) so exported PDFs stay crisp without ShadowRoot print clipping.
- **Click animations:** Avoid heavy `v-click` chaining so the exported static PDF backup remains complete and readable without `--with-clicks`.
