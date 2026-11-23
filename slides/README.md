# AI & Machine Learning — Final Project Slides

Modular **Slidev** deck (`16:9`, light academic theme, Vietnamese slides & speaker notes) paired with a Jupyter Notebook handoff contract.

## Quick Start (Bun)

```bash
cd final-project/slides
bun install
bun run dev
```

- Press **`p`** in the browser (or open `/presenter`) for **Presenter Mode** with Vietnamese speaker notes and the 8-minute timer cue.
- Build static SPA:
  ```bash
  bun run build
  ```
- Export PDF backup (recommended before presentation day):
  ```bash
  bun run export
  # Output: exports/final-project-slides.pdf
  ```

## Jupyter Notebook Handoff (`scripts/slide_export.py`)

Teammates training models in Jupyter can export light-mode 300-DPI figures and test metrics directly into `public/figures/` and `data/results.json`:

```python
import sys
sys.path.append("../slides/scripts")  # adjust relative path to final-project/slides/scripts
from slide_export import apply_slide_theme, save_figure, log_test_metrics

apply_slide_theme()
# fig, ax = plt.subplots(...)
save_figure(fig, "class_distribution")
log_test_metrics("ResNet18", y_test, y_pred, class_names=classes)
```

Standard figure filenames expected by `<FigureSlot>` in `pages/*.md`:
- `class_distribution.png` (Slide 3)
- `dataset_samples.png` (Slide 3)
- `training_curves.png` (Slide 7)
- `confusion_matrices.png` (Slide 8)
- `error_analysis.png` (Slide 9)
- `preprocessing_impact.png` (Slide 10)

Once figures and `data/results.json` are ready, see [AGENTS.md](./AGENTS.md) for instructions on populating the slides (manually or with an AI agent).
