# AI & Machine Learning — Final Project Slides

Modular **Slidev** deck (`16:9`, light academic theme, Vietnamese slides & speaker notes) for **Vietnam Housing 2024 House Price Prediction (Regression)**.

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
- Export PDF backup:
  ```bash
  bun run export
  # Output: exports/final-project-slides.pdf
  ```

## Jupyter Notebook Handoff (`scripts/slide_export.py`)

```python
import sys
sys.path.append("../slides/scripts")
from slide_export import apply_slide_theme, save_figure, log_regression_metrics

apply_slide_theme()
# fig, ax = plt.subplots(...)
save_figure(fig, "price_distribution")
log_regression_metrics("HistGradientBoosting", y_test, y_pred)
```

Standard figure filenames expected by `<FigureSlot>` in `pages/*.md`:
- `price_distribution.png` (Slide 3)
- `dataset_samples.png` (Slide 3)
- `training_curves.png` (Slide 7)
- `pred_vs_actual.png` (Slide 8)
- `residual_analysis.png` (Slide 9)
- `preprocessing_impact.png` (Slide 10)
