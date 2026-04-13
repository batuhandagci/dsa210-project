# DSA 210 — On Repeat: Do I Loop More Music During Exams?

**Batuhan Dağcı | 34059**

## Research Question

Do I loop music more during exam periods compared to regular days?

**Metric:** Daily loop ratio = proportion of plays on a given day that repeat a track already heard that same day.

## Hypothesis

- **H₀:** Loop ratio during exam days = loop ratio during regular days
- **H₁:** Loop ratio during exam days > loop ratio during regular days

## Dataset

Personal Spotify streaming history enriched with period labels (`regular`, `pre_exam`, `exam_day`) and audio-feature proxies. Located at `data/streaming_clean.csv`.

## Project Structure

```
dsa210-project/
├── data/
│   └── streaming_clean.csv   # Streaming history with features
├── figures/                  # Generated plots (auto-populated by notebook)
├── eda.ipynb                 # EDA and hypothesis testing notebook
├── requirements.txt          # Python dependencies
└── README.md
```

## How to Run

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the notebook**

   ```bash
   jupyter notebook eda.ipynb
   ```

   Or run end-to-end without a browser:

   ```bash
   jupyter nbconvert --to notebook --execute eda.ipynb --output eda.ipynb
   ```

## Key Results

| Comparison | Mean (exam) | Mean (regular) | p-value | Cohen's d |
|---|---|---|---|---|
| Exam Day vs Regular | 0.390 | 0.236 | **0.030** | 0.674 |
| Pre-Exam vs Regular | 0.278 | 0.236 | 0.075 | 0.248 |

The Mann-Whitney U test shows a statistically significant increase in loop ratio on exam days (p = 0.030, α = 0.05), with a medium-to-large effect size (d = 0.674). The pre-exam period shows a trend but does not reach significance.

## Dependencies

- Python 3.8+
- pandas, numpy, matplotlib, seaborn, scipy
