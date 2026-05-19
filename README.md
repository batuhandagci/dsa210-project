# On Repeat: Do I Loop More Music During Exams?

**DSA 210 — Introduction to Data Science**
**2025–2026 Spring Term · Sabancı University**
**Batuhan Dağcı | 34059**

A personal data science project investigating whether music-listening repetition behaviour changes during academic exam periods, using one year of personal Spotify streaming history.

---

## 1. Motivation

Like many students, I have an intuition that my music habits shift under academic pressure. During exam weeks I feel like I stop exploring new music and instead keep replaying a small set of familiar tracks — putting songs "on repeat" while studying. This project replaces that intuition with evidence.

> **Research question:** Do I loop music more during exam periods compared to regular days?

**Metric — daily loop ratio:** the proportion of a day's plays that repeat a track already heard earlier that same day. A loop ratio of 0 means every play that day was a distinct track; a ratio of 0.5 means half the plays were repeats.

This is a personal-data project (Spotify history), one of the project types explicitly encouraged by the course guidelines.

---

## 2. Data Source

### 2.1 Primary data — Spotify streaming history

The primary data is my personal Spotify streaming history, requested through Spotify's privacy portal (`spotify.com/account/privacy`). Spotify provides this as JSON files containing one record per track play, with track name, artist name, timestamp, and milliseconds played.

After parsing and cleaning, the dataset covers **April 2025 – April 2026** and contains **14,628 individual track plays**. Each row in `data/streaming_clean.csv` is one play, with the following fields:

| Field | Description |
|---|---|
| `ts`, `date`, `hour` | Timestamp of the play, derived date and hour |
| `artistName`, `trackName` | Track identity |
| `msPlayed` | Milliseconds the track was played |
| `day_of_week`, `day_of_week_num` | Day of week |
| `period` | Period label (see 2.2) |

### 2.2 Enrichment — exam schedule labelling

The streaming data was enriched with my personal **Sabancı University Fall 2025 exam schedule**. Each calendar day was labelled into one of three periods:

- **`exam_day`** — a day on which I had an exam (including the makeup exam I took after an illness)
- **`pre_exam`** — the three days immediately preceding an exam day
- **`regular`** — any other day

Exam dates used (Fall 2025): PSY201 MT (Nov 7), MATH203 MT1 (Nov 8), ENS208 MT (Nov 9), HUM202 MT (Nov 19), CS201 MT (Nov 22), ENS208 MT2 (Dec 13), MATH203 MT2 (Dec 20), PSY201 Final (Dec 26), MATH203 Final (Jan 8), HUM202 Final (Jan 11), CS201 Final (Jan 14), and the MATH203 makeup (Jan 14).

The period label is the key independent variable: it lets every day in the streaming history be compared as exam-related or regular.

---

## 3. Data Analysis

The analysis follows the full data science pipeline across two notebooks.

### 3.1 Exploratory data analysis & hypothesis testing — `eda.ipynb`

**Daily loop ratio** was computed for every day in the dataset. EDA visualised loop ratio over time, its distribution by period, and listening patterns by hour of day.

**Hypothesis test.** To test whether loop ratio differs between exam days and regular days:

- **H₀:** loop ratio on exam days = loop ratio on regular days
- **H₁:** loop ratio on exam days > loop ratio on regular days

Because daily loop ratios are not normally distributed and the exam-day sample is small, a one-sided **Mann-Whitney U test** was used (a non-parametric test that does not assume normality). The same test was repeated for the pre-exam period.

### 3.2 Machine learning — `ml.ipynb`

The ML stage extends the hypothesis test by asking a harder question: can a day's *period* be predicted from its overall listening behaviour, and which behavioural features carry the signal?

**Time window.** The ML analysis is restricted to the **Fall 2025 academic semester (September 15, 2025 – January 31, 2026)**. This excludes summer/break days, where a "regular" day reflects the absence of school rather than the absence of exam stress, and it produces a less degenerate class balance.

**Daily feature engineering.** Track-level plays were aggregated into one row per day (138 days after dropping days with fewer than 3 plays). Seventeen behavioural features were engineered, grouped as:

- *Repetition:* `loop_ratio`, `unique_tracks`, `unique_artists`, `tracks_per_play`, `total_plays`
- *Listening volume:* `avg_ms_played`, `median_ms_played`, `std_ms_played`, `total_listening_minutes`
- *Timing:* `night/morning/afternoon/evening_play_ratio`, `peak_hour`, `hour_entropy`
- *Calendar controls:* `day_of_week_num`, `is_weekend`

The feature set is restricted to behavioural signals — repetition, listening volume, and timing — to keep the model aligned with the loop-focused research question.

**Two classification setups.** Because the class balance is severe (104 regular / 24 pre-exam / 10 exam days), two setups were run in parallel:

- **Binary (primary):** `exam_period` (pre-exam + exam, n=34) vs `regular` (n=104)
- **3-class (secondary):** the original three labels — reported as illustrative only

**Models and validation.** Three models — Logistic Regression, Random Forest, Gradient Boosting — were each compared against a `DummyClassifier` baseline using **5-fold stratified cross-validation**, then evaluated on a held-out 20% test set. Metrics emphasised F1 and ROC-AUC rather than accuracy, since accuracy is misleading under class imbalance. **Permutation importance** was used to identify which features genuinely drive predictions.

---

## 4. Findings

### 4.1 Hypothesis test (EDA)

| Comparison | Mean loop ratio (exam) | Mean (regular) | p-value | Effect size (Cohen's d) |
|---|---|---|---|---|
| Exam Day vs Regular | 0.390 | 0.236 | **0.030** | 0.674 |
| Pre-Exam vs Regular | 0.278 | 0.236 | 0.075 | 0.248 |

On exam days my loop ratio is significantly higher than on regular days (p = 0.030, α = 0.05), with a medium-to-large effect size. **The hypothesis is supported for exam days.** The pre-exam period shows the same direction but does not reach significance — the behavioural shift appears concentrated on exam days themselves rather than the lead-up.

### 4.2 Machine learning

**Binary setup (5-fold CV on training set):**

| Model | F1 (exam) | ROC-AUC | Avg Precision |
|:--|:--|:--|:--|
| Logistic Regression | 0.421 ± 0.115 | 0.708 ± 0.164 | 0.574 ± 0.217 |
| Random Forest | 0.254 ± 0.136 | 0.723 ± 0.061 | 0.543 ± 0.076 |
| Gradient Boosting | 0.345 ± 0.248 | 0.719 ± 0.097 | 0.588 ± 0.114 |
| Dummy (baseline) | 0.209 ± 0.127 | 0.500 ± 0.071 | 0.262 ± 0.046 |

**Best model on held-out test set — Logistic Regression:** F1 = 0.526, ROC-AUC = 0.701, Average Precision = 0.576.

The binary classifier shows a **clear, real signal**: test ROC-AUC of 0.701 is well above the 0.500 random baseline, and test F1 (0.526) more than doubles the Dummy's CV F1 (0.209). Exam-period days are genuinely distinguishable from regular days by listening behaviour alone.

**3-class setup.** The 3-class classifier reached only F1-macro = 0.333 / ROC-AUC = 0.506 on the test set — **below the Dummy baseline (CV F1-macro 0.369)**. With only 2 exam-day samples in the test set, this setup is statistically underpowered: there is not enough data to reliably separate `pre_exam` from `exam_day`. This is reported honestly as a negative result.

**Which features matter (permutation importance, binary setup):**

| Feature | Importance (mean decrease in F1) |
|---|---|
| `unique_artists` | 0.166 ± 0.071 |
| `total_listening_minutes` | 0.127 ± 0.065 |
| `loop_ratio` | 0.126 ± 0.048 |
| `tracks_per_play` | 0.126 ± 0.048 |
| `unique_tracks` | 0.093 ± 0.080 |

`loop_ratio` ranks third. Its exact tie with `tracks_per_play` is structural, not coincidental: by construction `tracks_per_play = 1 − loop_ratio`, so the two are perfectly anti-correlated and the model treats them as one feature. Read together, the repetition signal ranks third overall, behind artist diversity and listening volume.

### 4.3 Key insights

1. **Exam stress measurably changes how I listen.** Both the hypothesis test and the ML classifier independently confirm a behavioural shift on exam days — the central question is answered *yes*.

2. **Repetition is real but not the whole story.** Loop ratio is significant on its own (EDA) and a top-3 predictor in the ML model. But `unique_artists` carries an even stronger signal: on exam days I don't just repeat tracks, I narrow my listening to fewer artists overall. A univariate test and a multivariate model answer different questions, and both are informative.

3. **The lead-up to exams looks like regular life; exam days don't.** Pre-exam days were not significantly different from regular days, and the 3-class model could not isolate them. The behavioural change is sharp and concentrated on exam days, not a gradual build-up.

---

## 5. Limitations and Future Work

### Limitations

- **Single user, single semester.** All data is from one person over one Fall semester (138 modelled days, only 10 exam days). Cross-validation estimates therefore carry high variance, and nothing here generalises to other students.
- **Loop ratio is a simplified metric.** It counts any same-day repeat as a loop regardless of how many times a track is replayed, and it cannot distinguish active replaying from passively leaving a playlist running.
- **3-class classification is underpowered.** With 24 pre-exam and 10 exam days, the model cannot reliably separate the two exam-related classes — only the binary exam-vs-regular distinction is trustworthy.
- **Temporal labelling.** Exam sessions cluster in specific calendar weeks, so calendar features (`is_weekend`, `day_of_week_num`) may partly act as proxies for exam timing rather than capturing pure behavioural change.

### Future Work

- **More semesters of data.** Pooling several academic years would multiply the exam-day sample and stabilise both the hypothesis test and the ML estimates.
- **Finer-grained modelling.** Instead of daily aggregation, modelling the decision to replay a specific track within a listening session (a sequence or survival model) would capture repetition dynamics more directly.
- **Richer stress signal.** Cross-referencing with other personal data (calendar load, sleep, step count) would test whether loop ratio tracks general stress, not just exams.

---

## 6. Repository Structure

```
dsa210-project/
├── data/
│   ├── streaming_clean.csv     # Track-level streaming history (14,628 plays)
│   └── streaming_daily.csv     # Daily aggregated features (generated by ml.ipynb)
├── figures/                    # Generated plots (auto-populated by notebooks)
├── eda.ipynb                   # EDA and hypothesis testing
├── ml.ipynb                    # ML pipeline (feature engineering → CV → evaluation)
├── requirements.txt            # Python dependencies
└── README.md                   # This report
```

## 7. How to Reproduce

```bash
# Install dependencies
pip install -r requirements.txt

# Run the EDA and hypothesis-testing notebook
jupyter nbconvert --to notebook --execute eda.ipynb --output eda.ipynb

# Run the ML pipeline
jupyter nbconvert --to notebook --execute ml.ipynb --output ml.ipynb
```

All randomness is seeded (`random_state=42`) for reproducibility. Figures are written to `figures/` automatically.

**Dependencies:** Python 3.8+, pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, tabulate.

---

## 8. AI Assistance Disclosure

AI tools are used only for writing, getting help when pushing to github, assistance and debugging.

All statistical results, figures, and numbers in this report were produced by running the code in this repository.
