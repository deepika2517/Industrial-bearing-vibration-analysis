# Methodology

Signal-processing and analysis pipeline used by the toolkit, with the feature
definitions and the statistical / severity methods.

## 1. Pipeline overview

```
raw signal (MAT / CSV / NPZ)
   │
   ├─ remove DC / detrend            (src.preprocessing)
   ├─ band-pass filter               (src.preprocessing)
   ├─ optional normalisation         (src.preprocessing)
   │
   ├─ time-domain features           (src.statistical_analysis)
   ├─ frequency-domain features      (src.fft_analysis)
   │
   ├─ condition comparison           (src.comparison)
   │     per-recording table + condition-averaged summary
   │
   ├─ significance testing           (src.statistical_tests)
   │     ANOVA / Kruskal-Wallis + pairwise Welch t-tests
   │
   ├─ severity / damage progression  (src.severity)
   │
   └─ report generation              (src.reporting)
         report.html / report.md  +  figures + CSV
```

## 2. Data loading and preprocessing

Each recording is loaded into a `VibrationSignal(name, signal, fs, condition)`.
For CWRU `.mat` files the `*_DE_time` channel is preferred; when a filename
embeds a reference number (e.g. `IR014_1_175.mat` → `X175_DE_time`) that
channel is selected to avoid picking up stray variables.

Preprocessing (`src.preprocessing`) applies:

1. **Detrending** — removes the DC component and slow drift.
2. **Band-pass filtering** — a Butterworth band-pass isolates the informative
   vibration band and rejects sensor/electrical noise.
3. **Normalisation** (optional) — z-score scaling; disabled for the main
   comparison so absolute amplitude (RMS, peak) stays physically meaningful.

## 3. Time-domain features

| Feature           | Definition                                            | Fault sensitivity                  |
| ----------------- | ----------------------------------------------------- | ---------------------------------- |
| Mean              | arithmetic average                                    | sensor bias / DC                   |
| RMS               | √(mean(x²))                                           | overall energy increase            |
| Variance / std    | dispersion about the mean                             | energy increase                    |
| Peak              | max(|x|)                                              | shock amplitude                    |
| Peak-to-peak      | max(x) − min(x)                                       | total excursion                    |
| Skewness          | third standardised moment                             | asymmetric impacts                 |
| Kurtosis          | fourth standardised moment (excess)                   | impulsive faults (early detection) |
| Crest factor      | peak / RMS                                            | isolated spikes                    |
| Shape factor      | RMS / mean(|x|)                                       | waveform shape change              |
| Impulse factor    | peak / mean(|x|)                                      | repeated impacts                   |

Kurtosis and crest factor are the classic early-warning indicators because a
localised defect produces impulsive shocks that raise the tails of the
amplitude distribution long before overall RMS rises much.

## 4. Frequency-domain features

Computed with a NumPy FFT (`src.fft_analysis`):

- **Dominant frequencies** — the `top_n` spectral peaks.
- **Spectral centroid** — amplitude-weighted mean frequency; rises when energy
  shifts to higher bands.
- **Spectral bandwidth** — spread of the spectrum about the centroid.
- **Band-energy ratios** — fraction of energy in low / mid / high bands, which
  localises defect-frequency energy.

## 5. Statistical significance testing

`src.statistical_tests` turns "does this indicator differ between conditions?"
into a testable hypothesis. Each recording is cut into overlapping windows
(0.25 s, 50 % overlap) so every condition contributes many observations.

- **Omnibus test** — one-way **ANOVA** (parametric) and **Kruskal-Wallis**
  (rank-based, robust to non-normality). A significant result means at least one
  condition differs.
- **Pairwise test** — **Welch's t-test** (unequal variances) comparing each fault
  condition against the healthy baseline, reported with **Cohen's d**:

  | |d|      | effect     |
  | -------- | ---------- |
  | ≥ 0.8    | large      |
  | 0.5–0.8  | medium     |
  | 0.2–0.5  | small      |
  | < 0.2    | negligible |

## 6. Severity / damage-progression analysis

`src.severity` recovers the CWRU damage diameter from the recording code
(`IR007` → 0.007 in, `OR014` → 0.014 in, `B021` → 0.021 in) and:

- tabulates the indicators for every faulty recording with a known size, and
- fits a line of each indicator versus damage size, reporting the **slope** and
  **Pearson correlation** `r`.

A positive slope with strong positive `r` means the indicator tracks fault
severity — it grows as the defect gets larger.

## 7. Interpretation and reporting

`src.reporting` assembles the figures and tables into a self-contained
`report.html` (figures embedded as base64) and a `report.md`, and adds a
rule-based interpretation: the healthy baseline, extreme recordings, per-fault
observations, the statistically significant features and the severity trend.

## 8. Limitations

- **Pseudoreplication.** Windows cut from one recording are *not* independent, so
  the per-window p-values understate the true uncertainty. They are best read as
  a ranking of which indicators separate the conditions, not as exact
  probabilities. A recording-level or mixed-effects model would be the rigorous
  alternative.
- **Load and speed.** Comparisons are most reliable at a fixed shaft speed and
  load; CWRU files at 0–3 HP and several speeds are otherwise not directly
  interchangeable.
- **Single channel.** Only one accelerometer channel (DE) is analysed; combining
  DE + FE channels would improve fault localisation.
- **Severity sample size.** Each fault type has only three damage sizes, so the
  trend slope is indicative rather than statistically strong.
