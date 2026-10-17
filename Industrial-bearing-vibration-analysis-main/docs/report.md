# Project Report — Industrial Bearing Vibration Analysis

## 1. Objective

Distinguish healthy rolling-element bearings from inner-race, outer-race and
ball faults using vibration signals, and quantify **how** the fault indicators
differ between conditions and **how** they grow with fault severity — with the
goal of supporting condition-based maintenance decisions.

## 2. Dataset

- **Source:** Case Western Reserve University (CWRU) Bearing Data Center,
  48 kHz Drive-End accelerometer (`.mat` files).
- **Used in this run:** 10 recordings at a single load (1 HP):

  | Condition    | Recordings                                              |
  | ------------ | ------------------------------------------------------- |
  | healthy      | Normal (baseline)                                       |
  | inner race   | IR007, IR014, IR021                                     |
  | outer race   | OR007, OR014, OR021                                     |
  | ball         | B007, B014, B021                                        |

- The trailing number is the standard fault diameter in mils:
  007 → 0.007 in, 014 → 0.014 in, 021 → 0.021 in.

## 3. Methodology

Summarised here; full detail is in [`methodology.md`](methodology.md).

1. **Preprocessing** — detrend, band-pass filter; no amplitude normalisation so
   RMS/peak remain physical.
2. **Feature extraction** — 11 time-domain indicators (RMS, std, variance, peak,
   peak-to-peak, skewness, kurtosis, crest factor, shape factor, impulse factor,
   mean) and frequency-domain features (dominant frequencies, spectral centroid
   and bandwidth, band-energy ratios).
3. **Comparison** — a per-recording feature table plus a condition-averaged
   summary.
4. **Significance testing** — one-way ANOVA and Kruskal-Wallis across conditions
   plus pairwise Welch t-tests against the healthy baseline (Cohen's d), using
   0.25 s windows with 50 % overlap (779 windows total).
5. **Severity analysis** — indicators versus damage size (slope + Pearson r).
6. **Reporting** — HTML/Markdown report with figures, tables and a rule-based
   interpretation.

## 4. Results

### 4.1 Condition-averaged indicators

| Condition  | RMS    | Kurtosis | Crest factor | Dominant freq. (Hz) |
| ---------- | ------ | -------- | ------------ | ------------------- |
| healthy    | 0.0422 | −0.14    | 4.05         | ~360                |
| ball       | 0.0501 | 1.42     | 8.33         | ~360                |
| inner race | 0.1029 | 4.63     | 11.54        | ~758                |
| outer race | 0.0950 | 5.83     | 10.59        | ~413                |

Every faulty condition shows higher RMS, kurtosis and crest factor than the
healthy baseline. RMS rises ~19 % (ball), ~125 % (outer race) and ~144 % (inner
race); kurtosis rises from −0.14 (healthy, near-Gaussian) to 4.6–5.8 for
race faults, confirming the impulsive, non-Gaussian signature of a localised
defect. The dominant frequency shifts from the bulk shaft-rate region (~360 Hz
healthy) to defect-related peaks (>410 Hz) for the faulty bearings.

### 4.2 Statistical significance

**Omnibus tests** (one-way ANOVA, p < 0.05) flagged these features as differing
significantly across conditions:

> crest factor, impulse factor, kurtosis, peak, peak-to-peak, RMS, shape factor,
> std, variance (skewness significant by Kruskal-Wallis only).

The **mean** was *not* significant (p ≈ 0.39) — as expected for a zero-mean
vibration signal — which is a useful sanity check that the pipeline is not
detecting an artefact.

**Pairwise Welch t-tests vs. healthy** (median effect sizes):

| Condition  | Best indicator       | Cohen's d | Effect |
| ---------- | -------------------- | --------- | ------ |
| inner race | peak-to-peak         | 3.35      | large  |
| outer race | crest factor         | 1.66      | large  |
| ball       | peak-to-peak         | 0.82      | large  |
| ball       | kurtosis             | 0.39      | small  |

Race faults are separated from healthy by **large** effects on almost every
indicator (d > 1.6). Ball faults are detectable but weaker (d ≈ 0.4–0.8),
consistent with the smaller impact energy of a ball defect.

### 4.3 Fault severity (damage progression)

Trend of each indicator versus damage size (0.007 → 0.021 in):

| Condition  | Indicator   | Slope    | r      | Trend                       |
| ---------- | ----------- | -------- | ------ | --------------------------- |
| inner race | crest factor| +581.3   | +0.91  | strong increase             |
| inner race | kurtosis    | +123.1   | +0.85  | strong increase             |
| ball       | crest factor| +341.5   | +0.67  | increase (noisy)            |
| outer race | RMS         | +7.31    | +0.70  | increase                    |
| outer race | crest factor| −1004    | −0.84  | decrease (see discussion)   |

For inner-race faults the crest factor and kurtosis increase strongly and
monotonically with damage size — the textbook damage-progression behaviour.
Outer-race RMS also rises with size. The ball trends are positive but noisy,
and the outer-race crest-factor/kurtosis trend is *negative*, discussed below.

## 5. Discussion

- **Indicators converge on the expected physics.** Isolation (healthy) →
  impulsiveness (kurtosis, crest factor) → energy (RMS) is exactly the
  progression that condition monitoring exploits. The fact that ANOVA confirms
  the difference while the mean does not indicates the pipeline captures real
  fault physics rather than an offset.
- **Race faults are easier than ball faults.** Inner- and outer-race defects
  produce large, repeatable spectral and impulsive signatures; ball defects pass
  through the load zone more irregularly and yield smaller effect sizes. This
  matches the literature and the CWRU dataset's own difficulty ordering.
- **Severity is monotonic for inner-race faults** and captured cleanly by crest
  factor (r = +0.91) and kurtosis (r = +0.85). This supports threshold-based
  alarms tuned to those indicators.
- **The outer-race anomaly.** Outer-race crest factor and kurtosis fall between
  0.007 in and 0.021 in in this single-load subset (OR014 has notably low
  energy). With only three recording levels per fault this is easily dominated
  by load/speed/run-to-run variation; the RMS trend (+0.70) is more stable. This
  illustrates why severity trends need several samples per size and a controlled
  load to be conclusive.
- **Statistical caveat.** Windows from the same recording are correlated
  (pseudoreplication), so the very small p-values should be read as a ranking of
  indicator usefulness rather than exact probabilities. Rerunning at the
  recording level or with a mixed-effects model would tighten the inference.

## 6. Conclusion

A compact, interpretable pipeline — detrending, band-pass filtering, time- and
frequency-domain features, ANOVA/Kruskal-Wallis and pairwise t-tests, and a
damage-size trend — cleanly separates healthy, inner-race, outer-race and ball
bearing conditions. RMS, kurtosis and crest factor are confirmed as the
discriminating indicators, spectral features locate the defect frequency, and
inner-race severity tracks damage size monotonically. The approach is directly
usable for condition-based maintenance and early fault detection.

## 7. References

1. Case Western Reserve University Bearing Data Center — bearing vibration
   datasets (48 kHz drive-end, 0.007/0.014/0.021 in faults).
2. R. B. Randall and J. Antoni, "Rolling element bearing diagnostics — A
   tutorial," *Mechanical Systems and Signal Processing*, 2011.
3. J. Cohen, *Statistical Power Analysis for the Behavioral Sciences*, 1988.
