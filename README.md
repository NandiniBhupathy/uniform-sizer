# uniform-sizer

Assigns a school uniform size from a girl's height and weight, and hands the uncertain cases to a person for a hand fitting.

I built this to explore one question: can a model take over the easy fittings and be honest about the hard ones? A tool that has to guess for every child will be wrong often enough that nobody trusts it. So the system has two outputs:

- **AUTO**: one size is clearly right, so assign it.
- **FLAG**: the model isn't sure, so someone measures her by hand.

> **Note on the data:** this repo ships with a synthetic data generator, not real student data.
> Measurements and fitted sizes of children aren't something I can publish, so `scripts/generate_synthetic_data.py`
> creates a realistic-looking dataset that lets anyone clone the repo and run the full pipeline.
> The numbers below describe that simulated data, not real girls. Swap in real data (format below) and re-run.

## The setup

- Girls in grades 1-5 (ages 6-11), three schools, about 1,000 per school.
- Inputs: height and weight (grade and age are used for grouping and as an optional feature).
- Label: the size that actually fit, mapped to an ordered index so one size apart is always 1.
- Output: a size (AUTO) or "measure by hand" (FLAG).

## About the synthetic data

`scripts/generate_synthetic_data.py` simulates 3 schools x 5 grades x 4 sections (about 3,000 girls). Height grows with age, weight varies around a typical build, and the fitted size is a function of height and weight plus noise that is larger for atypical builds. It adds small school and section effects, and a few deliberately dirty rows (missing weights, impossible heights, duplicates) so the cleaning step has something to do.

What it's good for: running the pipeline end to end, testing the splits and the AUTO/FLAG logic, and seeing how the trade-offs behave. What it can't do: tell you how accurate this would be on real girls, because I chose how much weight matters and how noisy the labels are.

## How it works

1. **Point model.** Regression on the size index. I compare a median-per-height-band lookup, linear regression and gradient boosting, each with height only, height + weight, and height + weight + age.
2. **Error model.** A small second model predicts how wrong the first one is likely to be for this particular girl, trained on out-of-fold errors grouped by section. Unusual builds get a larger expected error.
3. **Conformal calibration.** On held-back sections, I compute `|true - predicted| / expected error` per grade and take a high quantile (`q`) so that the range `prediction ± q × expected error` contains the true size about 90% of the time.
4. **Decision.** Count the whole sizes inside that range. Exactly one means AUTO. Two or more, or invalid input, means FLAG.

## Evaluation

The part I spent the most care on. Each school is held out as the test set once (leave-one-school-out), because the real use case is a school the model has never seen. Inside the training schools, whole sections go to train or calibration, never individual girls, since girls in one section share a measurer and a cohort. The code asserts that no school, section or student is shared between train, calibration and test.

Results are reported per school, plus breakdowns by grade and by weight-for-height band.

### Results on the synthetic data (3 held-out schools, 90% target coverage)

| | |
|---|---|
| Coverage (true size inside the range) | about 88% |
| Girls assigned AUTO | about 58% |
| Exact size match among AUTO | about 81% |
| Always assign a size, never flag (no abstention) | about 73% exact |
| Grade-only rule | about 40% exact |

Lean and heavy builds get flagged more and have larger errors than typical builds, which is the behaviour I wanted.

## What I found, including the uncomfortable parts

- **The conformal layer didn't beat a simple confidence threshold.** Picking the same 58% of girls by "probability the rounded size is right" gave 81.5% exact match, versus 81.2% for the conformal rule. I'm keeping conformal in the code, but I wouldn't claim it's the reason the abstention works.
- **Coverage is not the same as AUTO accuracy.** 88% coverage sits next to 81% exact match among AUTO, because the guarantee holds across everyone, not among the confident subset I select. For a school, AUTO accuracy is the number that matters.
- **Calibrating on other schools undercovers a new one.** Coverage lands about 1-2 points below target, and one school (which I built with a size offset) is clearly worse than the others.
- **A direct accuracy target helps, but doesn't fix the shift.** In a side experiment, choosing a confidence threshold to hit a target AUTO accuracy raised accuracy as the AUTO share fell (target 90%: about 36% AUTO, about 86% accurate on held-out schools). It still fell short of the target on the shifted school. Calibrating on the new school itself should help, but I haven't tested that.
- **Model selection is thin.** Choosing "the simplest model within one standard deviation of the best" uses a standard deviation over only three folds. I'd trust it less than the output suggests.

## Limitations

- **Results are from simulated data.** The linear models win partly because the generator builds labels from a linear formula, so these numbers show the pipeline works, not that the approach works on real fittings.
- **The label is "the size that fit", not ground truth.** It depends on the tailor, the garment, and whether parents asked for room to grow. That noise limits how accurate any model can be.
- **Weight is an indirect measure.** Garments fit chest, waist and length. Heavier girls do tend to need bigger sizes at the same height, but chest and waist circumference should carry more signal than weight. If a school allows it, I'd test those as features.
- **Weighing children is sensitive.** Use anonymised IDs, get school and parental consent, and check the data-protection rules that apply where you work.
- **Single-garment sizes only.** If shirt and skirt sizes differ, run the pipeline once per garment and treat a girl as AUTO only when both are AUTO.

## Next steps

- Try it on real data: even a small pilot (100-200 girls at one school) with height, weight and fitted size at the same visit, optionally chest and waist.
- Add realistic baselines (a height-band size chart, a vendor chart) and show whether the model actually beats them.
- Replace the interval rule with a direct AUTO accuracy target, calibrated on the new school's first fittings, and compare it with the conformal version.
- Change the "no size inside the range" case to FLAG. Right now it assigns the nearest size, which is the wrong call for a girl sitting between two sizes.

## Quick start

```bash
pip install -r requirements.txt
python scripts/generate_synthetic_data.py   # optional: regenerate the fake data
python run_pipeline.py                      # clean, evaluate (leave-one-school-out), train, save
python tests/test_core.py                   # sanity tests
python scripts/predict.py --input data/example_new_students.csv
```

Outputs go to `results/` (metric tables and plots) and `models/` (`size_model.joblib`, `final_config.json`).

## Using your own data

`data/students.csv`, one row per girl per measurement visit:

| column | example | notes |
|---|---|---|
| student_id | S0412 | anonymised, stable across years |
| school_id | A | needed for leave-one-school-out |
| grade | 3 | 1-5 |
| section | 3B | used to split without leakage |
| height_cm | 128.5 | measured at the same visit as the fitting |
| weight_kg | 26.0 | leave empty if missing, never 0 |
| fitted_size | 28 | the size that fit, before any alteration |
| year | 2025 | |
| age_years | 8.4 | optional; defaults to grade + 5.5 |

Then edit `data/size_chart.csv` (your real ordered sizes; the 22-34 labels are placeholders) and the settings in `src/config.py`: target coverage, minimum group sizes, valid height and weight ranges.

Rows with impossible values (height outside 90-170 cm, weight outside 12-70 kg), duplicates, unknown sizes or missing inputs are dropped and counted in `results/cleaning_report.json`. Sizes with fewer than 30 girls are merged into a neighbour.

## Using it at a school

```bash
python scripts/predict.py --input new_school.csv --output results/new_school_predictions.csv
```

Each row gets `decision`, `size`, `candidate_sizes` (for FLAG), `flag_reason`, and `audit_hand_fit`, which marks a random 10% of AUTO girls for a hand-fit check.

1. Hand-fit every FLAG girl and the audit sample, and log the final fitted size for everyone.
2. Run `python scripts/audit_check.py --input file.csv --target 0.85`. If audited accuracy falls below the target, go back to manual fitting until you recalibrate.
3. Once a year, run `python scripts/recalibrate.py --new new_year.csv` to update the calibration. Retrain with `run_pipeline.py` if the data has drifted.

## Repo layout

```
data/       students.csv (synthetic), size_chart.csv, example_new_students.csv
src/        config, load_clean, splits, models, conformal, evaluate, plots, predict
scripts/    generate_synthetic_data, predict, recalibrate, audit_check
tests/      test_core.py
results/    metric tables and plots (written by run_pipeline.py)
models/     saved model and config
run_pipeline.py
```

## Stack

Python, pandas, NumPy, scikit-learn, matplotlib.