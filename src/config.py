"""Central configuration. Edit this file (and data/size_chart.csv) for your real project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "students.csv"
SIZE_CHART_PATH = ROOT / "data" / "size_chart.csv"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = ROOT / "models"

# ---- Conformal / decision settings -------------------------------------
ALPHA = 0.10                 # target miscoverage -> 90% coverage
ALPHA_SWEEP = [0.20, 0.10, 0.05]
MIN_GROUP_N = 30             # min calibration students per grade group (else merge)
MIN_SIZE_N = 30              # sizes with fewer students are merged into a neighbour
SIGMA_FLOOR = 0.15           # lower bound on predicted error (index units)

# ---- Splitting ---------------------------------------------------------
CAL_FRACTION = 0.30          # share of sections used for calibration
RANDOM_STATE = 42
N_OOF_FOLDS = 5              # folds for out-of-fold residuals (grouped by section)

# ---- Cleaning limits ---------------------------------------------------
HEIGHT_RANGE = (90.0, 170.0)   # cm
WEIGHT_RANGE = (12.0, 70.0)    # kg

# ---- Experiment grid ---------------------------------------------------
MODEL_NAMES = ["lookup", "linear", "gbm"]
FEATURE_SETS = {
    "height": ["height_cm"],
    "height+weight": ["height_cm", "weight_kg", "bmi"],
    "height+weight+age": ["height_cm", "weight_kg", "bmi", "age_years"],
}
# complexity order used when choosing the "simplest model within noise of the best"
MODEL_RANK = {"lookup": 0, "linear": 1, "gbm": 2}
FEATURE_RANK = {"height": 0, "height+weight": 1, "height+weight+age": 2}

REQUIRED_COLUMNS = ["student_id", "school_id", "grade", "section",
                    "height_cm", "weight_kg", "fitted_size", "year"]
