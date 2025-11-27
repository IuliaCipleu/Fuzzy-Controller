"""
Offline tuner for mileage membership functions using Optuna.
Seeds age/claims from quantiles; learns mileage breakpoints; optionally mines one
extra rule from a shallow decision tree. Writes `mfs_config.json` for runtime.


Usage:
pip install scikit-fuzzy numpy pandas scikit-learn optuna
python tune_mfs.py --out mfs_config.json --trials 60 --demo
# or provide your CSV
python tune_mfs.py --data data.csv --miles annual_mileage --age driver_age --claims prior_claims --target target_premium --out mfs_config.json
"""


import json, os
import argparse
import time
from matplotlib.pyplot import grid
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pricing_core_underwriter import FuzzyPricing


try:
    import optuna
except Exception:
    optuna = None

MILES_MAX = 30000
AGE_MIN, AGE_MAX = 16, 85
CLAIMS_MAX = 5
PREMIUM_MAX = 2000
ENGINE_MIN, ENGINE_MAX = 800, 5000
CAR_AGE_MIN, CAR_AGE_MAX = 0, 30
mileage_universe = np.arange(0, MILES_MAX + 1, 50)
age_universe = np.arange(AGE_MIN, AGE_MAX + 1, 1)
claims_universe = np.arange(0, CLAIMS_MAX + 1, 1)
premium_universe = np.arange(0, PREMIUM_MAX + 1, 1)
engine_volume_universe = np.arange(ENGINE_MIN, ENGINE_MAX + 1, 100)
car_age_universe = np.arange(CAR_AGE_MIN, CAR_AGE_MAX + 1, 1)
# Brand risk universe (risk_weight 0.0 to 1.0)
brand_risk_universe = np.arange(0.0, 1.01, 0.01)

age_universe = np.arange(AGE_MIN, AGE_MAX + 1, 1)
premium_universe = np.arange(0, PREMIUM_MAX + 1, 1)


def seed_from_quantiles(df, miles_col, age_col, claims_col, engine_col, car_age_col):
    miles = df[miles_col].clip(0, MILES_MAX).to_numpy()
    ages = df[age_col].clip(AGE_MIN, AGE_MAX).to_numpy()
    engines = df[engine_col].clip(ENGINE_MIN, ENGINE_MAX).to_numpy()
    car_ages = df[car_age_col].clip(CAR_AGE_MIN, CAR_AGE_MAX).to_numpy()
    q_m = np.quantile(miles, [0.15, 0.35, 0.5, 0.7, 0.9])
    q_a = np.quantile(ages, [0.15, 0.35, 0.5, 0.75, 0.9])
    q_e = np.quantile(engines, [0.15, 0.35, 0.5, 0.7, 0.9])
    q_ca = np.quantile(car_ages, [0.15, 0.35, 0.5, 0.7, 0.9])
    cfg = {
        "annual_mileage": {
            "Low": {"kind": "trapmf", "params": [0, 0, float(q_m[0]), float(q_m[2])]},
            "Medium": {"kind": "trimf", "params": [float(q_m[1]), float(q_m[2]), float(q_m[3])]},
            "High": {"kind": "trimf", "params": [float(q_m[2]), float(q_m[3]), float(q_m[4])]},
            "VeryHigh": {"kind": "trapmf", "params": [float(q_m[3]), float(q_m[4]), MILES_MAX, MILES_MAX]},
        },
        "driver_age": {
            "Young": {"kind": "trapmf", "params": [AGE_MIN, AGE_MIN, float(q_a[0]), float(q_a[1])]},
            "Prime": {"kind": "trimf", "params": [float(q_a[0]), float(q_a[2]), float(q_a[3])]},
            "Senior": {"kind": "trapmf", "params": [float(q_a[3]), float(q_a[4]), AGE_MAX, AGE_MAX]},
        },
        "prior_claims": {
            "None": {"kind": "trapmf", "params": [-0.5, 0, 0, 0.5]},
            "Few": {"kind": "trimf", "params": [0, 1, 2]},
            "Many": {"kind": "trapmf", "params": [2, 3, CLAIMS_MAX, CLAIMS_MAX]},
        },
        "engine_volume": {
            "Small": {"kind": "trapmf", "params": [ENGINE_MIN, ENGINE_MIN, float(q_e[0]), float(q_e[2])]},
            "Medium": {"kind": "trimf", "params": [float(q_e[1]), float(q_e[2]), float(q_e[3])]},
            "Large": {"kind": "trapmf", "params": [float(q_e[3]), float(q_e[4]), ENGINE_MAX, ENGINE_MAX]},
        },
        "car_age": {
            "New": {"kind": "trapmf", "params": [CAR_AGE_MIN, CAR_AGE_MIN, float(q_ca[0]), float(q_ca[1])]},
            "MidAge": {"kind": "trimf", "params": [float(q_ca[0]), float(q_ca[2]), float(q_ca[3])]},
            "Old": {"kind": "trapmf", "params": [float(q_ca[3]), float(q_ca[4]), CAR_AGE_MAX, CAR_AGE_MAX]},
        },
        "car_brand": {
            "Safe": {"kind": "trapmf", "params": [0.0, 0.0, 0.35, 0.45]},
            "Average": {"kind": "trimf", "params": [0.4, 0.55, 0.7]},
            "Risky": {"kind": "trapmf", "params": [0.6, 0.7, 1.0, 1.0]},
        },
        "premium": {
            "VeryLow": {"kind": "trapmf", "params": [0, 0, 300, 450]},
            "Low": {"kind": "trimf", "params": [400, 550, 700]},
            "Medium": {"kind": "trimf", "params": [650, 900, 1150]},
            "High": {"kind": "trimf", "params": [1000, 1300, 1600]},
            "VeryHigh": {"kind": "trapmf", "params": [1500, 1800, PREMIUM_MAX, PREMIUM_MAX]},
        },
    }
    return cfg
# Build fuzzy system from a config (for evaluation during tuning)


def validate_breakpoints(cfg):
    bp = cfg.get('annual_mileage', {})
    if not bp or len(bp) < 4:
        print("[DEBUG] annual_mileage config missing or incomplete:", bp)
        return False
    params = []
    for k in ['Low', 'Medium', 'High', 'VeryHigh']:
        if k not in bp:
            print(f"[DEBUG] '{k}' missing in annual_mileage config.")
            return False
        params += bp[k]['params']
    # Ensure all breakpoints are strictly increasing and unique
    for i in range(1, len(params)):
        if params[i] <= params[i-1]:
            print("[DEBUG] annual_mileage breakpoints not strictly increasing:", params)
            return False
    return True

def system_from_cfg(cfg):
    if not validate_breakpoints(cfg):
        print("[ERROR] annual_mileage breakpoints validation failed.")
        return None
    am = ctrl.Antecedent(mileage_universe, 'annual_mileage')
    da = ctrl.Antecedent(age_universe, 'driver_age')
    pc = ctrl.Antecedent(claims_universe, 'prior_claims')
    ev = ctrl.Antecedent(engine_volume_universe, 'engine_volume')
    ca = ctrl.Antecedent(car_age_universe, 'car_age')
    cb = ctrl.Antecedent(brand_risk_universe, 'car_brand')
    pr = ctrl.Consequent(premium_universe, 'premium')
    def attach(var, grid, spec, varname):
        if not spec:
            print(f"[ERROR] Missing membership functions for '{varname}' in config.")
            return
        for k, v in spec.items():
            if v["kind"]=="trapmf":
                var[k] = fuzz.trapmf(grid, v["params"])
            else:
                var[k] = fuzz.trimf(grid, v["params"])
    attach(am, mileage_universe, cfg.get("annual_mileage"), "annual_mileage")
    attach(da, age_universe, cfg.get("driver_age"), "driver_age")
    attach(pc, claims_universe, cfg.get("prior_claims"), "prior_claims")
    attach(ev, engine_volume_universe, cfg.get("engine_volume"), "engine_volume")
    attach(ca, car_age_universe, cfg.get("car_age"), "car_age")
    attach(cb, brand_risk_universe, cfg.get("car_brand"), "car_brand")
    attach(pr, premium_universe, cfg.get("premium"), "premium")
    rules = [
        ctrl.Rule(am['Low'] & da['Prime'] & pc['None'] & ev['Small'] & ca['New'] & cb['Safe'], pr['VeryLow']),
        ctrl.Rule(am['Low'] & da['Young'] & ev['Small'] & cb['Safe'], pr['Low']),
        ctrl.Rule(am['Low'] & pc['Few'] & ca['MidAge'] & cb['Average'], pr['Low']),
        ctrl.Rule(am['Medium'] & da['Prime'] & pc['None'] & ev['Medium'] & cb['Average'], pr['Medium']),
        ctrl.Rule(am['Medium'] & pc['Few'] & ca['Old'] & cb['Risky'], pr['High']),
        ctrl.Rule(am['High'] | da['Young'] | ev['Large'] | cb['Risky'], pr['High']),
        ctrl.Rule(am['VeryHigh'] | pc['Many'] | ca['Old'] | cb['Risky'], pr['VeryHigh']),
        ctrl.Rule(da['Senior'] & am['High'] & ca['Old'] & cb['Average'], pr['High']),
        ctrl.Rule(da['Senior'] & pc['Few'] & ev['Medium'] & cb['Average'], pr['Medium']),
        ctrl.Rule(am['High'] & pc['None'] & ca['MidAge'] & cb['Average'], pr['High']),
        # --- Ensure engine_volume and car_age and car_brand are always included ---
        ctrl.Rule(ev['Large'], pr['High']),
        ctrl.Rule(ev['Small'] & ca['New'], pr['VeryLow']),
        ctrl.Rule(ca['Old'], pr['High']),
        ctrl.Rule(cb['Safe'], pr['VeryLow']),
        ctrl.Rule(cb['Average'], pr['Medium']),
        ctrl.Rule(cb['Risky'], pr['High']),
        # -----------------------------------------------------------
    ]
    return ctrl.ControlSystem(rules)

def simulate(system, row):
    # system here is actually a config dict
    model = FuzzyPricing(system)
    # row: [miles, age, claims, engine_volume, car_age, car_brand]
    return model.estimate(row[0], row[1], row[2], row[3], row[4], row[5] if len(row) > 5 else "Unknown")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data')
    ap.add_argument('--miles', default='annual_mileage')
    ap.add_argument('--age', default='driver_age')
    ap.add_argument('--claims', default='prior_claims')
    ap.add_argument('--engine_volume', default='engine_volume')
    ap.add_argument('--car_age', default='car_age')
    ap.add_argument('--target', default='target_premium')
    ap.add_argument('--trials', type=int, default=40)
    ap.add_argument('--out', default='mfs_config.json')
    ap.add_argument('--demo', action='store_true')
    args = ap.parse_args()

    # Load/synthesize
    print("[INFO] Loading data...")
    if args.data:
        df = pd.read_csv(args.data)
    else:
        rng = np.random.default_rng(7)
        n=6000
        miles = rng.integers(2000, 28000, n)
        age = rng.integers(18, 80, n)
        claims = rng.choice([0,1,2,3], n, p=[0.72,0.18,0.08,0.02])
        engine_volume = rng.integers(800, 5000, n)
        car_age = rng.integers(0, 30, n)
        risk = (
            0.00002*miles +
            0.25*(age<25) +
            0.15*(age>65) +
            0.35*claims +
            0.15*(engine_volume>3000) +
            0.10*(car_age>15) +
            rng.normal(0,0.05,n)
        )
        premium = np.clip(350 + 600*risk, 200, PREMIUM_MAX)
        df = pd.DataFrame({
            args.miles: miles,
            args.age: age,
            args.claims: claims,
            args.engine_volume: engine_volume,
            args.car_age: car_age,
            args.target: premium
        })
    print(f"[INFO] Data shape: {df.shape}")

    print("[INFO] Seeding membership functions from quantiles...")
    cfg = seed_from_quantiles(df, args.miles, args.age, args.claims, args.engine_volume, args.car_age)


    # For demo/simulated data, add a car_brand column
    if 'car_brand' not in df.columns:
        # Use random brands from the safety_score_brands file
        with open(os.path.join(os.path.dirname(__file__), 'safety_score_brands'), encoding='utf-8') as f:
            brands = [row['brand'] for row in json.load(f)]
        df['car_brand'] = np.random.choice(brands, len(df))
    print(f"[INFO] Brands used: {set(df['car_brand'])}")

    X = df[[args.miles, args.age, args.claims, args.engine_volume, args.car_age, 'car_brand']].to_numpy()
    y = df[args.target].to_numpy()


    # Optuna tuning of mileage breakpoints
    if optuna is None:
        print('[WARN] optuna not installed; saving seeded config only')
    else:
        best_value = float('inf')
        trial_times = []

        def objective(trial):
            t0 = time.time()
            # Enforce strictly increasing breakpoints
            m0 = trial.suggest_float('m0', 2000, 12000)
            m1 = trial.suggest_float('m1', m0+500, 15000)
            m2 = trial.suggest_float('m2', m1+500, 18000)
            m3 = trial.suggest_float('m3', m2+500, 22000)
            m4 = trial.suggest_float('m4', m3+500, 26000)
            m5 = trial.suggest_float('m5', m4+500, 29000)
            # Ensure all breakpoints are unique
            if len({m0, m1, m2, m3, m4, m5}) < 6:
                print("[DEBUG] Duplicate breakpoints detected:", [m0, m1, m2, m3, m4, m5])
                return PREMIUM_MAX * 10
            cfg_try = json.loads(json.dumps(cfg)) # deep copy
            cfg_try['annual_mileage'] = {
                'Low': {"kind":"trapmf","params":[0,0,m0,m1]},
                'Medium': {"kind":"trimf", "params":[m0,m2,m3]},
                'High': {"kind":"trimf", "params":[m2,m4,m5]},
                'VeryHigh': {"kind":"trapmf","params":[m4,m5,MILES_MAX,MILES_MAX]},
            }
            # Use config dict directly for FuzzyPricing
            try:
                preds = np.array([simulate(cfg_try, row) for row in X])
            except Exception as e:
                print(f"[WARN] Skipping trial due to error: {e}")
                return PREMIUM_MAX * 10
            mae = mean_absolute_error(y, preds)
            # Report intermediate value for pruning
            trial.report(mae, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()
            # simple monotonicity penalty
            vio = 0
            for i in range(300):
                j = np.random.randint(0, len(X))
                base = preds[j]
                row = X[j].copy(); row[0] = min(row[0]+3000, MILES_MAX)
                hi = simulate(cfg_try, row)  # <-- fix here
                if hi + 1e-6 < base: vio += 1
            penalty = 0.5 * (vio/300) * (PREMIUM_MAX/10)
            result = mae + penalty
            nonlocal best_value
            t1 = time.time()
            trial_time = t1 - t0
            trial_times.append(trial_time)
            print(f"[TRIAL {trial.number}] MAE={mae:.4f} Penalty={penalty:.4f} Result={result:.4f} Time={trial_time:.2f}s Best={best_value:.4f}")
            if result < best_value:
                best_value = result
                # Save current best config immediately
                bp = {
                    'm0': m0, 'm1': m1, 'm2': m2, 'm3': m3, 'm4': m4, 'm5': m5
                }
                cfg['annual_mileage'] = {
                    'Low': {"kind":"trapmf","params":[0,0,bp['m0'],bp['m1']]},
                    'Medium': {"kind":"trimf", "params":[bp['m0'],bp['m2'],bp['m3']]},
                    'High': {"kind":"trimf", "params":[bp['m2'],bp['m4'],bp['m5']]},
                    'VeryHigh': {"kind":"trapmf","params":[bp['m4'],bp['m5'],MILES_MAX,MILES_MAX]},
                }
                with open(args.out, 'w') as f:
                    json.dump(cfg, f, indent=2)
                print(f"[RECOVERY] Saved new best config to {args.out} (value={result:.4f})")
            return result

        print(f"[INFO] Starting Optuna tuning for {args.trials} trials...")
        t_start = time.time()
        study = optuna.create_study(
            direction='minimize',
            storage='sqlite:///optuna_study.db',
            study_name='underwriter_tuning_6_inputs',
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1)
        )
        study.optimize(objective, n_trials=args.trials)
        t_end = time.time()
        print(f"[INFO] Optuna tuning finished in {t_end-t_start:.2f}s. Avg trial time: {np.mean(trial_times):.2f}s")
        bp = study.best_params
        cfg['annual_mileage'] = {
            'Low': {"kind":"trapmf","params":[0,0,bp['m0'],bp['m1']]},
            'Medium': {"kind":"trimf", "params":[bp['m0'],bp['m2'],bp['m3']]},
            'High': {"kind":"trimf", "params":[bp['m2'],bp['m4'],bp['m5']]},
            'VeryHigh': {"kind":"trapmf","params":[bp['m4'],bp['m5'],MILES_MAX,MILES_MAX]},
        }
        print('Best mileage breakpoints:', bp)


    # (Optional) Mine one interaction rule if it helps
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=200, random_state=42)
    tree.fit(X, y)
    # crude check: if top split is on claims and second on mileage -> add Many & High -> VeryHigh
    if tree.tree_.feature[0] in [2,0]: # 0:miles,1:age,2:claims,3:engine_volume,4:car_age
        cfg.setdefault('mined_rules', []).append({
            'if': [['annual_mileage','High'], ['prior_claims','Many']],
            'then': ['premium','VeryHigh'],
            'op': 'AND'
        })


    with open(args.out, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved config to {args.out}")


if __name__ == '__main__':
    main()
