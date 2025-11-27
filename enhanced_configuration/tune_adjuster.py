"""
Offline tuner for car claim triage fuzzy controller using Optuna.
Learns damage breakpoints and saves best config for recovery.

Usage:
pip install scikit-fuzzy numpy pandas scikit-learn optuna
python tune_adjuster.py --out adjuster_config.json --trials 60 --demo
"""

import json, os
import argparse
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.metrics import mean_absolute_error
import sys

try:
    import optuna
except Exception:
    optuna = None

DAMAGE_MAX = 50000
HOSP_DAYS_MAX = 60
VICTIMS_MAX = 5
COVERAGE_MAX = 100
FRAUD_MAX = 100
PRIORITY_MAX = 100
HANDLER_MAX = 3

damage_universe = np.arange(0, DAMAGE_MAX + 1, 500)
hosp_days_universe = np.arange(0, HOSP_DAYS_MAX + 1, 1)
victims_universe = np.arange(0, VICTIMS_MAX + 1, 1)
coverage_universe = np.arange(0, COVERAGE_MAX + 1, 1)
fraud_universe = np.arange(0, FRAUD_MAX + 1, 1)
priority_universe = np.arange(0, PRIORITY_MAX + 1, 1)
handler_universe = np.arange(0, HANDLER_MAX + 1, 1)

def seed_from_quantiles(df):
    dmg = df['damage'].clip(0, DAMAGE_MAX).to_numpy()
    q_d = np.quantile(dmg, [0.15, 0.35, 0.5, 0.7, 0.9])
    cfg = {
        "damage": {
            "Low": {"kind": "gaussmf", "params": [7000, 6000]},
            "Medium": {"kind": "trimf", "params": [10000, 25000, 40000]},
            "High": {"kind": "sigmf", "params": [35000, 0.0003]},
        },
        "hospitalization_days": {
            "None": {"kind": "trapmf", "params": [0, 0, 1, 2]},
            "Short": {"kind": "gaussmf", "params": [7, 5]},
            "Long": {"kind": "sigmf", "params": [20, 0.2]},
        },
        "victims": {
            "None": {"kind": "trapmf", "params": [0, 0, 0, 1]},
            "Few": {"kind": "gaussmf", "params": [2, 1]},
            "Many": {"kind": "sigmf", "params": [3, 1]},
        },
        "coverage": {
            "Low": {"kind": "sigmf", "params": [50, -0.1]},
            "High": {"kind": "sigmf", "params": [50, 0.1]},
        },
        "fraud": {
            "Low": {"kind": "gaussmf", "params": [20, 20]},
            "High": {"kind": "gaussmf", "params": [80, 20]},
        },
        "priority": {
            "Low": {"kind": "gaussmf", "params": [20, 15]},
            "Medium": {"kind": "trimf", "params": [30, 50, 70]},
            "High": {"kind": "sigmf", "params": [70, 0.1]},
            "Critical": {"kind": "sigmf", "params": [95, 0.2]},
        },
        "handler": {
            "Auto": {"kind": "gaussmf", "params": [0, 0.5]},
            "Junior": {"kind": "gaussmf", "params": [1, 0.5]},
            "Senior": {"kind": "gaussmf", "params": [2, 0.5]},
            "Specialist": {"kind": "gaussmf", "params": [3, 0.5]},
        }
    }
    return cfg

def system_from_cfg(cfg):
    damage = ctrl.Antecedent(damage_universe, 'damage')
    hosp_days = ctrl.Antecedent(hosp_days_universe, 'hospitalization_days')
    victims = ctrl.Antecedent(victims_universe, 'victims')
    coverage = ctrl.Antecedent(coverage_universe, 'coverage')
    fraud = ctrl.Antecedent(fraud_universe, 'fraud')
    priority = ctrl.Consequent(priority_universe, 'priority')
    handler = ctrl.Consequent(handler_universe, 'handler')
    def attach(var, grid, spec):
        for k, v in spec.items():
            if v["kind"]=="trapmf":
                var[k] = fuzz.trapmf(grid, v["params"])
            elif v["kind"]=="trimf":
                var[k] = fuzz.trimf(grid, v["params"])
            elif v["kind"]=="gaussmf":
                var[k] = fuzz.gaussmf(grid, *v["params"])
            elif v["kind"]=="sigmf":
                var[k] = fuzz.sigmf(grid, *v["params"])
    attach(damage, damage_universe, cfg["damage"])
    attach(hosp_days, hosp_days_universe, cfg["hospitalization_days"])
    attach(victims, victims_universe, cfg["victims"])
    attach(coverage, coverage_universe, cfg["coverage"])
    attach(fraud, fraud_universe, cfg["fraud"])
    attach(priority, priority_universe, cfg["priority"])
    attach(handler, handler_universe, cfg["handler"])
    rules = [
        ctrl.Rule(damage['High'] | victims['Many'] | hosp_days['Long'], priority['Critical']),
        ctrl.Rule(damage['High'] | victims['Many'] | hosp_days['Long'], handler['Specialist']),
        ctrl.Rule(damage['Medium'] & (hosp_days['Short'] | victims['Few']), priority['High']),
        ctrl.Rule(damage['Medium'] & (hosp_days['Short'] | victims['Few']), handler['Senior']),
        ctrl.Rule(~fraud['High'] & coverage['High'], handler['Junior']),
        ctrl.Rule((damage['Low'] | victims['None']) & fraud['Low'], priority['Low']),
        ctrl.Rule((damage['Low'] | victims['None']) & fraud['Low'], handler['Auto']),
        ctrl.Rule((damage['Medium'] & fraud['High']) | (hosp_days['Long'] & fraud['High']), priority['Critical']),
        ctrl.Rule((damage['Medium'] & fraud['High']) | (hosp_days['Long'] & fraud['High']), handler['Specialist']),
        ctrl.Rule(coverage['High'] & damage['Medium'], priority['Medium']),
        ctrl.Rule(coverage['High'] & damage['Medium'], handler['Junior']),
        ctrl.Rule(coverage['Low'] & damage['High'], handler['Senior']),
        ctrl.Rule(victims['Few'] & fraud['Low'], priority['Medium']),
        ctrl.Rule(victims['Few'] & fraud['Low'], handler['Junior']),
        ctrl.Rule(damage['Low'] | damage['Medium'] | damage['High'], priority['Medium']),
        ctrl.Rule(damage['Low'] | damage['Medium'] | damage['High'], handler['Junior']),
    ]
    return ctrl.ControlSystem(rules)

def simulate(system, row):
    sim = ctrl.ControlSystemSimulation(system)
    sim.input['damage'] = row[0]
    sim.input['hospitalization_days'] = row[1]
    sim.input['victims'] = row[2]
    sim.input['coverage'] = row[3]
    sim.input['fraud'] = row[4]
    sim.compute()
    priority = sim.output.get('priority', None)
    handler = sim.output.get('handler', None)
    if priority is None or handler is None:
        return PRIORITY_MAX * 10, HANDLER_MAX * 10
    return priority, handler

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data')
    ap.add_argument('--trials', type=int, default=200)  # Increased default trials
    ap.add_argument('--out', default='adjuster_config.json')
    ap.add_argument('--demo', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    if optuna is not None:
        optuna.logging.set_verbosity(optuna.logging.INFO)
        sampler = optuna.samplers.TPESampler(seed=args.seed)
    else:
        sampler = None

    # Load/synthesize
    if args.data:
        df = pd.read_csv(args.data)
    else:
        rng = np.random.default_rng(args.seed)
        n = 2000
        damage = rng.integers(0, DAMAGE_MAX, n)
        hosp_days = rng.integers(0, HOSP_DAYS_MAX, n)
        victims = rng.integers(0, VICTIMS_MAX, n)
        coverage = rng.integers(0, COVERAGE_MAX, n)
        fraud = rng.integers(0, FRAUD_MAX, n)
        # Synthetic targets: higher damage, victims, hosp_days, fraud → higher priority/handler
        priority = np.clip(
            0.4*damage/DAMAGE_MAX*100 + 0.2*victims/VICTIMS_MAX*100 + 0.2*hosp_days/HOSP_DAYS_MAX*100 +
            0.2*fraud + 0.2*coverage - 0.3*coverage, 0, PRIORITY_MAX)
        handler = np.clip(
            0.02*damage/DAMAGE_MAX*HANDLER_MAX + 0.03*fraud/HANDLER_MAX + 0.02*hosp_days/HOSP_DAYS_MAX*HANDLER_MAX +
            0.02*victims/VICTIMS_MAX*HANDLER_MAX, 0, HANDLER_MAX)
        df = pd.DataFrame({
            'damage': damage,
            'hospitalization_days': hosp_days,
            'victims': victims,
            'coverage': coverage,
            'fraud': fraud,
            'priority': priority,
            'handler': handler
        })

    cfg = seed_from_quantiles(df)
    X = df[['damage', 'hospitalization_days', 'victims', 'coverage', 'fraud']].to_numpy()
    y_priority = df['priority'].to_numpy()
    y_handler = df['handler'].to_numpy()

    if optuna is None:
        print('[WARN] optuna not installed; saving seeded config only')
    else:
        best_value = float('inf')
        best_cfg = None

        def objective(trial):
            # Widened search ranges for damage breakpoints
            g_mean = trial.suggest_float('g_mean', 1000, 20000)
            g_sigma = trial.suggest_float('g_sigma', 1000, 20000)
            t_left = trial.suggest_float('t_left', 5000, 40000)
            t_mid = trial.suggest_float('t_mid', t_left, 50000)
            s_center = trial.suggest_float('s_center', 20000, 50000)
            s_slope = trial.suggest_float('s_slope', 0.00005, 0.002)

            # Ensure valid trimf parameters
            if not (t_left <= t_mid <= 40000):
                print(f"[SKIP] Invalid trimf params: t_left={t_left}, t_mid={t_mid}")
                return PRIORITY_MAX * 10

            cfg_try = json.loads(json.dumps(cfg)) # deep copy
            cfg_try['damage'] = {
                'Low': {"kind":"gaussmf","params":[g_mean, g_sigma]},
                'Medium': {"kind":"trimf", "params":[t_left, t_mid, 40000]},
                'High': {"kind":"sigmf", "params":[s_center, s_slope]},
            }
            system = system_from_cfg(cfg_try)
            try:
                preds = np.array([simulate(system, row) for row in X])
            except Exception as e:
                print(f"[WARN] Skipping trial due to error: {e}")
                return PRIORITY_MAX * 10
            preds_priority = preds[:,0]
            preds_handler = preds[:,1]
            mae_priority = mean_absolute_error(y_priority, preds_priority)
            mae_handler = mean_absolute_error(y_handler, preds_handler)
            result = mae_priority + mae_handler
            print(f"[LOG] Trial {trial.number}: MAE priority={mae_priority:.4f}, MAE handler={mae_handler:.4f}, SUM={result:.4f}")
            trial.report(result, step=0)
            if trial.should_prune():
                print(f"[PRUNE] Trial pruned (value={result:.4f})")
                raise optuna.TrialPruned()
            nonlocal best_value, best_cfg
            if result < best_value:
                best_value = result
                bp = {
                    'g_mean': g_mean, 'g_sigma': g_sigma,
                    't_left': t_left, 't_mid': t_mid,
                    's_center': s_center, 's_slope': s_slope
                }
                cfg['damage'] = {
                    'Low': {"kind":"gaussmf","params":[bp['g_mean'], bp['g_sigma']]},
                    'Medium': {"kind":"trimf", "params":[bp['t_left'], bp['t_mid'], 40000]},
                    'High': {"kind":"sigmf", "params":[bp['s_center'], bp['s_slope']]},
                }
                with open(args.out, 'w') as f:
                    json.dump(cfg, f, indent=2)
                print(f"[RECOVERY] Saved new best config to {args.out} (value={result:.4f})")
                best_cfg = json.loads(json.dumps(cfg))
            return result

        study = optuna.create_study(
            direction='minimize',
            storage='sqlite:///optuna_adjuster.db',
            study_name='adjuster_tuning_5_inputs',
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1),
            sampler=sampler
        )
        study.optimize(objective, n_trials=args.trials)
        bp = study.best_params
        cfg['damage'] = {
            'Low': {"kind":"gaussmf","params":[bp['g_mean'], bp['g_sigma']]},
            'Medium': {"kind":"trimf", "params":[bp['t_left'], bp['t_mid'], 40000]},
            'High': {"kind":"sigmf", "params":[bp['s_center'], bp['s_slope']]},
        }
        print('Best damage breakpoints:', bp)

    with open(args.out, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved config to {args.out}")

if __name__ == '__main__':
    main()
