

"""
Runtime fuzzy estimator that loads membership functions (MFs) from a JSON
produced by `tune_mfs.py`. Exposes a CLI and a callable `estimate_premium`.


Usage:
python pricing_core.py --config mfs_config.json --miles 12000 --age 40 --claims 0
"""
import json
import argparse
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import os
import json as _json


# Load brand risk weights from safety_score_brands
_BRAND_RISK_PATH = os.path.join(os.path.dirname(__file__), 'safety_score_brands')
with open(_BRAND_RISK_PATH, encoding='utf-8') as _f:
    _BRAND_RISK = {row['brand'].lower(): row['risk_weight'] for row in _json.load(_f)}


MILES_MAX = 30000
AGE_MIN, AGE_MAX = 16, 85
CLAIMS_MAX = 5
PREMIUM_MAX = 2000


mileage_universe = np.arange(0, MILES_MAX + 1, 50)
age_universe = np.arange(AGE_MIN, AGE_MAX + 1, 1)
claims_universe = np.arange(0, CLAIMS_MAX + 1, 1)
premium_universe = np.arange(0, PREMIUM_MAX + 1, 1)

# Add universes
# Add universes
engine_volume_universe = np.arange(800, 5001, 100)  # 800–5000 cc
car_age_universe = np.arange(0, 31, 1)              # 0–30 years

# Brand risk universe (risk_weight 0.0 to 1.0)
brand_risk_universe = np.arange(0.0, 1.01, 0.01)


def _mf_from_spec(kind: str, grid, params):
    if kind == "trapmf":
        return fuzz.trapmf(grid, params)
    if kind == "trimf":
        return fuzz.trimf(grid, params)
    raise ValueError(f"Unsupported MF kind: {kind}")

class FuzzyPricing:
    def __init__(self, cfg: dict):
        # Build variables
        self.annual_mileage = ctrl.Antecedent(mileage_universe, 'annual_mileage')
        self.driver_age = ctrl.Antecedent(age_universe, 'driver_age')
        self.prior_claims = ctrl.Antecedent(claims_universe, 'prior_claims')
        self.premium = ctrl.Consequent(premium_universe, 'premium')

        # Additions for engine volume, car age, and car brand risk
        self.engine_volume = ctrl.Antecedent(engine_volume_universe, 'engine_volume')
        self.car_age = ctrl.Antecedent(car_age_universe, 'car_age')
        self.car_brand = ctrl.Antecedent(brand_risk_universe, 'car_brand')
        # Membership functions for car_brand (risk_weight) from config if present, else default
        if "car_brand" in cfg:
            for label, spec in cfg["car_brand"].items():
                if spec["kind"] == "trapmf":
                    self.car_brand[label] = fuzz.trapmf(brand_risk_universe, spec["params"])
                else:
                    self.car_brand[label] = fuzz.trimf(brand_risk_universe, spec["params"])
        else:
            self.car_brand['Safe'] = fuzz.trapmf(brand_risk_universe, [0.0, 0.0, 0.35, 0.45])
            self.car_brand['Average'] = fuzz.trimf(brand_risk_universe, [0.4, 0.55, 0.7])
            self.car_brand['Risky'] = fuzz.trapmf(brand_risk_universe, [0.6, 0.7, 1.0, 1.0])


        # Attach MFs from config
        for label, spec in cfg["annual_mileage"].items():
            self.annual_mileage[label] = _mf_from_spec(spec["kind"], mileage_universe, spec["params"])
        for label, spec in cfg["driver_age"].items():
            self.driver_age[label] = _mf_from_spec(spec["kind"], age_universe, spec["params"])
        for label, spec in cfg["prior_claims"].items():
            self.prior_claims[label] = _mf_from_spec(spec["kind"], claims_universe, spec["params"])
        for label, spec in cfg["premium"].items():
            self.premium[label] = _mf_from_spec(spec["kind"], premium_universe, spec["params"])

        # Additions for engine volume and car age
        for label, spec in cfg["engine_volume"].items():
            if spec["kind"] == "trapmf":
                self.engine_volume[label] = fuzz.trapmf(engine_volume_universe, spec["params"])
            else:
                self.engine_volume[label] = fuzz.trimf(engine_volume_universe, spec["params"])
        for label, spec in cfg["car_age"].items():
            if spec["kind"] == "trapmf":
                self.car_age[label] = fuzz.trapmf(car_age_universe, spec["params"])
            else:
                self.car_age[label] = fuzz.trimf(car_age_universe, spec["params"])


        # Rules: baseline + optional mined_rules from config (if present)
        rules = [
            ctrl.Rule(self.annual_mileage['Low'] & self.driver_age['Prime'] & self.prior_claims['None'], self.premium['VeryLow']),
            ctrl.Rule(self.annual_mileage['Low'] & self.driver_age['Young'], self.premium['Low']),
            ctrl.Rule(self.annual_mileage['Low'] & self.prior_claims['Few'], self.premium['Low']),
            ctrl.Rule(self.annual_mileage['Medium'] & self.driver_age['Prime'] & self.prior_claims['None'], self.premium['Medium']),
            ctrl.Rule(self.annual_mileage['Medium'] & self.prior_claims['Few'], self.premium['High']),
            ctrl.Rule(self.annual_mileage['High'] | self.driver_age['Young'], self.premium['High']),
            ctrl.Rule(self.annual_mileage['VeryHigh'] | self.prior_claims['Many'], self.premium['VeryHigh']),
            ctrl.Rule(self.driver_age['Senior'] & self.annual_mileage['High'], self.premium['High']),
            ctrl.Rule(self.driver_age['Senior'] & self.prior_claims['Few'], self.premium['Medium']),
            ctrl.Rule(self.annual_mileage['High'] & self.prior_claims['None'], self.premium['High']),
        ]

        # Add catch-all rules for edge cases
        rules.append(ctrl.Rule(self.annual_mileage['Low'] & self.driver_age['Senior'], self.premium['Medium']))
        rules.append(ctrl.Rule(self.annual_mileage['VeryHigh'] & self.driver_age['Senior'], self.premium['VeryHigh']))
        rules.append(ctrl.Rule(self.annual_mileage['Low'] & self.prior_claims['Many'], self.premium['High']))
        rules.append(ctrl.Rule(self.annual_mileage['VeryHigh'] & self.prior_claims['None'], self.premium['High']))
        rules.append(ctrl.Rule(self.driver_age['Young'] & self.prior_claims['Many'], self.premium['High']))


        # Add rules that use engine_volume, car_age, and car_brand
        rules.append(ctrl.Rule(self.engine_volume['Large'], self.premium['High']))
        rules.append(ctrl.Rule(self.engine_volume['Small'] & self.car_age['New'], self.premium['VeryLow']))
        rules.append(ctrl.Rule(self.car_age['Old'], self.premium['High']))
        # Brand risk rules
        rules.append(ctrl.Rule(self.car_brand['Safe'], self.premium['VeryLow']))
        rules.append(ctrl.Rule(self.car_brand['Average'], self.premium['Medium']))
        rules.append(ctrl.Rule(self.car_brand['Risky'], self.premium['High']))

        # Optional: Default rule (fires for any input, lowest priority)
        rules.append(ctrl.Rule(
            self.annual_mileage['Low'] | self.annual_mileage['Medium'] | self.annual_mileage['High'] | self.annual_mileage['VeryHigh'],
            self.premium['Medium']
        ))

        for r in cfg.get("mined_rules", []):
            # r example: {"if": [["annual_mileage","High"],["prior_claims","Many"]], "then": ["premium","VeryHigh"], "op": "AND"}
            antecedents = []
            for var, term in r["if"]:
                antecedents.append(getattr(self, var)[term])
            antecedent = antecedents[0]
            for nxt in antecedents[1:]:
                antecedent = (antecedent & nxt) if r.get("op","AND")=="AND" else (antecedent | nxt)
            cons_var, cons_term = r["then"]
            rules.append(ctrl.Rule(antecedent, getattr(self, cons_var)[cons_term]))


        self.system = ctrl.ControlSystem(rules)


    def estimate(self, miles: float, age: float, claims: int, engine_volume: float, car_age: float, car_brand: str) -> float:
        miles = float(max(0, min(MILES_MAX, miles)))
        age = float(max(AGE_MIN, min(AGE_MAX, age)))
        claims = int(max(0, min(CLAIMS_MAX, int(claims))))
        engine_volume = float(max(800, min(5000, engine_volume)))
        car_age = float(max(0, min(30, car_age)))
        # Map brand to risk_weight
        brand_key = str(car_brand).lower().strip()
        risk_weight = _BRAND_RISK.get(brand_key, 0.5)  # Default to 0.5 if unknown
        sim = ctrl.ControlSystemSimulation(self.system)
        sim.input['annual_mileage'] = miles
        sim.input['driver_age'] = age
        sim.input['prior_claims'] = float(claims)
        sim.input['engine_volume'] = engine_volume
        sim.input['car_age'] = car_age
        sim.input['car_brand'] = risk_weight
        sim.compute()
        if 'premium' not in sim.output:
            # Return a penalty value or np.nan so tuning can handle it
            return -1
        return float(sim.output['premium'])


def estimate_premium(config_path: str, miles: float, age: float, claims: int, engine_volume: float, car_age: float, car_brand: str) -> float:
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    model = FuzzyPricing(cfg)
    return model.estimate(miles, age, claims, engine_volume, car_age, car_brand)


def _plot_input_activation(var_name, antecedent, universe, x_val):
    fig, ax = plt.subplots()
    ax.set_title(f"Input: {var_name} – membership & activation")
    mu_map = {}
    for label, term in antecedent.terms.items():
        ax.plot(universe, term.mf, label=label)
        mu = float(fuzz.interp_membership(universe, term.mf, x_val))
        mu_map[label] = mu
        ax.fill_between(universe, 0, np.minimum(term.mf, mu), alpha=0.3)
    ax.axvline(x_val, linestyle='--')
    ax.set_xlabel(var_name.replace('_', ' '))
    ax.set_ylabel('membership')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right')
    plt.tight_layout()
    return fig, mu_map

def _plot_output(y_val, premium, premium_universe, aggregated=None):
    fig, ax = plt.subplots()
    ax.set_title("Output: premium - membership & defuzzified result")
    for label, term in premium.terms.items():
        ax.plot(premium_universe, term.mf, label=label)
    if aggregated is not None:
        ax.fill_between(premium_universe, 0, aggregated, alpha=0.3)
    ax.axvline(y_val, linestyle='--')
    ax.set_xlabel('premium')
    ax.set_ylabel('membership')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right')
    plt.tight_layout()
    return fig


# After estimating premium in __main__:
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--miles', type=float, required=True)
    ap.add_argument('--age', type=float, required=True)
    ap.add_argument('--claims', type=int, required=True)
    ap.add_argument('--engine_volume', type=float, required=True)
    ap.add_argument('--car_age', type=float, required=True)
    ap.add_argument('--car_brand', type=str, required=True)
    args = ap.parse_args()
    p = estimate_premium(args.config, args.miles, args.age, args.claims, args.engine_volume, args.car_age, args.car_brand)
    print(f"Estimated premium: {p:.2f}")

    # Load config and fuzzy system for plotting
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    model = FuzzyPricing(cfg)

    # Plot input activations
    fig1, mu_miles  = _plot_input_activation('annual_mileage', model.annual_mileage, mileage_universe, args.miles)
    fig2, mu_age    = _plot_input_activation('driver_age',     model.driver_age,     age_universe,     args.age)
    fig3, mu_claims = _plot_input_activation('prior_claims',   model.prior_claims,   claims_universe,  args.claims)
    fig4, mu_engine = _plot_input_activation('engine_volume',  model.engine_volume,  engine_volume_universe, args.engine_volume)
    fig5, mu_cage   = _plot_input_activation('car_age',        model.car_age,        car_age_universe, args.car_age)
    fig6, mu_brand  = _plot_input_activation('car_brand',      model.car_brand,      brand_risk_universe, _BRAND_RISK.get(str(args.car_brand).lower().strip(), 0.5))

    # Plot output
    fig7 = _plot_output(p, model.premium, premium_universe)

    plt.show()
