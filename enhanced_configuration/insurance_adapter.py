import json
import argparse
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyInsuranceAdapter:
    def __init__(self, cfg: dict):
        self.inputs = {}
        self.universes = {}
        # Build input variables dynamically
        for var, spec in cfg["inputs"].items():
            universe = np.arange(spec["min"], spec["max"] + spec.get("step", 1), spec.get("step", 1))
            self.universes[var] = universe
            self.inputs[var] = ctrl.Antecedent(universe, var)
            for label, mf in spec["mfs"].items():
                self.inputs[var][label] = getattr(fuzz, mf["kind"])(universe, mf["params"])
        # Build output variable(s)
        self.outputs = {}
        for var, spec in cfg["outputs"].items():
            universe = np.arange(spec["min"], spec["max"] + spec.get("step", 1), spec.get("step", 1))
            self.universes[var] = universe
            self.outputs[var] = ctrl.Consequent(universe, var)
            for label, mf in spec["mfs"].items():
                self.outputs[var][label] = getattr(fuzz, mf["kind"])(universe, mf["params"])
        # Build rules
        rules = []
        for rule in cfg["rules"]:
            antecedent = self.inputs[rule["if"][0][0]][rule["if"][0][1]]
            for var, term in rule["if"][1:]:
                antecedent = antecedent & self.inputs[var][term]
            cons_var, cons_term = rule["then"]
            rules.append(ctrl.Rule(antecedent, self.outputs[cons_var][cons_term]))
        self.system = ctrl.ControlSystem(rules)
        self.sim = ctrl.ControlSystemSimulation(self.system)

    def estimate(self, **kwargs):
        for var in self.inputs:
            self.sim.input[var] = kwargs[var]
        self.sim.compute()
        # Return all outputs
        return {var: self.sim.output[var] for var in self.outputs}

def estimate_insurance(config_path: str, **kwargs):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    model = FuzzyInsuranceAdapter(cfg)
    return model.estimate(**kwargs)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--type', required=True, choices=['car', 'health', 'house', 'life'])
    ap.add_argument('--config', required=True)
    # Dynamically add input arguments based on type/config (for demo, just use car)
    ap.add_argument('--annual_mileage', type=float)
    ap.add_argument('--driver_age', type=float)
    ap.add_argument('--prior_claims', type=int)
    args = ap.parse_args()
    # Build kwargs for estimate
    inputs = {k: v for k, v in vars(args).items() if k not in ['type', 'config'] and v is not None}
    result = estimate_insurance(args.config, **inputs)
    print("Estimated outputs:", result)
    