"""
Car Insurance – Fuzzy Premium Estimator (starter)

Inputs
  - annual_mileage (0..30_000 miles/year)
  - driver_age     (16..85 years)
  - prior_claims   (0..5 at-fault claims in last 3 years)

Output
  - premium (0..2000, arbitrary currency units)

What this does
  - lets a user enter the 3 inputs
  - computes a fuzzy premium with readable baseline rules
  - auto-plots membership activations for each input + the aggregated output MF

Install deps:
    pip install scikit-fuzzy matplotlib numpy
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
except ImportError:
    sys.stderr.write(
        "Missing dependency: scikit-fuzzy.\n"
        "Install with: pip install scikit-fuzzy\n"
    )
    raise

# -----------------------------
# 1) Universes (domains)
# -----------------------------
mileage_universe = np.arange(0, 30001, 50)   # miles/year
age_universe     = np.arange(16, 86, 1)      # years
claims_universe  = np.arange(0, 6, 1)        # integer count 0..5

premium_universe = np.arange(0, 2001, 1)     # currency units

# -----------------------------
# 2) Linguistic variables
# -----------------------------
annual_mileage = ctrl.Antecedent(mileage_universe, 'annual_mileage')
driver_age     = ctrl.Antecedent(age_universe, 'driver_age')
prior_claims   = ctrl.Antecedent(claims_universe, 'prior_claims')

premium = ctrl.Consequent(premium_universe, 'premium')

# -----------------------------
# 3) Membership functions
#    (Tweak these to your book/pricing philosophy)
# -----------------------------
# Annual mileage
annual_mileage['Low']      = fuzz.trapmf(mileage_universe, [0, 0, 8_000, 12_000])
annual_mileage['Medium']   = fuzz.trimf(mileage_universe, [10_000, 15_000, 20_000])
annual_mileage['High']     = fuzz.trimf(mileage_universe, [18_000, 23_000, 28_000])
annual_mileage['VeryHigh'] = fuzz.trapmf(mileage_universe, [26_000, 28_000, 30_000, 30_000])

# Driver age
driver_age['Young']  = fuzz.trapmf(age_universe, [16, 16, 22, 26])
driver_age['Prime']  = fuzz.trimf(age_universe, [25, 40, 60])
driver_age['Senior'] = fuzz.trapmf(age_universe, [58, 70, 85, 85])

# Prior at-fault claims (last 3 yrs)
prior_claims['None'] = fuzz.trapmf(claims_universe, [-0.5, 0, 0, 0.5])
prior_claims['Few']  = fuzz.trimf(claims_universe, [0, 1, 2])
prior_claims['Many'] = fuzz.trapmf(claims_universe, [2, 3, 5, 5])

# Premium (output)
premium['VeryLow']  = fuzz.trapmf(premium_universe, [0, 0, 300, 450])
premium['Low']      = fuzz.trimf(premium_universe, [400, 550, 700])
premium['Medium']   = fuzz.trimf(premium_universe, [650, 900, 1150])
premium['High']     = fuzz.trimf(premium_universe, [1000, 1300, 1600])
premium['VeryHigh'] = fuzz.trapmf(premium_universe, [1500, 1800, 2000, 2000])

# -----------------------------
# 4) Rules (baseline, readable)
# -----------------------------
rules = [
    # Best risk: prime age + low mileage + no claims
    ctrl.Rule(annual_mileage['Low'] & driver_age['Prime'] & prior_claims['None'], premium['VeryLow']),

    # Low mileage but youthful driver => still some risk
    ctrl.Rule(annual_mileage['Low'] & driver_age['Young'], premium['Low']),
    ctrl.Rule(annual_mileage['Low'] & prior_claims['Few'], premium['Low']),

    # Typical case
    ctrl.Rule(annual_mileage['Medium'] & driver_age['Prime'] & prior_claims['None'], premium['Medium']),
    ctrl.Rule(annual_mileage['Medium'] & prior_claims['Few'], premium['High']),

    # High risk levers
    ctrl.Rule(annual_mileage['High'] | driver_age['Young'], premium['High']),
    ctrl.Rule(annual_mileage['VeryHigh'] | prior_claims['Many'], premium['VeryHigh']),

    # Edge cases for seniors
    ctrl.Rule(driver_age['Senior'] & annual_mileage['High'], premium['High']),
    ctrl.Rule(driver_age['Senior'] & prior_claims['Few'], premium['Medium']),

    # If high miles but spotless history, keep it high
    ctrl.Rule(annual_mileage['High'] & prior_claims['None'], premium['High']),
]

system = ctrl.ControlSystem(rules)
sim    = ctrl.ControlSystemSimulation(system)

# -----------------------------
# 5) Plot helpers
# -----------------------------
def _plot_input_activation(var_name, antecedent, universe, x_val):
    """Plot all membership functions for an input and shade activation at x_val."""
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

def _plot_output(y_val, aggregated=None):
    """Plot the output membership functions and shade the aggregated result."""
    fig, ax = plt.subplots()
    ax.set_title("Output: premium – membership & defuzzified result")

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

# -----------------------------
# 6) Aggregation preview (for shading)
#    Builds an approximate aggregated MF using the same logical rules.
# -----------------------------
def _aggregate_output(mu_miles, mu_age, mu_claims):
    agg = np.zeros_like(premium_universe, dtype=float)

    def AND(*vals): return min(vals)
    def OR(*vals):  return max(vals)

    def apply(activation, cons_term):
        nonlocal agg
        agg = np.fmax(agg, np.fmin(activation, cons_term.mf))

    # Mirror the rules above (keep in sync)
    apply(AND(mu_miles['Low'], mu_age['Prime'], mu_claims['None']), premium['VeryLow'])
    apply(AND(mu_miles['Low'], mu_age['Young']),                    premium['Low'])
    apply(AND(mu_miles['Low'], mu_claims['Few']),                   premium['Low'])
    apply(AND(mu_miles['Medium'], mu_age['Prime'], mu_claims['None']), premium['Medium'])
    apply(AND(mu_miles['Medium'], mu_claims['Few']),                premium['High'])
    apply(OR(mu_miles['High'], mu_age['Young']),                    premium['High'])
    apply(OR(mu_miles['VeryHigh'], mu_claims['Many']),              premium['VeryHigh'])
    apply(AND(mu_age['Senior'], mu_miles['High']),                  premium['High'])
    apply(AND(mu_age['Senior'], mu_claims['Few']),                  premium['Medium'])
    apply(AND(mu_miles['High'], mu_claims['None']),                 premium['High'])

    return agg

# -----------------------------
# 7) Compute & visualize one run
# -----------------------------
def compute_and_plot(miles: float, age: float, claims: int, save_prefix: str = None):
    # Clamp to universes
    miles  = float(np.clip(miles, mileage_universe[0], mileage_universe[-1]))
    age    = float(np.clip(age,   age_universe[0],     age_universe[-1]))
    claims = int(np.clip(claims,  claims_universe[0],  claims_universe[-1]))

    # Run fuzzy inference
    sim.input['annual_mileage'] = miles
    sim.input['driver_age']     = age
    sim.input['prior_claims']   = float(claims)
    sim.compute()
    out = float(sim.output['premium'])

    # Input visualizations + membership degrees
    fig1, mu_miles  = _plot_input_activation('annual_mileage', annual_mileage, mileage_universe, miles)
    fig2, mu_age    = _plot_input_activation('driver_age',     driver_age,     age_universe,     age)
    fig3, mu_claims = _plot_input_activation('prior_claims',   prior_claims,   claims_universe,  claims)

    # Aggregated MF shading (approximate)
    aggregated = _aggregate_output(mu_miles, mu_age, mu_claims)
    fig4 = _plot_output(out, aggregated=aggregated)

    if save_prefix:
        import os
        output_dir = os.path.join(os.path.dirname(__file__), "resulted_plots")
        os.makedirs(output_dir, exist_ok=True)
        fig1.savefig(os.path.join(output_dir, f"{save_prefix}_input_mileage.png"), dpi=150, bbox_inches='tight')
        fig2.savefig(os.path.join(output_dir, f"{save_prefix}_input_age.png"),     dpi=150, bbox_inches='tight')
        fig3.savefig(os.path.join(output_dir, f"{save_prefix}_input_claims.png"),  dpi=150, bbox_inches='tight')
        fig4.savefig(os.path.join(output_dir, f"{save_prefix}_output.png"),        dpi=150, bbox_inches='tight')

    print("\n--- Result ---")
    print(f"Annual mileage: {miles:,.0f} mi/year")
    print(f"Driver age:     {age:.0f} years")
    print(f"Prior claims:   {claims} (last 3 years)")
    print(f"Estimated premium: {out:.2f}\n")

    plt.show()

# -----------------------------
# 8) CLI loop
# -----------------------------
def main():
    print("Car Insurance – Fuzzy Premium Estimator")
    print("Enter 'q' at any prompt to quit.\n")

    demo = input("Run a quick demo with (12,000 mi/yr, age 40, 0 claims)? [y/n]: ").strip().lower()
    if demo == 'y':
        compute_and_plot(12_000, 40, 0, save_prefix="demo")

    while True:
        s = input("\nannual_mileage [0..30000]: ").strip().lower()
        if s in ("q", "quit", "exit"):
            break
        try:
            miles = float(s)
        except ValueError:
            print("Please enter a number.")
            continue

        s = input("driver_age [16..85]: ").strip().lower()
        if s in ("q", "quit", "exit"):
            break
        try:
            age = float(s)
        except ValueError:
            print("Please enter a number.")
            continue

        s = input("prior_claims [0..5]: ").strip().lower()
        if s in ("q", "quit", "exit"):
            break
        try:
            claims = int(s)
        except ValueError:
            print("Please enter an integer.")
            continue

        compute_and_plot(miles, age, claims)

if __name__ == "__main__":
    main()
