import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Universes for car-specific claim inputs
damage_universe = np.arange(0, 50001, 500)            # 0–50,000 currency units
hosp_days_universe = np.arange(0, 61, 1)              # 0–60 days
victims_universe = np.arange(0, 6, 1)                 # 0–5 victims
coverage_universe = np.arange(0, 101, 1)
fraud_universe = np.arange(0, 101, 1)
priority_universe = np.arange(0, 101, 1)
handler_universe = np.arange(0, 4, 1)

# Antecedents
damage = ctrl.Antecedent(damage_universe, 'damage')
hosp_days = ctrl.Antecedent(hosp_days_universe, 'hospitalization_days')
victims = ctrl.Antecedent(victims_universe, 'victims')
coverage = ctrl.Antecedent(coverage_universe, 'coverage')
fraud = ctrl.Antecedent(fraud_universe, 'fraud')

# Consequents
priority = ctrl.Consequent(priority_universe, 'priority')
handler = ctrl.Consequent(handler_universe, 'handler')

# Membership functions (play with more types and overlap)
damage['Low'] = fuzz.gaussmf(damage_universe, 7000, 6000)
damage['Medium'] = fuzz.trimf(damage_universe, [10000, 25000, 40000])
damage['High'] = fuzz.sigmf(damage_universe, 35000, 0.0003)

hosp_days['None'] = fuzz.trapmf(hosp_days_universe, [0, 0, 1, 2])
hosp_days['Short'] = fuzz.gaussmf(hosp_days_universe, 7, 5)
hosp_days['Long'] = fuzz.sigmf(hosp_days_universe, 20, 0.2)

victims['None'] = fuzz.trapmf(victims_universe, [0, 0, 0, 1])
victims['Few'] = fuzz.gaussmf(victims_universe, 2, 1)
victims['Many'] = fuzz.sigmf(victims_universe, 3, 1)

coverage['Low'] = fuzz.sigmf(coverage_universe, 50, -0.1)
coverage['High'] = fuzz.sigmf(coverage_universe, 50, 0.1)

fraud['Low'] = fuzz.gaussmf(fraud_universe, 20, 20)
fraud['High'] = fuzz.gaussmf(fraud_universe, 80, 20)

priority['Low'] = fuzz.gaussmf(priority_universe, 20, 15)
priority['Medium'] = fuzz.trimf(priority_universe, [30, 50, 70])
priority['High'] = fuzz.sigmf(priority_universe, 70, 0.1)
priority['Critical'] = fuzz.sigmf(priority_universe, 95, 0.2)

handler['Auto'] = fuzz.gaussmf(handler_universe, 0, 0.5)
handler['Junior'] = fuzz.gaussmf(handler_universe, 1, 0.5)
handler['Senior'] = fuzz.gaussmf(handler_universe, 2, 0.5)
handler['Specialist'] = fuzz.gaussmf(handler_universe, 3, 0.5)

# More complex rules for increased coverage
rules = [
    # OR: High damage OR many victims OR long hospitalization triggers Critical
    ctrl.Rule(damage['High'] | victims['Many'] | hosp_days['Long'], priority['Critical']),
    ctrl.Rule(damage['High'] | victims['Many'] | hosp_days['Long'], handler['Specialist']),

    # AND/OR: Medium damage AND (short hospitalization OR few victims) triggers High
    ctrl.Rule(damage['Medium'] & (hosp_days['Short'] | victims['Few']), priority['High']),
    ctrl.Rule(damage['Medium'] & (hosp_days['Short'] | victims['Few']), handler['Senior']),

    # NOT: If NOT high fraud AND high coverage, handler is Junior
    ctrl.Rule(~fraud['High'] & coverage['High'], handler['Junior']),

    # Complex: (Low damage OR none victims) AND low fraud triggers Low
    ctrl.Rule((damage['Low'] | victims['None']) & fraud['Low'], priority['Low']),
    ctrl.Rule((damage['Low'] | victims['None']) & fraud['Low'], handler['Auto']),

    # Mixed: (Medium damage AND high fraud) OR (long hospitalization AND high fraud) triggers Critical
    ctrl.Rule((damage['Medium'] & fraud['High']) | (hosp_days['Long'] & fraud['High']), priority['Critical']),
    ctrl.Rule((damage['Medium'] & fraud['High']) | (hosp_days['Long'] & fraud['High']), handler['Specialist']),

    # Coverage and urgency (simulate urgency with coverage for demo)
    ctrl.Rule(coverage['High'] & damage['Medium'], priority['Medium']),
    ctrl.Rule(coverage['High'] & damage['Medium'], handler['Junior']),

    # If coverage is low and damage is high, handler is Senior
    ctrl.Rule(coverage['Low'] & damage['High'], handler['Senior']),

    # If victims are few and fraud is low, priority is Medium
    ctrl.Rule(victims['Few'] & fraud['Low'], priority['Medium']),
    ctrl.Rule(victims['Few'] & fraud['Low'], handler['Junior']),

    # Catch-all: Any damage triggers at least Medium priority and Junior handler
    ctrl.Rule(damage['Low'] | damage['Medium'] | damage['High'], priority['Medium']),
    ctrl.Rule(damage['Low'] | damage['Medium'] | damage['High'], handler['Junior']),
]

system = ctrl.ControlSystem(rules)
sim = ctrl.ControlSystemSimulation(system)

def estimate_triage(damage_val, hosp_days_val, victims_val, coverage_val, fraud_val):
    sim.input['damage'] = damage_val
    sim.input['hospitalization_days'] = hosp_days_val
    sim.input['victims'] = victims_val
    sim.input['coverage'] = coverage_val
    sim.input['fraud'] = fraud_val
    sim.compute()
    return sim.output['priority'], sim.output['handler']

def get_priority_class(score):
    if score >= 90:
        return "Critical"
    elif score >= 60:
        return "High"
    elif score >= 30:
        return "Medium"
    else:
        return "Low"

def get_handler_class(score):
    if score >= 2.5:
        return "Specialist"
    elif score >= 1.5:
        return "Senior"
    elif score >= 0.5:
        return "Junior"
    else:
        return "Auto"

if __name__ == "__main__":
    damage_val = 35000
    hosp_days_val = 12
    victims_val = 2
    coverage_val = 80
    fraud_val = 20

    priority_score, handler_type = estimate_triage(
        damage_val=damage_val,
        hosp_days_val=hosp_days_val,
        victims_val=victims_val,
        coverage_val=coverage_val,
        fraud_val=fraud_val
    )
    priority_class = get_priority_class(priority_score)
    handler_class = get_handler_class(handler_type)
    print(f"Triage Priority: {priority_score:.2f} ({priority_class}), Handler: {handler_type:.2f} ({handler_class})")

    # Plot input activations
    def plot_input_activation(var_name, antecedent, universe, x_val):
        fig, ax = plt.subplots()
        ax.set_title(f"Input: {var_name} – membership & activation")
        for label, term in antecedent.terms.items():
            ax.plot(universe, term.mf, label=label)
            mu = float(fuzz.interp_membership(universe, term.mf, x_val))
            ax.fill_between(universe, 0, np.minimum(term.mf, mu), alpha=0.3)
        ax.axvline(x_val, linestyle='--', color='black')
        ax.set_xlabel(var_name.replace('_', ' '))
        ax.set_ylabel('membership')
        ax.set_ylim(0, 1.05)
        ax.legend(loc='upper right')
        plt.tight_layout()
        return fig

    def plot_output(y_val, consequent, universe):
        fig, ax = plt.subplots()
        ax.set_title(f"Output: {consequent.label} – membership & defuzzified result")
        for label, term in consequent.terms.items():
            ax.plot(universe, term.mf, label=label)
        ax.axvline(y_val, linestyle='--', color='black')
        ax.set_xlabel(consequent.label)
        ax.set_ylabel('membership')
        ax.set_ylim(0, 1.05)
        ax.legend(loc='upper right')
        plt.tight_layout()
        return fig

    plot_input_activation('damage', damage, damage_universe, damage_val)
    plot_input_activation('hospitalization_days', hosp_days, hosp_days_universe, hosp_days_val)
    plot_input_activation('victims', victims, victims_universe, victims_val)
    plot_input_activation('coverage', coverage, coverage_universe, coverage_val)
    plot_input_activation('fraud', fraud, fraud_universe, fraud_val)

    plot_output(priority_score, priority, priority_universe)
    plot_output(handler_type, handler, handler_universe)

    plt.show()