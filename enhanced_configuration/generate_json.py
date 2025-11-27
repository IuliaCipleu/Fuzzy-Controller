import os
import json

os.makedirs("insurance_configs", exist_ok=True)

insurance_types = {
    "car": {
        "type": "car",
        "inputs": {
            "annual_mileage": {
                "min": 0, "max": 30000, "step": 50,
                "mfs": {
                    "Low": {"kind": "trapmf", "params": [0, 0, 5000, 10000]},
                    "Medium": {"kind": "trimf", "params": [5000, 15000, 20000]},
                    "High": {"kind": "trapmf", "params": [15000, 20000, 30000, 30000]}
                }
            },
            "driver_age": {
                "min": 16, "max": 85, "step": 1,
                "mfs": {
                    "Young": {"kind": "trapmf", "params": [16, 16, 25, 35]},
                    "Prime": {"kind": "trimf", "params": [25, 40, 60]},
                    "Senior": {"kind": "trapmf", "params": [60, 70, 85, 85]}
                }
            },
            "prior_claims": {
                "min": 0, "max": 5, "step": 1,
                "mfs": {
                    "None": {"kind": "trapmf", "params": [0, 0, 0, 1]},
                    "Few": {"kind": "trimf", "params": [0, 2, 4]},
                    "Many": {"kind": "trapmf", "params": [3, 4, 5, 5]}
                }
            }
        },
        "outputs": {
            "premium": {
                "min": 0, "max": 2000, "step": 1,
                "mfs": {
                    "VeryLow": {"kind": "trapmf", "params": [0, 0, 300, 500]},
                    "Low": {"kind": "trimf", "params": [300, 600, 900]},
                    "Medium": {"kind": "trimf", "params": [800, 1100, 1400]},
                    "High": {"kind": "trimf", "params": [1300, 1600, 1800]},
                    "VeryHigh": {"kind": "trapmf", "params": [1700, 1900, 2000, 2000]}
                }
            }
        },
        "rules": [
            {"if": [["annual_mileage", "Low"], ["driver_age", "Prime"], ["prior_claims", "None"]], "then": ["premium", "VeryLow"]},
            {"if": [["annual_mileage", "High"], ["prior_claims", "Many"]], "then": ["premium", "VeryHigh"]}
        ]
    },
    "health": {
        "type": "health",
        "inputs": {
            "age": {
                "min": 0, "max": 100, "step": 1,
                "mfs": {
                    "Child": {"kind": "trapmf", "params": [0, 0, 12, 18]},
                    "Adult": {"kind": "trimf", "params": [18, 40, 65]},
                    "Senior": {"kind": "trapmf", "params": [60, 75, 100, 100]}
                }
            },
            "pre_existing_conditions": {
                "min": 0, "max": 5, "step": 1,
                "mfs": {
                    "None": {"kind": "trapmf", "params": [0, 0, 0, 1]},
                    "Few": {"kind": "trimf", "params": [0, 2, 4]},
                    "Many": {"kind": "trapmf", "params": [3, 4, 5, 5]}
                }
            }
        },
        "outputs": {
            "premium": {
                "min": 0, "max": 3000, "step": 1,
                "mfs": {
                    "Low": {"kind": "trapmf", "params": [0, 0, 500, 1000]},
                    "Medium": {"kind": "trimf", "params": [800, 1500, 2200]},
                    "High": {"kind": "trapmf", "params": [2000, 2500, 3000, 3000]}
                }
            }
        },
        "rules": [
            {"if": [["age", "Child"], ["pre_existing_conditions", "None"]], "then": ["premium", "Low"]},
            {"if": [["age", "Senior"], ["pre_existing_conditions", "Many"]], "then": ["premium", "High"]}
        ]
    },
    "house": {
        "type": "house",
        "inputs": {
            "house_age": {
                "min": 0, "max": 100, "step": 1,
                "mfs": {
                    "New": {"kind": "trapmf", "params": [0, 0, 5, 15]},
                    "Mid": {"kind": "trimf", "params": [10, 30, 60]},
                    "Old": {"kind": "trapmf", "params": [50, 80, 100, 100]}
                }
            },
            "location_risk": {
                "min": 0, "max": 10, "step": 1,
                "mfs": {
                    "Low": {"kind": "trapmf", "params": [0, 0, 2, 4]},
                    "Medium": {"kind": "trimf", "params": [3, 5, 7]},
                    "High": {"kind": "trapmf", "params": [6, 8, 10, 10]}
                }
            }
        },
        "outputs": {
            "premium": {
                "min": 0, "max": 5000, "step": 1,
                "mfs": {
                    "Low": {"kind": "trapmf", "params": [0, 0, 1000, 2000]},
                    "Medium": {"kind": "trimf", "params": [1500, 3000, 4000]},
                    "High": {"kind": "trapmf", "params": [3500, 4500, 5000, 5000]}
                }
            }
        },
        "rules": [
            {"if": [["house_age", "New"], ["location_risk", "Low"]], "then": ["premium", "Low"]},
            {"if": [["house_age", "Old"], ["location_risk", "High"]], "then": ["premium", "High"]}
        ]
    },
    "life": {
        "type": "life",
        "inputs": {
            "age": {
                "min": 0, "max": 100, "step": 1,
                "mfs": {
                    "Young": {"kind": "trapmf", "params": [0, 0, 20, 30]},
                    "Adult": {"kind": "trimf", "params": [25, 45, 65]},
                    "Senior": {"kind": "trapmf", "params": [60, 80, 100, 100]}
                }
            },
            "smoker": {
                "min": 0, "max": 1, "step": 1,
                "mfs": {
                    "No": {"kind": "trapmf", "params": [0, 0, 0, 0.5]},
                    "Yes": {"kind": "trapmf", "params": [0.5, 1, 1, 1]}
                }
            }
        },
        "outputs": {
            "premium": {
                "min": 0, "max": 4000, "step": 1,
                "mfs": {
                    "Low": {"kind": "trapmf", "params": [0, 0, 800, 1600]},
                    "Medium": {"kind": "trimf", "params": [1200, 2000, 3000]},
                    "High": {"kind": "trapmf", "params": [2500, 3500, 4000, 4000]}
                }
            }
        },
        "rules": [
            {"if": [["age", "Young"], ["smoker", "No"]], "then": ["premium", "Low"]},
            {"if": [["age", "Senior"], ["smoker", "Yes"]], "then": ["premium", "High"]}
        ]
    }
}

for name, cfg in insurance_types.items():
    with open(f"insurance_configs/{name}_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        