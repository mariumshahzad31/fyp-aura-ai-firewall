from evo_integration import load_policy

policy = load_policy()
if policy:
    print('Evo policy loaded:', policy.include_rf_proba)
else:
    print('No evo policy')