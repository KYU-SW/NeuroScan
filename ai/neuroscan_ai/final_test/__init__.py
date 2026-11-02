"""Final Test evaluation of NeuroScan v1 (2D vs 2.5D), pre-registered in configs/final_test_v1.yaml.

This package is the only code allowed to construct ``SplitGuard(..., allow_test=True)`` and only
after every pre-flight check has passed and the run lock has been taken (see ``pipeline``).
Modes: ``validation-dry-run`` (default; the validation split rehearses the whole procedure, the
test split is never touched) and ``final-test`` (explicit approval required).
"""
