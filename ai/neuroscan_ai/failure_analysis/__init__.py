"""NeuroScan v1 failure-case analysis (qualitative review support), separate from the frozen
``final_test`` package.

Stage 1 (this package as implemented) is metrics-only: it reads the completed Final Test result
tables (CSV / JSON) read-only, selects review cases by pre-registered rules, assigns local aliases
and records objective metric-based flags. It never opens an image or prediction NIfTI, never runs
a model and never writes into the Final Test result folder. Nothing here is a medical judgement,
a statement about ground-truth correctness or a reason to change the v1 model or policy.
"""
