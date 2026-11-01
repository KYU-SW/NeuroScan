"""Phase 7-B Research Validation Mode (Validation split only; the test split stays sealed).

Dataset-independent pieces (difference labelmap, component matching) live in ``compare`` and
``matching``; dataset-specific file discovery lives in ``sources`` adapters; ``bundle`` builds a
verified research case with the frozen Phase 5A inference engine. Importing this package does
not import torch (``bundle`` imports the inference engine lazily).
"""
