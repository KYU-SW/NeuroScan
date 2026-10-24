"""NeuroScan-ai Phase 5B FastAPI backend (research prototype, not a medical device).

The API process never imports torch or loads the model; inference runs in the single GPU worker
(``python -m neuroscan_api.worker``) through the Phase 5A public service.
"""

__version__ = "0.1.0"
API_PREFIX = "/api/v1"
