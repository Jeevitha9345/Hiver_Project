import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure Python 3.12 Windows finalizer compatibility
try:
    import multiprocess.resource_tracker as rt
    def _safe_stop_locked(self, *args, **kwargs):
        pass
    rt.ResourceTracker._stop_locked = _safe_stop_locked
except Exception:
    pass
