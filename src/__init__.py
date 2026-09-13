# Python 3.12 compatibility patch for multiprocess resource tracker on Windows
try:
    import multiprocess.resource_tracker as rt
    def _safe_stop_locked(self, *args, **kwargs):
        pass
    rt.ResourceTracker._stop_locked = _safe_stop_locked
except Exception:
    pass
