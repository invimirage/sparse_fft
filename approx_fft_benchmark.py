from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_HISTORY_PATH = Path(__file__).with_name("history") / "approx_fft_benchmark.py"
_SPEC = spec_from_file_location("history_approx_fft_benchmark", _HISTORY_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load {_HISTORY_PATH}")
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

globals().update({name: getattr(_MODULE, name) for name in dir(_MODULE) if not name.startswith("__")})
