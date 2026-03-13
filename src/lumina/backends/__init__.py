from lumina.backends.base import Backend
from lumina.backends.numpy_backend import NumpyBackend
from lumina.backends.python import PythonBackend

# registry of available backends
_BACKENDS = {
    "numpy": NumpyBackend,
    "python": PythonBackend,
}

def get_backend(name: str = "numpy") -> Backend:
    # grab the right backend by name
    # defaults to numpy because it's way faster
    if name not in _BACKENDS:
        available = ", ".join(_BACKENDS.keys())
        raise ValueError(f"unknown backend '{name}', available: {available}")
    return _BACKENDS[name]()
