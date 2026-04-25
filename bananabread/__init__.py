__version__ = "0.5.2"

def __getattr__(name):
    if name == "main":
        from .main import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
