"""UTF-8 at CLI boundaries, including redirected Windows output; no ML imports."""
import sys


def configure_console():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
