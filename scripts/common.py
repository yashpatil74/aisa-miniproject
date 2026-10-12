"""
Common environment and helper utilities for AMR prediction pipeline.
Ensures runtime libraries (OpenMP/libgomp) are cleanly loaded for LightGBM/XGBoost.
"""
import os
import sys

def setup_environment():
    """Ensure OpenMP runtime library (libgomp) is available in dynamic linker search path."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    libs_dir = os.path.join(project_root, ".venv", "lib", "python3.14", "site-packages", "scikit_learn.libs")
    
    if os.path.isdir(libs_dir) and libs_dir not in os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["LD_LIBRARY_PATH"] = libs_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        if not os.environ.get("_AMR_REEXEC"):
            os.environ["_AMR_REEXEC"] = "1"
            if len(sys.argv) > 1 and sys.argv[0] == "-c":
                cmd_args = [sys.executable, "-c"] + sys.argv[1:]
            else:
                cmd_args = [sys.executable] + sys.argv
            os.execv(sys.executable, cmd_args)

setup_environment()
