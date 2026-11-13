#!/usr/bin/env python3
"""
setup_and_debug.py - Unified Apple Silicon Installer, Secrets Manager & Debugger.

Usage:
  python setup_and_debug.py                  # Full interactive setup, API key / MLX config & model installer
  python setup_and_debug.py --check          # Non-interactive hardware, package, code & model status report
  python setup_and_debug.py --install-core   # Non-interactive install of all required packages & default models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.doctor import (
    MLX_PIPELINE_MODELS,
    check_and_repair_environment,
    detect_mac_specs,
    download_hf_model,
    ensure_campplus_installed,
    is_hf_model_cached,
    print_hardware_report,
    run_diagnostics_and_repair,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TranslateThisShow - Interactive Installer, Hardware Advisor, Secrets Manager & Debugger"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run non-interactive diagnostics, Mac hardware compatibility report, and model cache check",
    )
    parser.add_argument(
        "--install-core",
        action="store_true",
        help="Automatically install all required Python packages, CAM++ weights, and default MLX models",
    )
    args = parser.parse_args()

    if args.install_core:
        specs = detect_mac_specs()
        print_hardware_report(specs)
        ok = check_and_repair_environment(auto_install_packages=True)
        ensure_campplus_installed(auto_fix=True)
        for model in MLX_PIPELINE_MODELS:
            if model.required_by_default and not is_hf_model_cached(model.repo_id):
                download_hf_model(model.repo_id)
        return 0 if ok else 1

    return run_diagnostics_and_repair(check_only=args.check)


if __name__ == "__main__":
    sys.exit(main())
