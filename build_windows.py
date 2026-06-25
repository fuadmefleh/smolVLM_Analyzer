#!/usr/bin/env python3
"""
Build script: packages the SmolVLM Highlighter into a Windows installer.

Usage (run on Windows or via Wine on Linux):
    python build_windows.py [--skip-model-download]

Outputs:
    dist/SmolVLMHighlighter/   — PyInstaller bundle (standalone folder)
    installer/setup.iss        — Inno Setup script
    installer/SmolVLMHighlighter_Setup.exe  — final installer (if ISCC is on PATH)

Requirements:
    pip install pyinstaller
    Inno Setup 6 (https://jrsoftware.org/isinfo.php) — for the installer step
    ffmpeg.exe + ffprobe.exe placed in the project root (bundled into the exe)
"""

import argparse
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
DIST = ROOT / "dist" / "SmolVLMHighlighter"
INSTALLER_DIR = ROOT / "installer"
APP_NAME = "SmolVLM Highlighter"
APP_VERSION = "1.0.0"
PUBLISHER = "SmolVLM Highlighter"
DEFAULT_PORT = "8019"


def run(cmd, **kwargs):
    print(f">>> {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def check_ffmpeg_binaries():
    for name in ("ffmpeg.exe", "ffprobe.exe"):
        p = ROOT / name
        if not p.exists():
            sys.exit(
                f"ERROR: {name} not found in project root.\n"
                "Download a Windows ffmpeg build from https://www.gyan.dev/ffmpeg/builds/ "
                f"and place {name} alongside build_windows.py."
            )
    print("ffmpeg.exe and ffprobe.exe found.")


def download_model():
    """Pre-download the model weights so the installer bundles them."""
    print("Pre-downloading model weights (this may take a few minutes)...")
    script = textwrap.dedent("""\
        from transformers import AutoProcessor, AutoModelForImageTextToText
        model_name = "HuggingFaceTB/SmolVLM-500M-Instruct"
        AutoProcessor.from_pretrained(model_name)
        AutoModelForImageTextToText.from_pretrained(model_name)
        print("Model cached successfully.")
    """)
    subprocess.run([sys.executable, "-c", script], check=True)


def write_launcher():
    """Write a thin launcher script that uvicorn picks up."""
    launcher = ROOT / "_launcher.py"
    launcher.write_text(
        textwrap.dedent(f"""\
            import os, sys, multiprocessing
            # PyInstaller sets sys._MEIPASS; add it to PATH so ffmpeg is found.
            if hasattr(sys, "_MEIPASS"):
                os.environ["PATH"] = sys._MEIPASS + os.pathsep + os.environ.get("PATH", "")
            import uvicorn
            if __name__ == "__main__":
                multiprocessing.freeze_support()
                port = int(os.environ.get("PORT", "{DEFAULT_PORT}"))
                uvicorn.run("app:app", host="0.0.0.0", port=port)
        """)
    )
    return launcher


def build_pyinstaller(launcher: Path, bundle_model: bool = False):
    # Auto-install PyInstaller if not present
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller not found — installing...")
        run([sys.executable, "-m", "pip", "install", "pyinstaller"])

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--name", "SmolVLMHighlighter",
        "--distpath", str(ROOT / "dist"),
        "--workpath", str(ROOT / "build"),
        "--specpath", str(ROOT),
        # Bundle the web frontend
        "--add-data", f"{ROOT / 'static'}{os.pathsep}static",
        # Bundle ffmpeg binaries
        "--add-binary", f"{ROOT / 'ffmpeg.exe'}{os.pathsep}.",
        "--add-binary", f"{ROOT / 'ffprobe.exe'}{os.pathsep}.",
        # Hidden imports that PyInstaller misses
        "--hidden-import", "uvicorn.logging",
        "--hidden-import", "uvicorn.loops",
        "--hidden-import", "uvicorn.loops.auto",
        "--hidden-import", "uvicorn.protocols",
        "--hidden-import", "uvicorn.protocols.http",
        "--hidden-import", "uvicorn.protocols.http.auto",
        "--hidden-import", "uvicorn.lifespan",
        "--hidden-import", "uvicorn.lifespan.on",
        "--hidden-import", "transformers",
        "--hidden-import", "torch",
        "--hidden-import", "PIL",
        "--hidden-import", "av",
        "--collect-all", "transformers",
        "--collect-all", "tokenizers",
        str(launcher),
    ]

    if bundle_model:
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        if not hf_cache.exists():
            sys.exit(
                "ERROR: HuggingFace cache not found. Run without --no-bundle-model "
                "or download the model first."
            )
        cmd += ["--add-data", f"{hf_cache}{os.pathsep}huggingface_cache"]

    run(cmd, cwd=ROOT)


def write_nsis_script():
    INSTALLER_DIR.mkdir(exist_ok=True)
    nsi = INSTALLER_DIR / "setup.nsi"
    # NSIS uses backslash paths; convert from POSIX
    dist_win = str(DIST).replace("/", "\\")
    out_win = str(INSTALLER_DIR).replace("/", "\\")
    nsi.write_text(
        textwrap.dedent(f"""\
            Unicode True
            !include "MUI2.nsh"

            Name "{APP_NAME}"
            OutFile "{out_win}\\SmolVLMHighlighter_Setup.exe"
            InstallDir "$PROGRAMFILES64\\{APP_NAME}"
            InstallDirRegKey HKLM "Software\\{APP_NAME}" "Install_Dir"
            RequestExecutionLevel admin

            !define MUI_ABORTWARNING
            !insertmacro MUI_PAGE_WELCOME
            !insertmacro MUI_PAGE_DIRECTORY
            !insertmacro MUI_PAGE_INSTFILES
            !define MUI_FINISHPAGE_RUN "$INSTDIR\\SmolVLMHighlighter.exe"
            !define MUI_FINISHPAGE_RUN_TEXT "Launch {APP_NAME}"
            !insertmacro MUI_PAGE_FINISH

            !insertmacro MUI_UNPAGE_CONFIRM
            !insertmacro MUI_UNPAGE_INSTFILES

            !insertmacro MUI_LANGUAGE "English"

            VIProductVersion "{APP_VERSION}.0"
            VIAddVersionKey "ProductName" "{APP_NAME}"
            VIAddVersionKey "FileVersion" "{APP_VERSION}"
            VIAddVersionKey "LegalCopyright" "{PUBLISHER}"

            Section "Install"
              SetOutPath "$INSTDIR"
              File /r "{dist_win}\\*.*"

              WriteRegStr HKLM "Software\\{APP_NAME}" "Install_Dir" "$INSTDIR"
              WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{APP_NAME}" \\
                "DisplayName" "{APP_NAME}"
              WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{APP_NAME}" \\
                "UninstallString" '"$INSTDIR\\uninstall.exe"'
              WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{APP_NAME}" \\
                "DisplayVersion" "{APP_VERSION}"
              WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{APP_NAME}" \\
                "Publisher" "{PUBLISHER}"
              WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{APP_NAME}" \\
                "NoModify" 1
              WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{APP_NAME}" \\
                "NoRepair" 1
              WriteUninstaller "$INSTDIR\\uninstall.exe"

              CreateDirectory "$SMPROGRAMS\\{APP_NAME}"
              CreateShortcut "$SMPROGRAMS\\{APP_NAME}\\{APP_NAME}.lnk" "$INSTDIR\\SmolVLMHighlighter.exe"
              CreateShortcut "$SMPROGRAMS\\{APP_NAME}\\Uninstall.lnk" "$INSTDIR\\uninstall.exe"
              CreateShortcut "$DESKTOP\\{APP_NAME}.lnk" "$INSTDIR\\SmolVLMHighlighter.exe"
            SectionEnd

            Section "Uninstall"
              Delete "$INSTDIR\\uninstall.exe"
              RMDir /r "$INSTDIR"
              Delete "$SMPROGRAMS\\{APP_NAME}\\{APP_NAME}.lnk"
              Delete "$SMPROGRAMS\\{APP_NAME}\\Uninstall.lnk"
              RMDir "$SMPROGRAMS\\{APP_NAME}"
              Delete "$DESKTOP\\{APP_NAME}.lnk"
              DeleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{APP_NAME}"
              DeleteRegKey HKLM "Software\\{APP_NAME}"
            SectionEnd
        """)
    )
    print(f"NSIS script written to {nsi}")
    return nsi


def run_nsis(nsi: Path):
    makensis = shutil.which("makensis")
    if not makensis:
        # Chocolatey and common install locations on Windows
        for candidate in [
            r"C:\ProgramData\chocolatey\bin\makensis.exe",
            r"C:\Program Files (x86)\NSIS\makensis.exe",
            r"C:\Program Files\NSIS\makensis.exe",
        ]:
            if os.path.isfile(candidate):
                makensis = candidate
                break
    if not makensis:
        print(
            "\nINFO: makensis not found. Install NSIS and re-run, or run manually:\n"
            f"  makensis \"{nsi}\"\n"
        )
        return
    print(f"Using makensis: {makensis}")
    run([makensis, str(nsi)])
    out = INSTALLER_DIR / "SmolVLMHighlighter_Setup.exe"
    if out.exists():
        print(f"\nInstaller ready: {out}")
    else:
        sys.exit("ERROR: makensis ran but installer was not produced — check NSIS output above.")


def main():
    parser = argparse.ArgumentParser(description="Build Windows installer for SmolVLM Highlighter")
    parser.add_argument("--skip-model-download", action="store_true",
                        help="Skip pre-downloading HuggingFace model weights")
    parser.add_argument("--no-bundle-model", action="store_true",
                        help="Don't bundle model weights — app downloads them on first launch")
    parser.add_argument("--skip-nsis", action="store_true",
                        help="Write the NSI script but do not run makensis (run it manually or in a separate CI step)")
    args = parser.parse_args()

    print(f"=== Building {APP_NAME} v{APP_VERSION} ===\n")

    check_ffmpeg_binaries()

    if not args.skip_model_download and not args.no_bundle_model:
        download_model()

    launcher = write_launcher()

    try:
        build_pyinstaller(launcher, bundle_model=not args.no_bundle_model)
    finally:
        launcher.unlink(missing_ok=True)

    nsi = write_nsis_script()
    if not args.skip_nsis:
        run_nsis(nsi)

    print("\nDone. Bundle is in dist/SmolVLMHighlighter/")


if __name__ == "__main__":
    main()
