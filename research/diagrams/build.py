#!/usr/bin/env python3
"""Inject shared style (defs + CSS) into Physical AI workflow SVGs.

Usage:
    python build.py                  # Process all SVGs in this directory
    python build.py 01-logical.svg   # Process specific file(s)
    python build.py --check          # Verify SVGs use shared styles (no modification)
    python build.py --render         # Also render PNGs via cairosvg
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

DIAGRAM_DIR = Path(__file__).parent
STYLE_DIR = DIAGRAM_DIR / "style"
DEFS_FILE = STYLE_DIR / "defs.svg"
CSS_FILE = STYLE_DIR / "classes.css"

DEFS_MARKER_START = "<!-- SHARED-DEFS-START -->"
DEFS_MARKER_END = "<!-- SHARED-DEFS-END -->"
STYLE_MARKER_START = "<!-- SHARED-STYLE-START -->"
STYLE_MARKER_END = "<!-- SHARED-STYLE-END -->"


def load_shared_defs() -> str:
    return DEFS_FILE.read_text().strip()


def load_shared_css() -> str:
    return CSS_FILE.read_text().strip()


def inject_styles(svg_content: str) -> str:
    """Replace or insert shared defs and CSS into an SVG string.

    On first injection, replaces the entire <defs> block with shared defs
    (diagram-specific defs like custom markers should use shared ones instead).
    CSS is wrapped in CDATA for XML parser compatibility.
    """
    defs_block = f"{DEFS_MARKER_START}\n{load_shared_defs()}\n{DEFS_MARKER_END}"
    css = load_shared_css()
    style_block = f"{STYLE_MARKER_START}\n<style><![CDATA[\n{css}\n]]></style>\n{STYLE_MARKER_END}"

    # Replace existing shared blocks if present
    if DEFS_MARKER_START in svg_content:
        svg_content = re.sub(
            rf"{re.escape(DEFS_MARKER_START)}.*?{re.escape(DEFS_MARKER_END)}",
            defs_block,
            svg_content,
            flags=re.DOTALL,
        )
    else:
        # Replace entire <defs>...</defs> with shared defs
        if re.search(r"<defs[^>]*>.*?</defs>", svg_content, re.DOTALL):
            svg_content = re.sub(
                r"<defs[^>]*>.*?</defs>",
                f"<defs>\n{defs_block}\n</defs>",
                svg_content,
                count=1,
                flags=re.DOTALL,
            )
        else:
            svg_content = re.sub(
                r"(<svg[^>]*>)",
                rf"\1\n<defs>\n{defs_block}\n</defs>",
                svg_content,
                count=1,
            )

    if STYLE_MARKER_START in svg_content:
        svg_content = re.sub(
            rf"{re.escape(STYLE_MARKER_START)}.*?{re.escape(STYLE_MARKER_END)}",
            style_block,
            svg_content,
            flags=re.DOTALL,
        )
    else:
        # Insert <style> right after </defs>
        if "</defs>" in svg_content:
            svg_content = svg_content.replace("</defs>", f"</defs>\n{style_block}", 1)
        else:
            svg_content = re.sub(
                r"(<svg[^>]*>)",
                rf"\1\n{style_block}",
                svg_content,
                count=1,
            )

    return svg_content


def check_svg(path: Path) -> bool:
    """Return True if SVG contains up-to-date shared styles."""
    content = path.read_text()
    if DEFS_MARKER_START not in content:
        print(f"  MISSING shared defs: {path.name}")
        return False
    if STYLE_MARKER_START not in content:
        print(f"  MISSING shared CSS: {path.name}")
        return False
    expected = inject_styles(content)
    if content != expected:
        print(f"  STALE styles: {path.name}")
        return False
    print(f"  OK: {path.name}")
    return True


def render_png(svg_path: Path) -> None:
    """Render SVG to PNG using cairosvg."""
    png_path = svg_path.with_suffix(".png")
    try:
        subprocess.run(
            ["uvx", "--from", "cairosvg", "cairosvg", str(svg_path), "-o", str(png_path), "-d", "150"],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"  Rendered: {png_path.name}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  Render failed for {svg_path.name}: {e}")


def get_svg_files(args: list[str]) -> list[Path]:
    """Get SVG files to process — from args or all in diagram dir."""
    if args:
        return [DIAGRAM_DIR / a for a in args]
    return sorted(DIAGRAM_DIR.glob("*.svg"))


def main():
    parser = argparse.ArgumentParser(description="Inject shared styles into diagram SVGs")
    parser.add_argument("files", nargs="*", help="SVG files to process (default: all)")
    parser.add_argument("--check", action="store_true", help="Check without modifying")
    parser.add_argument("--render", action="store_true", help="Also render PNGs")
    args = parser.parse_args()

    svg_files = get_svg_files(args.files)
    if not svg_files:
        print("No SVG files found.")
        return

    if args.check:
        ok = all(check_svg(f) for f in svg_files)
        sys.exit(0 if ok else 1)

    for svg_path in svg_files:
        content = svg_path.read_text()
        updated = inject_styles(content)
        if content != updated:
            svg_path.write_text(updated)
            print(f"  Updated: {svg_path.name}")
        else:
            print(f"  No change: {svg_path.name}")

        if args.render:
            render_png(svg_path)


if __name__ == "__main__":
    main()
