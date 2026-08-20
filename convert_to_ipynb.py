"""
Convert DSCA-ViT .py notebook scripts to .ipynb Jupyter notebooks.

Usage (run from the project root):
    python convert_to_ipynb.py

This will create .ipynb files next to the .py files in notebooks/.
"""

import json
import re
from pathlib import Path


def py_to_ipynb(py_path: Path, ipynb_path: Path) -> None:
    """Convert a .py script with cell markers to a .ipynb notebook."""

    text = py_path.read_text(encoding="utf-8")

    # Split on the cell marker pattern:
    #   # ============================================================
    #   # Cell N — Title
    #   # ============================================================
    #
    # We split on lines that look like "# =====...=====" (10+ equal signs)
    # and group the content between them into cells.

    lines = text.split("\n")

    cells = []
    current_cell_lines = []
    header_comment_lines = []   # Lines before the first cell marker
    in_header = True

    i = 0
    while i < len(lines):
        line = lines[i]

        # Detect cell separator: a line of "# ====...====" with 20+ '='
        if re.match(r"^# ={20,}$", line.strip()):

            # Check if next line is a cell title (# Cell N — ...)
            # Pattern: separator, title line(s), separator
            if i + 2 < len(lines) and re.match(r"^# ={20,}$", lines[i + 2].strip()):

                # Save previous cell
                if current_cell_lines:
                    cells.append(current_cell_lines)
                elif header_comment_lines and in_header:
                    # Save file-level header as a markdown cell
                    md_text = "\n".join(
                        line.lstrip("# ").rstrip() if line.startswith("#") else line
                        for line in header_comment_lines
                    ).strip()
                    if md_text:
                        cells.append(("markdown", md_text))

                in_header = False
                current_cell_lines = []

                # Skip the 3 separator lines (===, title, ===)
                title = lines[i + 1].lstrip("# ").strip()
                current_cell_lines.append(f"# {title}\n")
                i += 3
                continue

            elif i + 1 < len(lines):
                # Might be a single separator line, just include it
                if in_header:
                    header_comment_lines.append(line)
                else:
                    current_cell_lines.append(line)
                i += 1
                continue

        if in_header and not re.match(r"^# ={20,}$", line.strip()):
            header_comment_lines.append(line)
        else:
            current_cell_lines.append(line)

        i += 1

    # Save last cell
    if current_cell_lines:
        cells.append(current_cell_lines)

    # Build the notebook JSON
    nb_cells = []

    for cell in cells:

        if isinstance(cell, tuple) and cell[0] == "markdown":
            nb_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    line + "\n" for line in cell[1].split("\n")
                ]
            })
        else:
            # Code cell — clean up
            source_lines = cell

            # Strip trailing empty lines
            while source_lines and source_lines[-1].strip() == "":
                source_lines.pop()

            # Strip leading empty lines (but keep the title comment)
            while source_lines and source_lines[0].strip() == "" :
                source_lines.pop(0)

            if not source_lines:
                continue

            # Format each line with \n except the last
            formatted = []
            for j, line in enumerate(source_lines):
                if j < len(source_lines) - 1:
                    formatted.append(line + "\n")
                else:
                    formatted.append(line)

            nb_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": formatted
            })

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            },
            "colab": {
                "provenance": []
            }
        },
        "cells": nb_cells
    }

    ipynb_path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False),
        encoding="utf-8"
    )


def main():
    notebooks_dir = Path(__file__).parent / "notebooks"

    py_files = [
        "train.py",
        "train_stage2.py",
        "gate_analysis.py",
        "02_Cross_Attention_Analysis.py",
        "02b_Spatial_Bias_Initialization_vs_Trained.py",
        "03_DSCA_Ablation_Stream_and_CrossAttention.py",
        "04_DSCA_Input_Distribution_and_Fusion_Gate_Diagnostic.py",
        "05_DSCA_ViT_v2_Training.py",
        "06_DSCA_ViT_v2_Evaluation.py",
        "06_DSCA_ViT_v3_Training.py",
        "07_DSS_ViT_Training.py",
        "sanity_check.py",
        "visualize.py",
        "deconv_sanity_check.py",
    ]

    print("Converting .py → .ipynb")
    print("=" * 50)

    for filename in py_files:
        py_path = notebooks_dir / filename
        ipynb_path = notebooks_dir / filename.replace(".py", ".ipynb")

        if not py_path.exists():
            print(f"  ⚠️  Not found: {py_path}")
            continue

        py_to_ipynb(py_path, ipynb_path)
        print(f"  ✅ {filename} → {ipynb_path.name}")

    print("=" * 50)
    print("Done! Upload the .ipynb files to Google Colab.")


if __name__ == "__main__":
    main()
