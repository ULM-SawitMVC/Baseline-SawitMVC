"""Synchronize the anonymous companion and build both revised ICIC PDFs.

Run from any directory: python scripts/build_icic_revision.py
The submitted main.tex is deliberately not used as the revision source.
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "draft-icic-2026"


def anonymous_source(source):
    start = source.index("\\author{")
    end = source.index("\\maketitle", start)
    source = source[:start] + "\\author{Anonymous Authors}\n\n" + source[end:]
    source = source.replace(
        "\\footnote{Source code: \\url{https://github.com/ULM-SawitMVC/Baseline-SawitMVC}.}",
        "\\footnote{Repository link withheld in this anonymous companion.}")
    # Remove both the heading and funding paragraph, not just the heading.
    source, count = re.subn(
        r"\\section\*\{Acknowledgment\}.*?(?=\\balance)", "", source, flags=re.S)
    assert count == 1, "Acknowledgment section boundary changed"
    for marker in ("PRJ-36", "f.indriani@", "\\IEEEauthorblock", "anonymous.4open.science"):
        assert marker not in source, marker
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tectonic", type=Path, default=ROOT / "tectonic.exe")
    parser.add_argument("--max-pages", type=int, default=6,
                        help="Maximum pages per PDF (author's current limit: 6)")
    args = parser.parse_args()
    source = (DRAFT / "main-new.tex").read_text(encoding="utf-8-sig")
    (DRAFT / "main-blind.tex").write_text(anonymous_source(source), encoding="utf-8")
    for name in ("main-new", "main-blind"):
        subprocess.run([str(args.tectonic.resolve()), "--keep-logs", name + ".tex"],
                       cwd=DRAFT, check=True)
        log = (DRAFT / (name + ".log")).read_text(encoding="utf-8", errors="replace")
        assert "Overfull \\hbox" not in log, f"Horizontal overflow in {name}"
        assert not re.search(r"(?:Reference|Citation) .+ undefined", log), name
        assert "multiply defined" not in log, name
        with fitz.open(DRAFT / (name + ".pdf")) as pdf:
            assert len(pdf) <= args.max_pages, f"{name}: {len(pdf)} pages exceeds {args.max_pages}"
            assert "\u2014" not in "\n".join(page.get_text() for page in pdf), f"Em dash in {name}"
            # IEEE Xplore rejects PDFs carrying bookmarks or link annotations.
            assert not pdf.get_toc(), f"{name}: PDF bookmarks are not allowed"
            assert not any(page.get_links() for page in pdf), \
                f"{name}: PDF link annotations are not allowed"
            width, height = round(pdf[0].rect.width), round(pdf[0].rect.height)
            assert (width, height) == (595, 842), \
                f"{name}: page is {width}x{height} pt, ICIC requires A4 (595x842)"
    print("Built both revised PDFs: A4, within the page limit, no em dashes, "
          "no links or bookmarks; inspect layout before submission.")


if __name__ == "__main__":
    main()
