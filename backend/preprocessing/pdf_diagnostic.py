import re
from pathlib import Path

import pymupdf


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "the sign of four.pdf"
)


# ============================================================
# BASIC CLEANING
# ============================================================

def clean_line(line: str) -> str:

    if not line:
        return ""

    # Remove control characters
    line = line.replace("\x00", "")
    line = line.replace("\u00ad", "")

    # Normalize whitespace
    line = re.sub(r"\s+", " ", line)

    return line.strip()


# ============================================================
# PDF EXTRACTION
# ============================================================

def inspect_pdf(pdf_path: Path):

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found:\n{pdf_path}"
        )

    document = pymupdf.open(pdf_path)

    print("=" * 80)
    print("PDF DIAGNOSTIC")
    print("=" * 80)

    print(f"PDF       : {pdf_path}")
    print(f"Total pages: {len(document)}")

    print("\n" + "=" * 80)
    print("PAGE INFORMATION")
    print("=" * 80)

    for page_number, page in enumerate(
        document,
        start=1
    ):

        text = page.get_text("text")

        stripped = text.strip()

        print(
            f"Page {page_number:03d} | "
            f"Characters: {len(text):5d} | "
            f"Lines: {len(text.splitlines()):4d}"
        )

    document.close()


# ============================================================
# SAMPLE PAGE INSPECTION
# ============================================================

def show_sample_pages(pdf_path: Path):

    document = pymupdf.open(pdf_path)

    sample_pages = [
        1,
        2,
        3,
        50,
        100,
        105,
        106,
        107,
        108,
        109,
        110,
        120,
        150,
        200,
        250,
        300,
        322
    ]

    print("\n" + "=" * 80)
    print("SAMPLE PAGE TEXT")
    print("=" * 80)

    for page_number in sample_pages:

        if page_number > len(document):
            continue

        page = document[page_number - 1]

        text = page.get_text("text")

        print("\n")
        print("#" * 80)
        print(f"PDF PAGE {page_number}")
        print("#" * 80)

        if not text.strip():

            print("[NO TEXT]")

        else:

            print(text[:2500])

    document.close()


# ============================================================
# CHAPTER-LIKE HEADINGS
# ============================================================

def find_chapter_candidates(pdf_path: Path):

    document = pymupdf.open(pdf_path)

    print("\n" + "=" * 80)
    print("CHAPTER CANDIDATES")
    print("=" * 80)

    patterns = [
        r"\bCHAPTER\b",
        r"\bChapter\b"
    ]

    found = []

    for page_number, page in enumerate(
        document,
        start=1
    ):

        text = page.get_text("text")

        for line in text.splitlines():

            cleaned = clean_line(line)

            if not cleaned:
                continue

            if any(
                re.search(pattern, cleaned)
                for pattern in patterns
            ):

                found.append(
                    (
                        page_number,
                        cleaned
                    )
                )

    document.close()

    for page_number, line in found:

        print(
            f"Page {page_number:03d}: {line}"
        )

    print(
        f"\nTotal chapter-like lines: {len(found)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not PDF_PATH.exists():

        raise FileNotFoundError(
            f"\nPDF does not exist:\n{PDF_PATH}"
        )

    inspect_pdf(PDF_PATH)

    show_sample_pages(PDF_PATH)

    find_chapter_candidates(PDF_PATH)


if __name__ == "__main__":
    main()