import json
import re
from pathlib import Path

import pymupdf


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PDF_PATH = PROJECT_ROOT / "data" / "raw" / "the sign of four.pdf"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

PAGES_OUTPUT = OUTPUT_DIR / "pages.jsonl"
CHUNKS_OUTPUT = OUTPUT_DIR / "chunks.jsonl"
METADATA_OUTPUT = OUTPUT_DIR / "corpus_metadata.json"


# ============================================================
# CHAPTER MAP
# Verified from PDF diagnostic
# ============================================================

CHAPTER_MAP = {
    1: 17,
    2: 38,
    3: 51,
    4: 63,
    5: 88,
    6: 106,
    7: 129,
    8: 159,
    9: 182,
    10: 209,
    11: 233,
    12: 247,
}


ROMAN = {
    1: "I",
    2: "II",
    3: "III",
    4: "IV",
    5: "V",
    6: "VI",
    7: "VII",
    8: "VIII",
    9: "IX",
    10: "X",
    11: "XI",
    12: "XII",
}


# ============================================================
# CHUNK SETTINGS
# ============================================================

TARGET_CHUNK_SIZE = 1600
MAX_CHUNK_SIZE = 2000
MIN_CHUNK_SIZE = 500


# ============================================================
# TEXT CLEANING
# ============================================================

def normalize_unicode(text):

    if not text:
        return ""

    replacements = {
        "\x00": "",
        "\u00ad": "",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "--",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def clean_line(line):

    line = normalize_unicode(line)

    line = re.sub(
        r"\s+",
        " ",
        line
    )

    return line.strip()


def is_page_number(line):

    line = clean_line(line)

    if not line:
        return False

    return bool(
        re.fullmatch(
            r"\d+",
            line
        )
    )


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf():

    if not PDF_PATH.exists():

        raise FileNotFoundError(
            f"PDF not found:\n{PDF_PATH}"
        )

    document = pymupdf.open(
        PDF_PATH
    )

    print("=" * 80)
    print("RAW PDF EXTRACTION")
    print("=" * 80)

    print(
        f"PDF: {PDF_PATH}"
    )

    print(
        f"PDF pages: {len(document)}"
    )

    pages = []

    for page_number, page in enumerate(
        document,
        start=1
    ):

        # Use sorted text extraction.
        text = page.get_text(
            "text",
            sort=True
        )

        text = normalize_unicode(
            text
        )

        pages.append({
            "pdf_page": page_number,
            "raw_text": text
        })

    document.close()

    print(
        f"Extracted {len(pages)} pages."
    )

    return pages


# ============================================================
# RECONSTRUCT TEXT
# ============================================================

def reconstruct_text(raw_text):

    raw_lines = raw_text.splitlines()

    paragraphs = []
    current = []

    for raw_line in raw_lines:

        line = clean_line(
            raw_line
        )

        if not line:
            if current:
                paragraphs.append(
                    " ".join(current)
                )
                current = []

            continue

        if is_page_number(line):
            continue

        # Don't treat chapter headings as normal prose.
        if re.match(
            r"^CHAPTER\b",
            line,
            re.IGNORECASE
        ):

            if current:
                paragraphs.append(
                    " ".join(current)
                )
                current = []

            paragraphs.append(
                line
            )

            continue

        current.append(
            line
        )

    if current:
        paragraphs.append(
            " ".join(current)
        )

    result = []

    for paragraph in paragraphs:

        # Fix words split across PDF line breaks.
        paragraph = re.sub(
            r"(\w)-\s+(\w)",
            r"\1\2",
            paragraph
        )

        paragraph = re.sub(
            r"\s+",
            " ",
            paragraph
        )

        paragraph = paragraph.strip()

        if paragraph:
            result.append(
                paragraph
            )

    return "\n\n".join(
        result
    )


# ============================================================
# CHAPTER ASSIGNMENT
# ============================================================

def get_chapter_for_page(
    page_number
):

    current_chapter = None

    for chapter, start_page in CHAPTER_MAP.items():

        if page_number >= start_page:
            current_chapter = chapter

        else:
            break

    return current_chapter


def get_chapter_heading(
    chapter
):

    if chapter is None:
        return None

    return (
        f"CHAPTER {ROMAN[chapter]}."
    )


# ============================================================
# PROCESS PAGES
# ============================================================

def process_pages(
    raw_pages
):

    pages = []

    novel_start = CHAPTER_MAP[1]

    # Novel ends before next content after Chapter XII.
    # We retain pages through the last non-empty novel page.
    for page in raw_pages:

        page_number = page[
            "pdf_page"
        ]

        # ----------------------------------------------------
        # Remove front matter.
        # ----------------------------------------------------

        if page_number < novel_start:
            continue

        text = reconstruct_text(
            page["raw_text"]
        )

        if not text.strip():
            continue

        chapter = get_chapter_for_page(
            page_number
        )

        heading = get_chapter_heading(
            chapter
        )

        pages.append({

            "document_id":
                "the_sign_of_four",

            "pdf_page":
                page_number,

            "chapter_number":
                chapter,

            "chapter_heading":
                heading,

            "text":
                text,

            "character_count":
                len(text),

            "word_count":
                len(text.split())

        })

    return pages


# ============================================================
# CHAPTER VALIDATION
# ============================================================

def validate_chapters():

    print("\n")
    print("=" * 80)
    print("CHAPTER MAP")
    print("=" * 80)

    for chapter in range(1, 13):

        print(
            f"Chapter {chapter:02d} | "
            f"PDF page {CHAPTER_MAP[chapter]:03d} | "
            f"CHAPTER {ROMAN[chapter]}."
        )

    print(
        "\nAll 12 chapter boundaries are defined."
    )


# ============================================================
# SPLIT LONG PARAGRAPH
# ============================================================

def split_long_paragraph(
    paragraph
):

    if len(paragraph) <= MAX_CHUNK_SIZE:
        return [paragraph]

    sentences = re.split(
        r"(?<=[.!?])\s+",
        paragraph
    )

    pieces = []
    current = ""

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        if not current:

            current = sentence

        elif (
            len(current)
            + len(sentence)
            + 1
            <= MAX_CHUNK_SIZE
        ):

            current += (
                " "
                + sentence
            )

        else:

            pieces.append(
                current
            )

            current = sentence

    if current:
        pieces.append(
            current
        )

    return pieces


# ============================================================
# CHUNK CREATION
# ============================================================

def create_chunks(
    pages
):

    print("\n")
    print("=" * 80)
    print("CREATING CHUNKS")
    print("=" * 80)

    chunks = []

    current_parts = []
    current_pages = []

    current_chapter = None
    current_heading = None

    def current_size():

        return sum(
            len(part)
            for part in current_parts
        ) + max(
            0,
            len(current_parts) - 1
        ) * 2

    def flush():

        nonlocal current_parts
        nonlocal current_pages

        if not current_parts:
            return

        text = "\n\n".join(
            current_parts
        ).strip()

        if not text:
            return

        chunks.append({

            "chunk_id":
                f"sof_{len(chunks)+1:04d}",

            "document_id":
                "the_sign_of_four",

            "chapter_number":
                current_chapter,

            "chapter_heading":
                current_heading,

            "pdf_page_start":
                min(current_pages),

            "pdf_page_end":
                max(current_pages),

            "text":
                text,

            "character_count":
                len(text),

            "word_count":
                len(text.split())

        })

        current_parts = []
        current_pages = []

    # --------------------------------------------------------
    # Process each page
    # --------------------------------------------------------

    for page in pages:

        chapter = page[
            "chapter_number"
        ]

        heading = page[
            "chapter_heading"
        ]

        # ----------------------------------------------------
        # NEVER allow a chunk to cross chapters.
        # ----------------------------------------------------

        if (
            current_chapter is not None
            and chapter != current_chapter
        ):

            flush()

        current_chapter = chapter
        current_heading = heading

        paragraphs = page[
            "text"
        ].split("\n\n")

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if not paragraph:
                continue

            # Skip chapter heading itself.
            if re.match(
                r"^CHAPTER\b",
                paragraph,
                re.IGNORECASE
            ):
                continue

            pieces = split_long_paragraph(
                paragraph
            )

            for piece in pieces:

                projected_size = (
                    current_size()
                    + len(piece)
                    + 2
                )

                if (
                    current_parts
                    and projected_size
                    > TARGET_CHUNK_SIZE
                ):

                    flush()

                current_parts.append(
                    piece
                )

                current_pages.append(
                    page[
                        "pdf_page"
                    ]
                )

                if (
                    current_size()
                    >= MAX_CHUNK_SIZE
                ):

                    flush()

    flush()

    return chunks


# ============================================================
# MERGE TINY CHUNKS
# ============================================================

def merge_tiny_chunks(
    chunks
):

    if not chunks:
        return chunks

    merged = []

    for chunk in chunks:

        if (
            merged
            and
            chunk["character_count"]
            < MIN_CHUNK_SIZE
            and
            chunk["chapter_number"]
            ==
            merged[-1]["chapter_number"]
        ):

            previous = merged[-1]

            previous["text"] += (
                "\n\n"
                + chunk["text"]
            )

            previous["pdf_page_end"] = (
                chunk[
                    "pdf_page_end"
                ]
            )

            previous["character_count"] = (
                len(
                    previous["text"]
                )
            )

            previous["word_count"] = (
                len(
                    previous["text"].split()
                )
            )

        else:

            merged.append(
                chunk
            )

    # Reassign chunk IDs.
    for index, chunk in enumerate(
        merged,
        start=1
    ):

        chunk[
            "chunk_id"
        ] = f"sof_{index:04d}"

    return merged


# ============================================================
# VALIDATE PAGES
# ============================================================

def validate_pages(
    pages
):

    print("\n")
    print("=" * 80)
    print("PAGE VALIDATION")
    print("=" * 80)

    print(
        f"Novel pages retained : "
        f"{len(pages)}"
    )

    print(
        f"First PDF page       : "
        f"{pages[0]['pdf_page']}"
    )

    print(
        f"Last PDF page        : "
        f"{pages[-1]['pdf_page']}"
    )

    total_chars = sum(
        page["character_count"]
        for page in pages
    )

    print(
        f"Total characters     : "
        f"{total_chars:,}"
    )

    print(
        f"Average chars/page   : "
        f"{total_chars / len(pages):,.0f}"
    )


# ============================================================
# VALIDATE CHUNKS
# ============================================================

def validate_chunks(
    chunks
):

    print("\n")
    print("=" * 80)
    print("CHUNK VALIDATION")
    print("=" * 80)

    print(
        f"Total chunks: "
        f"{len(chunks)}"
    )

    sizes = [
        chunk["character_count"]
        for chunk in chunks
    ]

    print(
        f"Minimum size: "
        f"{min(sizes)}"
    )

    print(
        f"Maximum size: "
        f"{max(sizes)}"
    )

    print(
        f"Average size: "
        f"{sum(sizes) / len(sizes):.0f}"
    )

    tiny = [
        chunk
        for chunk in chunks
        if chunk[
            "character_count"
        ] < MIN_CHUNK_SIZE
    ]

    print(
        f"Tiny chunks (<{MIN_CHUNK_SIZE}): "
        f"{len(tiny)}"
    )

    chapters = sorted(
        set(
            chunk[
                "chapter_number"
            ]
            for chunk in chunks
        )
    )

    print(
        f"Chapters represented: "
        f"{chapters}"
    )

    missing = set(
        range(1, 13)
    ) - set(
        chapters
    )

    if missing:

        print(
            "WARNING - missing chapters:",
            sorted(missing)
        )

    else:

        print(
            "All 12 chapters represented."
        )

    # Check citation metadata.
    bad_page_metadata = [
        chunk
        for chunk in chunks
        if not chunk.get(
            "pdf_page_start"
        )
        or not chunk.get(
            "pdf_page_end"
        )
    ]

    print(
        "Chunks missing page metadata:",
        len(bad_page_metadata)
    )

    print("\nSAMPLE CHUNKS")

    for chunk in chunks[:5]:

        print("\n")
        print("-" * 80)

        print(
            f"ID: {chunk['chunk_id']}"
        )

        print(
            f"Chapter: "
            f"{chunk['chapter_number']}"
        )

        print(
            f"Pages: "
            f"{chunk['pdf_page_start']}-"
            f"{chunk['pdf_page_end']}"
        )

        print(
            f"Characters: "
            f"{chunk['character_count']}"
        )

        print(
            chunk["text"][:1000]
        )


# ============================================================
# SAVE JSONL
# ============================================================

def save_jsonl(
    path,
    records
):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        for record in records:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 1. Extract PDF
    # --------------------------------------------------------

    raw_pages = extract_pdf()

    # --------------------------------------------------------
    # 2. Process novel pages
    # --------------------------------------------------------

    pages = process_pages(
        raw_pages
    )

    # --------------------------------------------------------
    # 3. Validate chapter map
    # --------------------------------------------------------

    validate_chapters()

    # --------------------------------------------------------
    # 4. Create chunks
    # --------------------------------------------------------

    chunks = create_chunks(
        pages
    )

    # --------------------------------------------------------
    # 5. Merge tiny chunks
    # --------------------------------------------------------

    chunks = merge_tiny_chunks(
        chunks
    )

    # --------------------------------------------------------
    # 6. Validate
    # --------------------------------------------------------

    validate_pages(
        pages
    )

    validate_chunks(
        chunks
    )

    # --------------------------------------------------------
    # 7. Save pages
    # --------------------------------------------------------

    save_jsonl(
        PAGES_OUTPUT,
        pages
    )

    # --------------------------------------------------------
    # 8. Save chunks
    # --------------------------------------------------------

    save_jsonl(
        CHUNKS_OUTPUT,
        chunks
    )

    # --------------------------------------------------------
    # 9. Save metadata
    # --------------------------------------------------------

    metadata = {

        "document_id":
            "the_sign_of_four",

        "source_file":
            "the sign of four.pdf",

        "total_pdf_pages":
            len(raw_pages),

        "novel_start_page":
            CHAPTER_MAP[1],

        "novel_end_page":
            max(
                page["pdf_page"]
                for page in pages
            ),

        "retained_pages":
            len(pages),

        "total_chunks":
            len(chunks),

        "chapter_map":
            CHAPTER_MAP,

        "chunk_configuration": {

            "target_size":
                TARGET_CHUNK_SIZE,

            "max_size":
                MAX_CHUNK_SIZE,

            "min_size":
                MIN_CHUNK_SIZE

        }

    }

    with open(
        METADATA_OUTPUT,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)

    print(
        f"Pages   : {PAGES_OUTPUT}"
    )

    print(
        f"Chunks  : {CHUNKS_OUTPUT}"
    )

    print(
        f"Metadata: {METADATA_OUTPUT}"
    )


if __name__ == "__main__":
    main()