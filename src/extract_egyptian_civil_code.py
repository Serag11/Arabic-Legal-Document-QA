#!/usr/bin/env python3
"""Extract the bilingual Egyptian Civil Code PDF into one JSONL record/article.

Designed for the uploaded two-column Arabic/English edition:
- English is in the left column, Arabic in the right column.
- Arabic-Indic article numbers are normalized to Python int.
- RTL text is reconstructed positionally (not with a naive page text dump).
- Repealed article ranges are detected and synthesized as records, never dropped.
- Bold hierarchy rows are carried forward as metadata.
- Search-normalized Arabic is stored separately from source-faithful Arabic.
- Optional QA HTML renders 20 random source-row crops for visual inspection.

`book` intentionally follows the PART title (e.g. "Obligations or Personal Rights")
to match the requested target schema. The literal PDF BOOK title is retained in
`source_book` as an extra field.
"""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import fitz  # PyMuPDF

try:
    from pyarabic import araby
except ImportError:  # script still works, but install dependencies shown below
    araby = None

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
except ImportError:
    arabic_reshaper = None
    get_display = None


AR_INDIC = "٠١٢٣٤٥٦٧٨٩"
EA_INDIC = "۰۱۲۳۴۵۶۷۸۹"
DIGIT_TRANS = str.maketrans(AR_INDIC + EA_INDIC, "0123456789" * 2)
AR_DIGIT_RUN_RE = re.compile(r"[٠-٩۰-۹]+")
BIDI_CONTROL_RE = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]")

EN_ARTICLE_RE = re.compile(
    r"^(?:Article|rticle)\s*(\d{1,4})\s*\.?\s*$", re.IGNORECASE
)
AR_ARTICLE_RE = re.compile(
    r"^مادة\s*[() ]*([٠-٩۰-۹]+)\s*[() ]*$"
)
REPEAL_RE = re.compile(
    r"\bArticles?\s+(\d{1,4})\s*[-–]\s*(\d{1,4})\b.*\brepealed\b",
    re.IGNORECASE,
)

STOPWORDS = {
    "and", "or", "of", "the", "in", "on", "as", "to", "for", "from", "by", "with", "at"
}


def clean_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = BIDI_CONTROL_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fix_visual_arabic_digit_runs(text: str) -> str:
    """Repair Arabic-Indic digit runs stored in visual RTL order by this PDF.

    Example: PyMuPDF may extract the displayed article ١٤٧ as ٧٤١ while the
    surrounding Arabic words are otherwise in the right logical order.
    """
    return AR_DIGIT_RUN_RE.sub(lambda m: m.group(0)[::-1], text)


def normalize_arabic_parenthetical_numbers(text: str) -> str:
    # PDF extraction often yields '( ١ (' or ') ١ (' for a displayed '(١)'.
    return re.sub(
        r"[()]\s*([٠-٩۰-۹0-9]+)\s*[()]",
        lambda m: f"({m.group(1)})",
        text,
    )


def normalize_source_arabic(text: str) -> str:
    """Light cleanup only; preserves legal wording/orthography for citation."""
    text = clean_unicode(text)
    text = normalize_arabic_parenthetical_numbers(text)
    text = re.sub(r"\s+([،؛:؟.!])", r"\1", text)
    return text.strip()


def normalize_arabic_for_search(text: str) -> str:
    """Aggressive normalization for retrieval/embeddings, not for quoting."""
    text = normalize_source_arabic(text)
    if araby is not None:
        text = araby.strip_tashkeel(text)
        text = araby.strip_tatweel(text)
    else:
        text = re.sub(r"[\u064b-\u065f\u0670\u06d6-\u06edـ]", "", text)

    # Alef/hamza variants + alef maqsura for search consistency.
    text = text.translate(
        str.maketrans(
            {
                "أ": "ا",
                "إ": "ا",
                "آ": "ا",
                "ٱ": "ا",
                "ؤ": "و",
                "ئ": "ي",
                "ى": "ي",
            }
        )
    )
    return clean_unicode(text)


def terminal_arabic_preview(text: str) -> str:
    """Visual-only shaping helper. Never store this reshaped form in the corpus."""
    if arabic_reshaper is None or get_display is None:
        return text
    return get_display(arabic_reshaper.reshape(text))


def normalize_heading_case(text: str) -> str:
    text = clean_unicode(text).strip(" :")
    if not text:
        return text
    letters = [c for c in text if c.isalpha()]
    mostly_upper = letters and sum(c.isupper() for c in letters) / len(letters) > 0.8
    if not mostly_upper:
        return text

    words = text.lower().split()
    out = []
    for i, word in enumerate(words):
        if i and word in STOPWORDS:
            out.append(word)
        else:
            out.append(word[:1].upper() + word[1:])
    return " ".join(out)


def normalize_topic(text: str) -> str:
    text = clean_unicode(text)
    text = re.sub(r"^\s*\d+\s*[.\-]\s*", "", text)
    return normalize_heading_case(text.rstrip(":"))


def is_bold_span(span: dict) -> bool:
    return "bold" in (span.get("font") or "").lower()


def extract_page_lines(page: fitz.Page, side: str, y_tol: float = 2.3) -> list[dict]:
    """Column-aware positional extraction.

    English lines are reconstructed x-left-to-right.
    Arabic lines are reconstructed x-right-to-left, with digit-run repair.
    """
    assert side in {"en", "ar"}
    left = side == "en"
    mid = page.rect.width / 2

    # Capture bold baselines so hierarchy headings can be filtered and parsed.
    bold_y: list[float] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            x0, y0, x1, _ = line["bbox"]
            cx = (x0 + x1) / 2
            if (left and cx >= mid) or ((not left) and cx < mid):
                continue
            spans = [s for s in line.get("spans", []) if (s.get("text") or "").strip()]
            if spans and all(is_bold_span(s) for s in spans):
                bold_y.append(y0)

    words = []
    for word in page.get_text("words", sort=False):
        x0, _, x1, _, _, *_ = word
        cx = (x0 + x1) / 2
        if (left and cx < mid) or ((not left) and cx >= mid):
            words.append(word)

    # Re-group split PDF text blocks using the visual baseline.
    words.sort(key=lambda w: (w[1], w[0]))
    groups: list[dict] = []
    for word in words:
        if groups and abs(word[1] - groups[-1]["mean_y"]) <= y_tol:
            groups[-1]["words"].append(word)
            ws = groups[-1]["words"]
            groups[-1]["mean_y"] = sum(w[1] for w in ws) / len(ws)
        else:
            groups.append({"mean_y": word[1], "words": [word]})

    lines: list[dict] = []
    for group in groups:
        ws = sorted(group["words"], key=lambda w: w[0], reverse=not left)
        tokens = []
        for word in ws:
            token = word[4]
            if side == "ar":
                token = fix_visual_arabic_digit_runs(token)
            tokens.append(token)

        text = clean_unicode(" ".join(tokens))
        if not text:
            continue
        y0 = min(w[1] for w in ws)
        y1 = max(w[3] for w in ws)
        lines.append(
            {
                "y0": y0,
                "y1": y1,
                "text": text,
                "bold": any(abs(y0 - by) <= 3.0 for by in bold_y),
            }
        )
    return lines


def pos(page_idx: int, y: float) -> tuple[int, float]:
    return (page_idx, float(y))


def parse_ar_article_number(text: str) -> int | None:
    match = AR_ARTICLE_RE.match(text)
    if not match:
        return None
    return int(match.group(1).translate(DIGIT_TRANS))


def is_repeal_text(text: str) -> bool:
    return bool(REPEAL_RE.search(text) or re.search(r"ألغيت|\bملغ(?:اة|ى|ي)?\b", text))


def gather_document(doc: fitz.Document):
    en_lines: list[dict] = []
    ar_lines: list[dict] = []
    en_candidates: dict[int, list[dict]] = defaultdict(list)
    ar_candidates: dict[int, list[dict]] = defaultdict(list)
    repeal_ranges: list[dict] = []

    for page_idx, page in enumerate(doc):
        for side, target in (("en", en_lines), ("ar", ar_lines)):
            for line in extract_page_lines(page, side):
                line.update(page_idx=page_idx, page=page_idx + 1, side=side)
                target.append(line)

                if side == "en":
                    match = EN_ARTICLE_RE.match(line["text"])
                    if match:
                        en_candidates[int(match.group(1))].append(line)
                    repeal = REPEAL_RE.search(line["text"])
                    if repeal:
                        repeal_ranges.append(
                            {
                                "start": int(repeal.group(1)),
                                "end": int(repeal.group(2)),
                                "page_idx": page_idx,
                                "page": page_idx + 1,
                                "y0": line["y0"],
                                "note_en": line["text"].lstrip("* ").strip(),
                            }
                        )
                else:
                    number = parse_ar_article_number(line["text"])
                    if number is not None:
                        ar_candidates[number].append(line)

    # Complete wrapped English repeal notes (e.g. "Presidential" / "Decree.").
    for rr in repeal_ranges:
        same_page = [
            l for l in en_lines
            if l["page_idx"] == rr["page_idx"] and rr["y0"] <= l["y0"] <= rr["y0"] + 35
        ]
        same_page.sort(key=lambda l: l["y0"])
        pieces = []
        for line in same_page:
            if pieces and line["bold"]:
                break
            if EN_ARTICLE_RE.match(line["text"]):
                continue
            pieces.append(line["text"].lstrip("* ").strip())
            if pieces and pieces[-1].endswith("."):
                break
        if pieces:
            rr["note_en"] = clean_unicode(" ".join(pieces))

    return en_lines, ar_lines, en_candidates, ar_candidates, repeal_ranges


def choose_anchors(en_candidates: dict, ar_candidates: dict) -> dict[int, dict]:
    """Prefer bilingual-aligned article anchors; fall back to one column when needed."""
    anchors: dict[int, dict] = {}
    for number in sorted(set(en_candidates) | set(ar_candidates)):
        ens = en_candidates.get(number, [])
        ars = ar_candidates.get(number, [])

        pairs = []
        for en in ens:
            for ar in ars:
                if en["page_idx"] == ar["page_idx"] and abs(en["y0"] - ar["y0"]) <= 10:
                    pairs.append((abs(en["y0"] - ar["y0"]), en, ar))

        if pairs:
            _, en, ar = min(pairs, key=lambda p: p[0])
            anchors[number] = {
                "n": number,
                "page_idx": en["page_idx"],
                "page": en["page"],
                "y0": min(en["y0"], ar["y0"]),
                "score": 2,
            }
        elif ens:
            en = min(ens, key=lambda l: (l["page_idx"], l["y0"]))
            anchors[number] = {
                "n": number,
                "page_idx": en["page_idx"],
                "page": en["page"],
                "y0": en["y0"],
                "score": 1,
            }
        else:
            ar = min(ars, key=lambda l: (l["page_idx"], l["y0"]))
            anchors[number] = {
                "n": number,
                "page_idx": ar["page_idx"],
                "page": ar["page"],
                "y0": ar["y0"],
                "score": 1,
            }
    return anchors


def build_heading_events(en_lines: list[dict]) -> list[tuple]:
    """Parse hierarchy from bold English rows and carry it forward."""
    lines = sorted(en_lines, key=lambda l: pos(l["page_idx"], l["y0"]))
    events: list[tuple] = []
    pending: str | None = None

    part_re = re.compile(r"^(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH)\s+PART\b\s*(.*)$", re.I)
    book_re = re.compile(r"^BOOK\s+[IVXLCDM]+\b\s*(.*)$", re.I)
    chapter_re = re.compile(r"^(?:CHAPTER\s+[IVXLCDM]+|PRELIMINARY CHAPTER)\b\s*(.*)$", re.I)
    section_re = re.compile(r"^SECTION\s+[IVXLCDM]+\b\s*(.*)$", re.I)

    hierarchy_patterns = (
        ("part", part_re),
        ("source_book", book_re),
        ("chapter", chapter_re),
        ("section", section_re),
    )

    for line in lines:
        text = line["text"].strip()
        if not line["bold"] or EN_ARTICLE_RE.match(text) or is_repeal_text(text):
            continue

        if pending is not None:
            is_new_marker = any(pattern.match(text) for _, pattern in hierarchy_patterns)
            if not is_new_marker:
                events.append((pos(line["page_idx"], line["y0"]), pending, normalize_heading_case(text)))
                pending = None
                continue
            pending = None

        matched = False
        for key, pattern in hierarchy_patterns:
            match = pattern.match(text)
            if not match:
                continue
            # The title is normally the next bold row in this PDF.
            rest = match.groups()[-1].strip(" :-") if match.groups() else ""
            if rest:
                events.append((pos(line["page_idx"], line["y0"]), key, normalize_heading_case(rest)))
            else:
                pending = key
            matched = True
            break
        if matched:
            continue

        # Other short bold rows are topical labels: Consent, Object, Nullity,
        # 2. The Effects of a Contract, Elements of Sale, etc.
        if len(text) <= 120:
            events.append((pos(line["page_idx"], line["y0"]), "topic", normalize_topic(text)))

    return sorted(events, key=lambda e: e[0])


def context_at(events: list[tuple], target_pos: tuple[int, float]) -> dict:
    state = {"part": None, "source_book": None, "chapter": None, "section": None, "topic": None}
    for event_pos, key, value in events:
        if event_pos >= target_pos:
            break
        if key == "part":
            state.update(part=value, source_book=None, chapter=None, section=None, topic=None)
        elif key == "source_book":
            state.update(source_book=value, chapter=None, section=None, topic=None)
        elif key == "chapter":
            state.update(chapter=value, section=None, topic=None)
        elif key == "section":
            state.update(section=value, topic=None)
        elif key == "topic":
            state["topic"] = value
    return state


def build_contexts(anchors: dict[int, dict], events: list[tuple]) -> dict[int, dict]:
    return {
        number: context_at(events, pos(anchor["page_idx"], anchor["y0"]))
        for number, anchor in anchors.items()
    }


def extract_article_text(lines: list[dict], start: tuple, end: tuple, side: str) -> str:
    chunks: list[str] = []
    for line in lines:
        line_pos = pos(line["page_idx"], line["y0"])
        if not (start < line_pos < end):
            continue
        text = line["text"]
        if EN_ARTICLE_RE.match(text) or AR_ARTICLE_RE.match(text):
            continue
        if is_repeal_text(text):
            continue
        if line["bold"]:  # hierarchy/topic heading
            continue
        chunks.append(text)

    joined = clean_unicode(" ".join(chunks))
    return normalize_source_arabic(joined) if side == "ar" else joined


def build_records(
    doc: fitz.Document,
    en_lines: list[dict],
    ar_lines: list[dict],
    anchors: dict[int, dict],
    repeal_ranges: list[dict],
    contexts: dict[int, dict],
    events: list[tuple],
) -> list[dict]:
    physical_anchors = sorted(anchors.values(), key=lambda a: pos(a["page_idx"], a["y0"]))
    next_position: dict[int, tuple] = {}
    for i, anchor in enumerate(physical_anchors):
        if i + 1 < len(physical_anchors):
            nxt = physical_anchors[i + 1]
            next_position[anchor["n"]] = pos(nxt["page_idx"], nxt["y0"])
        else:
            next_position[anchor["n"]] = (len(doc), 10**9)

    repealed_by_number: dict[int, dict] = {}
    for rr in repeal_ranges:
        rr["context"] = context_at(events, pos(rr["page_idx"], rr["y0"]))
        for number in range(rr["start"], rr["end"] + 1):
            repealed_by_number[number] = rr

    max_number = max(
        max(anchors),
        max((rr["end"] for rr in repeal_ranges), default=0),
    )

    records: list[dict] = []
    for number in range(1, max_number + 1):
        anchor = anchors.get(number)
        repeal = repealed_by_number.get(number)

        if anchor is not None:
            start = pos(anchor["page_idx"], anchor["y0"])
            end = next_position[number]
            text_en = extract_article_text(en_lines, start, end, "en")
            ar_text = extract_article_text(ar_lines, start, end, "ar")
            source_page = anchor["page"]
            context = contexts.get(number, {})
        elif repeal is not None:
            # No operative article row exists in the source for this repealed number.
            text_en = ""
            ar_text = ""
            source_page = repeal["page"]
            context = repeal["context"]
        else:
            raise RuntimeError(
                f"Unexplained missing Article {number}. Refusing to silently omit it."
            )

        is_repealed = repeal is not None
        if is_repealed:
            # Do not misrepresent the repeal notice as the operative article text.
            text_en = ""
            ar_text = ""

        qa_flags: list[str] = []
        if anchor is not None and anchor["score"] == 1:
            qa_flags.append("single_column_article_anchor")
        if not is_repealed and not ar_text:
            qa_flags.append("missing_ar_text")
        if not is_repealed and not text_en:
            qa_flags.append("missing_en_text")

        records.append(
            {
                "article_number": number,
                # Match the requested schema: PART title is used as top-level book.
                "book": context.get("part") or context.get("source_book"),
                "chapter": context.get("chapter"),
                "section": context.get("section"),
                "topic": context.get("topic"),
                "ar_text": ar_text,
                # Keep normalized text separate so citations remain source-faithful.
                "ar_text_normalized": normalize_arabic_for_search(ar_text),
                "text_en": text_en,
                "is_repealed": is_repealed,
                "repeal_note": repeal["note_en"] if repeal else None,
                # Literal BOOK I/II/... title from the PDF; optional ingestion metadata.
                "source_book": context.get("source_book"),
                # 1-based physical PDF page in this file.
                "source_page": source_page,
                "citation": f"Egyptian Civil Code, Article {number}",
                "qa_flags": qa_flags,
            }
        )

    return records


def write_jsonl(records: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_qa_html(
    doc: fitz.Document,
    records: list[dict],
    anchors: dict[int, dict],
    qa_dir: Path,
    count: int,
    seed: int,
) -> Path:
    """Render 20 random non-repealed article rows and show extraction beside them."""
    qa_dir.mkdir(parents=True, exist_ok=True)
    img_dir = qa_dir / "crops"
    img_dir.mkdir(exist_ok=True)

    candidates = [r for r in records if not r["is_repealed"] and r["article_number"] in anchors]
    rng = random.Random(seed)
    sample = rng.sample(candidates, min(count, len(candidates)))

    physical = sorted(anchors.values(), key=lambda a: pos(a["page_idx"], a["y0"]))
    next_by_n = {}
    for i, anchor in enumerate(physical):
        next_by_n[anchor["n"]] = physical[i + 1] if i + 1 < len(physical) else None

    cards = []
    for record in sample:
        number = record["article_number"]
        anchor = anchors[number]
        page = doc[anchor["page_idx"]]
        nxt = next_by_n[number]
        y0 = max(0, anchor["y0"] - 8)
        if nxt and nxt["page_idx"] == anchor["page_idx"]:
            y1 = min(page.rect.height, nxt["y0"] - 2)
        else:
            y1 = page.rect.height
        if y1 <= y0 + 20:
            y1 = min(page.rect.height, y0 + 180)

        clip = fitz.Rect(0, y0, page.rect.width, y1)
        pix = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), clip=clip, alpha=False)
        img_name = f"article_{number}_page_{record['source_page']}.png"
        pix.save(str(img_dir / img_name))

        cards.append(
            f"""
            <section>
              <h2>Article {number} - PDF page {record['source_page']}</h2>
              <img src="crops/{html.escape(img_name)}" alt="source crop">
              <div class="grid">
                <div><h3>Arabic extracted</h3><p dir="rtl">{html.escape(record['ar_text'])}</p></div>
                <div><h3>English extracted</h3><p>{html.escape(record['text_en'])}</p></div>
              </div>
              <p><b>Metadata:</b> {html.escape(str({k: record[k] for k in ['book','chapter','section','topic','qa_flags']}))}</p>
            </section>
            """
        )

    html_text = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Extraction QA</title>
<style>
body{{font-family:Arial,sans-serif;margin:30px;max-width:1300px}}
section{{border-top:2px solid #ddd;padding:20px 0}}
img{{max-width:100%;border:1px solid #bbb}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}
p{{line-height:1.55}}
</style></head><body>
<h1>Random extraction QA ({len(sample)} articles, seed={seed})</h1>
{''.join(cards)}
</body></html>"""
    output = qa_dir / "qa_random_articles.html"
    output.write_text(html_text, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--out", type=Path, default=Path("egyptian_civil_code_articles.jsonl"))
    parser.add_argument("--qa-dir", type=Path, default=Path("qa_civil_code"))
    parser.add_argument("--qa-count", type=int, default=20)
    parser.add_argument("--qa-seed", type=int, default=42)
    parser.add_argument("--no-qa", action="store_true")
    args = parser.parse_args()

    doc = fitz.open(args.pdf)
    en_lines, ar_lines, en_candidates, ar_candidates, repeal_ranges = gather_document(doc)
    anchors = choose_anchors(en_candidates, ar_candidates)
    events = build_heading_events(en_lines)
    contexts = build_contexts(anchors, events)
    records = build_records(doc, en_lines, ar_lines, anchors, repeal_ranges, contexts, events)
    write_jsonl(records, args.out)

    flagged = [r for r in records if r["qa_flags"]]
    repealed = [r for r in records if r["is_repealed"]]

    print(f"Wrote {len(records)} articles -> {args.out}")
    print(f"Repealed records: {len(repealed)}")
    print(f"QA-flagged records: {len(flagged)}")

    # Explicit sanity check for the user's example article.
    if len(records) >= 147:
        r147 = records[146]
        print("\nArticle 147 sanity check:")
        print(json.dumps({k: r147[k] for k in [
            "article_number", "book", "chapter", "section", "topic",
            "is_repealed", "source_page", "citation", "qa_flags"
        ]}, ensure_ascii=False, indent=2))
        print("Arabic preview:", terminal_arabic_preview(r147["ar_text"][:240]))
        print("English preview:", r147["text_en"][:240])

    if not args.no_qa:
        qa_html = write_qa_html(doc, records, anchors, args.qa_dir, args.qa_count, args.qa_seed)
        print(f"QA HTML -> {qa_html}")


if __name__ == "__main__":
    main()
