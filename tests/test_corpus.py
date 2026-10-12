"""Corpus integrity gate, including documented exceptions in the source PDF.

These checks validate structure and known regressions, not full legal accuracy.
The 20-article visual review remains necessary.
"""

import re

import pytest


EXPECTED_REPEALED = set(range(54, 81)) | set(range(389, 418))
MISSING_ARABIC_IN_SOURCE = {1022}
# Guard against accidentally merging several provisions into one record.
MAX_ARTICLE_CHARACTERS = 10_000


def test_article_numbers_are_unique_contiguous_integers(articles):
    numbers = [article["article_number"] for article in articles]
    assert all(type(number) is int for number in numbers)
    assert numbers == list(range(1, 1150))


@pytest.mark.parametrize("field", ["ar_text", "ar_text_normalized", "text_en"])
def test_active_article_text_is_present(articles, field):
    missing = [
        article["article_number"] for article in articles
        if not article["is_repealed"]
        and not (field != "text_en" and article["article_number"] in MISSING_ARABIC_IN_SOURCE)
        and not article[field].strip()
    ]
    assert not missing, f"Active articles with empty {field}: {missing}"


def test_repealed_articles_are_preserved_with_notices(articles):
    repealed = {article["article_number"] for article in articles if article["is_repealed"]}
    assert repealed == EXPECTED_REPEALED
    for article in articles:
        assert type(article["is_repealed"]) is bool
        if article["is_repealed"]:
            assert article["repeal_note"].strip()
            assert not article["ar_text"] and not article["text_en"]


@pytest.mark.parametrize("field", ["ar_text", "ar_text_normalized", "text_en"])
def test_article_lengths_are_bounded(articles, field):
    oversized = [
        article["article_number"] for article in articles
        if len(article[field]) > MAX_ARTICLE_CHARACTERS
    ]
    assert not oversized, f"Possible merged articles in {field}: {oversized}"


def test_citations_and_metadata(articles):
    for article in articles:
        number = article["article_number"]
        assert article["citation"] == f"Egyptian Civil Code, Article {number}"
        assert type(article["source_page"]) is int and article["source_page"] > 0
        for field in ("book", "chapter", "section", "topic", "source_book"):
            assert article[field] is None or (
                isinstance(article[field], str) and article[field].strip()
            )
        assert "missing_ar_text" not in article["qa_flags"], f"Article {number}"
        assert "missing_en_text" not in article["qa_flags"], f"Article {number}"


def test_bold_arabic_body_of_article_558_is_retained(by_number):
    article = by_number[558]
    assert "عقد يلتزم المؤجر" in article["ar_text"]
    assert "أجر معلوم" in article["ar_text"]
    assert "الفصل" not in article["ar_text"]
    assert "ثالث سنوات" not in article["ar_text"]  # Article 559's body
    assert not article["is_repealed"]


def test_article_1022_missing_arabic_is_an_explicit_source_exception(by_number):
    article = by_number[1022]
    assert not article["is_repealed"]
    assert article["ar_text"] == article["ar_text_normalized"] == ""
    assert article["text_en"].strip()
    assert article["source_page"] == 147
    assert "missing_ar_text_in_source" in article["qa_flags"]


def test_article_1021_boundary_with_1022_matches_the_source_rows(by_number):
    """Page 147 prints Arabic paragraphs (1)-(3) in the 1021 row; (2) and (3) are the Arabic of 1022's
    English (see reports/corpus-extraction.md). The corpus keeps the literal row structure."""
    article, following = by_number[1021], by_number[1022]
    assert article["source_page"] == following["source_page"] == 147
    assert not article["is_repealed"] and article["qa_flags"] == []
    arabic = article["ar_text"]
    assert arabic.startswith("(١) لا يلزم مالك العقار المرتفق به")
    assert arabic.index("(٢)") < arabic.index("(٣)") and "(٤)" not in arabic
    # 1021's English is only the first paragraph; nothing of 1022's English leaked into it.
    assert article["text_en"].startswith("In the absence of an agreement to the contrary, the owner of the servient tenement")
    assert article["text_en"].endswith("necessitated by the normal use of the servitude.")
    assert "cost" not in article["text_en"] and "abandoning" not in article["text_en"]
    # 1022 starts with its own first sentence and has no Arabic of its own (the exception above).
    assert following["text_en"].startswith("In the absence o f an agreement to the contrary, the cost of the necessary works")
    assert "abandoning the servient tenement" in following["text_en"]
    assert following["ar_text"] == ""
    # Neighbours stay separate.
    assert by_number[1020]["article_number"] == 1020 and by_number[1023]["article_number"] == 1023
    assert by_number[1020]["text_en"] not in article["text_en"]


def test_article_147_text_and_hierarchy(by_number):
    article = by_number[147]
    assert "العقد شريعة المتعاقدين" in article["ar_text"]
    assert "contract makes the law" in article["text_en"]
    assert article["section"] == "Contracts"
    assert article["topic"] == "The Effects of a Contract"


def test_zero_width_alef_lam_order_is_corrected_in_arabic_text(by_number):
    article = by_number[2]
    assert "لا يجوز إلغاء" in article["ar_text"]
    assert "ال يجوز إلغاء" not in article["ar_text"]

    # Here the PDF splits ال across touching text spans and reverses the
    # zero-width alef/lam pair inside the following word.
    article_with_split_word = by_number[3]
    assert "بالتقويم الميلادي" in article_with_split_word["ar_text"]
    assert "ال ميالدي" not in article_with_split_word["ar_text"]

    # Reordering must target the flagged glyph's own offset, not another
    # ordinary ال occurrence earlier in the same word.
    assert "مبادئ الشريعة الإسلامية" in by_number[1]["ar_text"]
    assert "الشريعة لاإسلامية" not in by_number[1]["ar_text"]


def test_normalized_arabic_has_no_diacritics_or_tatweel(articles):
    for article in articles:
        assert not re.search(r"[\u064b-\u065f\u0670ـ]", article["ar_text_normalized"])
