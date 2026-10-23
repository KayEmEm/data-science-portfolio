import pytest

from filings_qa.config import ChunkSettings
from filings_qa.data import SchemaError, documents_in_scope, normalize_documents, normalize_questions
from filings_qa.parse import chunk_page, clean_text, read_pdf


def pdf_with_pages(texts: list[str]) -> bytes:
    """A valid PDF, with a cross reference table, that shows one line of text per page."""
    n = len(texts)
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    font = 3 + 2 * n
    objects = ["<</Type/Catalog/Pages 2 0 R>>", f"<</Type/Pages/Kids[{kids}]/Count {n}>>"]
    for i, text in enumerate(texts):
        stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode()
        objects.append(
            f"<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 200]/Contents {4 + 2 * i} 0 R"
            f"/Resources<</Font<</F1 {font} 0 R>>>>>>"
        )
        objects.append(f"<</Length {len(stream)}>>stream\n{stream.decode()}\nendstream")
    objects.append("<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>")
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj{body}endobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer<</Size {len(objects) + 1}/Root 1 0 R>>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def test_read_pdf_returns_one_text_per_page_in_order(tmp_path):
    path = tmp_path / "two_pages.pdf"
    path.write_bytes(pdf_with_pages(["Revenue page", "Capex page"]))
    assert read_pdf(path) == ["Revenue page", "Capex page"]


def test_clean_text_collapses_spaces_and_blank_lines():
    assert clean_text("a  b  c\r\n\n\n\nd ") == "a b c\n\nd"


@pytest.mark.parametrize("n_words", [0, 1, 179, 180, 181, 400, 1000])
def test_chunks_cover_every_word_with_the_set_overlap(n_words):
    settings = ChunkSettings(words=180, overlap=40)
    words = [f"w{i}" for i in range(n_words)]
    chunks = chunk_page(" ".join(words), settings)
    seen = [w for c in chunks for w in c.split()]
    assert set(seen) == set(words)
    assert all(len(c.split()) <= settings.words for c in chunks)
    for left, right in zip(chunks, chunks[1:], strict=False):
        assert left.split()[-settings.overlap :] == right.split()[: settings.overlap]


def _question(**evidence):
    return {
        "financebench_id": "q1",
        "company": "Alder",
        "doc_name": "ALDER_2022_10K",
        "question_type": "metrics-generated",
        "question_reasoning": None,
        "question": "What?",
        "answer": 12,
        "evidence": [{"evidence_text": "x", "evidence_page_num": 3, **evidence}],
    }


@pytest.mark.parametrize("field", ["doc_name", "evidence_doc_name"])
def test_both_spellings_of_the_evidence_document_field_are_read(field):
    frame = normalize_questions([_question(**{field: "ALDER_2022_10K"})])
    assert frame.loc[0, "evidence_pages"] == ["ALDER_2022_10K#3"]
    assert frame.loc[0, "answer"] == "12"


def test_question_without_evidence_is_rejected():
    record = _question(doc_name="ALDER_2022_10K")
    record["evidence"] = []
    with pytest.raises(SchemaError):
        normalize_questions([record])


def _doc(name, link, period=2022):
    return {
        "doc_name": name,
        "company": "A",
        "doc_type": "10k",
        "doc_period": period,
        "doc_link": link,
        "gics_sector": "X",
    }


def test_repeated_document_with_the_same_link_is_kept_once():
    frame = normalize_documents([_doc("A_2022_10K", "u"), _doc("A_2022_10K", "u", 2021)])
    assert len(frame) == 1 and frame.loc[0, "doc_period"] == 2022 and frame.loc[0, "doc_type"] == "10K"


def test_repeated_document_with_two_links_is_rejected():
    with pytest.raises(SchemaError):
        normalize_documents([_doc("A_2022_10K", "u"), _doc("A_2022_10K", "v")])


def test_documents_in_scope_rejects_unknown_documents():
    questions = normalize_questions([_question(doc_name="ALDER_2022_10K")])
    with pytest.raises(SchemaError):
        documents_in_scope(questions, normalize_documents([_doc("OTHER_2022_10K", "u")]))
