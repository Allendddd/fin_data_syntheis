"""Read original documents while retaining page/paragraph provenance."""

from __future__ import annotations

from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
from pypdf.errors import PyPdfError


class HTMLText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.hidden, self.published_at = [], 0, None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag in {"script", "style"}:
            self.hidden += 1
        if tag in {"p", "div", "br", "tr", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")
        if tag == "meta" and attributes.get("property", attributes.get("name", "")) in {"article:published_time", "datePublished", "pubdate"}:
            self.published_at = attributes.get("content")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def document_units(data: bytes, suffix: str) -> list[dict]:
    try:
        return _document_units(data, suffix)
    except (PyPdfError, zipfile.BadZipFile, ET.ParseError, UnicodeError) as exc:
        raise ValueError(f"cannot parse original document: {exc}") from exc


def _document_units(data: bytes, suffix: str) -> list[dict]:
    """Each unit corresponds to the original file, never to synthesized facts."""
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(data))
        units = [{"locator": {"kind": "pdf_page", "page": n + 1}, "text": page.extract_text() or ""}
                 for n, page in enumerate(reader.pages)]
    elif suffix == ".docx":
        with zipfile.ZipFile(BytesIO(data)) as archive:
            info = archive.getinfo("word/document.xml")
            if info.file_size > 64 * 1024 * 1024:
                raise ValueError("DOCX document XML exceeds size limit")
            root = ET.fromstring(archive.read(info))
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        body = root.find("w:body", ns)
        units = []
        for block in body if body is not None else []:
            paragraphs = [block] if block.tag.endswith("}p") else block.findall(".//w:p", ns)
            text = "\n".join("".join(node.text or "" for node in p.findall(".//w:t", ns)) for p in paragraphs)
            units.append({"locator": {"kind": "docx_block", "block": len(units) + 1,
                                      "table": block.tag.endswith("}tbl")}, "text": text})
    elif suffix in {".html", ".htm"}:
        parser = HTMLText()
        parser.feed(data.decode("utf-8", errors="replace"))
        units = [{"locator": {"kind": "html_document"}, "text": "".join(parser.parts)}]
    else:
        units = [{"locator": {"kind": "text_document"}, "text": data.decode("utf-8-sig")}]
    if not any(unit["text"].strip() for unit in units):
        raise ValueError("document has no extractable text; scanned PDFs require an OCR adapter")
    return units


SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".csv", ".tsv", ".json", ".jsonl", ".xml", ".html", ".htm"}
