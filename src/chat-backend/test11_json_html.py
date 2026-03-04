from __future__ import annotations

import json
import mimetypes
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List
from pypdf import PdfReader
from bs4 import BeautifulSoup
import logging


# ---------------- Domain ----------------
@dataclass(frozen=True)
class RagDocument:
    doc_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------- Loader ----------------
@dataclass(frozen=True)
class LoadedItem:
    uri: str
    name: str
    media_type: str | None
    data: bytes


class LocalLoader:
    
    def process_directory(self, source: str):
        """Iterates through a directory and routes files to the correct processor."""
        path_obj = Path(source)

        # path string doesn't exist
        if not path_obj.exists():
            return
        
        # path string is a file
        if path_obj.is_file():
            logging.debug("The source sring is a file")
            self.route_file(path_obj)
        
        # path string is a directory
        if path_obj.is_dir():
            logging.debug("The source is a directory")
            # .rglob("*") searches the folder and all sub-folders
            for file_path in path_obj.rglob("*"):
                if file_path.is_file():
                    self.route_file(path_obj)

    
    def route_file(self, file_path: Path):
        """Checks the file extension and sends it to the correct processor."""
        extension = file_path.suffix.lower()
        
        if extension == '.json':
            return self.process_json(file_path)
        elif extension == '.pdf':
            return self.process_pdf(file_path)
        else:
            logging.debug(f"Skipping unsupported file type: {file_path.name}")
            return None
    
    def process_json(self, file_path: Path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data
        except Exception as e:
            return
    
    def process_pdf(file_path: Path):
        """Reads and extracts text from a PDF file."""
        try:
            reader = PdfReader(file_path)
            text_content = ""
            
            # Extract text from all pages
            for page in reader.pages:
                text_content += page.extract_text() + "\n"
                
            # Work further with 'text_content' here...
            logging.info(f"Successfully read PDF: {file_path.name}. Extracted {len(text_content)} characters.")
            return text_content
        except Exception as e:
            logging.error(f"Failed to read PDF {file_path.name}: {e}")


    # def load(self, source: str) -> Iterable[LoadedItem]:
    #     root = Path(source)
    #     files = [root] if root.is_file() else list(root.rglob("*"))
    #     for f in files:
    #         if not f.is_file():
    #             continue
    #         mt, _ = mimetypes.guess_type(str(f))

    #         doc = LoadedItem(
    #             uri=f"file://{f.resolve()}",
    #             name=f.name,
    #             media_type=mt,
    #             data=f.read_bytes(),
    #         )
    #         yield doc




# ---------------- Utils ----------------
def html_to_text(s: str) -> str:
    soup = BeautifulSoup(s or "", "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    return " ".join(soup.get_text(separator=" ").split())


def clean_text(text: str) -> str:
    text = re.sub(r"Nur zur internen Verwendung", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------- Parser ----------------
class FaqJsonParser:
    def __init__(self, embed_mode: str = "question_only"):
        self.embed_mode = embed_mode  # question_only | qa

    def supports(self, item: LoadedItem) -> bool:
        mt = (item.media_type or "").lower()
        supports: bool = item.name.lower().endswith(".json") or "json" in mt
        return supports

    def parse(self, item: LoadedItem):
        obj: Any = json.loads(item.data.decode("utf-8", errors="ignore") or "null")

        # Accept either list[...] or dict{items:[...]} shapes
        records = None
        if isinstance(obj, list):
            records = obj
        elif isinstance(obj, dict):
            # common keys
            for k in ("items", "data", "faqs", "results"):
                v = obj.get(k)
                if isinstance(v, list):
                    records = v
                    break

        if not records or not isinstance(records, list) or not records or not isinstance(records[0], dict):
            return

        # Find question/answer keys (sometimes capitalized)
        first = records[0]
        q_key = "question" if "question" in first else "Question" if "Question" in first else None
        a_key = "answer" if "answer" in first else "Answer" if "Answer" in first else None
        if not q_key or not a_key:
            return

        for rec in records:
            faq_id = rec.get("id") or rec.get("Id") or rec.get("faq_id")
            q = html_to_text(str(rec.get(q_key, "")))
            a = html_to_text(str(rec.get(a_key, "")))

            index_text = f"{q}\n{a}" if self.embed_mode == "qa" else q
            md = {
                "type": "faq",
                "faq_id": faq_id,
                "category": rec.get("category"),
                "updated_at": rec.get("updated"),
                "source_url": rec.get("link") or rec.get("url"),
                "prompt_text": f"FAQ\nFrage: {q}\nAntwort: {a}",
            }
            yield index_text, md


# ---------------- Provider ----------------
class RagDocumentProvider:
    def __init__(self, loader: LocalLoader, faq_parser: FaqJsonParser):
        self.loader = loader
        self.faq_parser = faq_parser

    def load_documents(self, source: str) -> List[RagDocument]:
        if not self.loader.supports(source):
            raise ValueError(f"Unsupported source: {source}")

        docs: List[RagDocument] = []

        for item in self.loader.process_directory(source):
            if self.faq_parser.supports(item):
                parser = self.faq_parser.parse(item)
                for text, md in parser:
                    faq_id = md.get("faq_id")
                    doc_id = f"faq:{faq_id}" if faq_id is not None else f"faq:unknown:{item.name}"

                    docs.append(
                        RagDocument(
                            doc_id=doc_id,
                            content=clean_text(text),
                            metadata={
                                "source_uri": item.uri,
                                "source_name": item.name,
                                "media_type": item.media_type,
                                **md,
                            },
                        )
                    )
        return docs


def main():
    source = "./data"
    #source = sys.argv[1] if len(sys.argv) > 1 else "./data"
    loader = LocalLoader()
    parser = FaqJsonParser(embed_mode="question_only")  # or "qa"
    provider = RagDocumentProvider(loader, parser)

    docs = provider.load_documents(source)
    print(f"Loaded {len(docs)} RagDocuments from: {source}")

    # show first 3
    for i, d in enumerate(docs[:3], start=1):
        print("\n" + "=" * 80)
        print(f"[{i}] doc_id={d.doc_id} category={d.metadata.get('category')} updated={d.metadata.get('updated_at')}")
        print(f"source_url={d.metadata.get('source_url')}")
        print("content preview:", d.content[:250], "...")


if __name__ == "__main__":
    main()