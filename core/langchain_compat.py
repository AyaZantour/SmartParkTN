"""
Compatibility shim: PaddleX (bundled with PaddleOCR >=2.8) imports several
langchain submodules that were moved/removed in langchain >=0.2.

Import this module BEFORE importing paddleocr.
"""
import sys
import types


def _ensure_langchain_base():
    """Return (or create) the top-level 'langchain' module stub."""
    if "langchain" not in sys.modules:
        sys.modules["langchain"] = types.ModuleType("langchain")
    return sys.modules["langchain"]


def _stub_module(full_name, attrs=None):
    """Register a stub module under full_name if not already present."""
    if full_name in sys.modules:
        return sys.modules[full_name]
    mod = types.ModuleType(full_name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[full_name] = mod
    # Attach as attribute on parent
    parts = full_name.split(".")
    if len(parts) > 1:
        parent_name = ".".join(parts[:-1])
        parent = sys.modules.get(parent_name)
        if parent:
            setattr(parent, parts[-1], mod)
    return mod


def apply():
    """Patch all langchain submodules that PaddleX tries to import."""
    lc = _ensure_langchain_base()

    # ── langchain.docstore ────────────────────────────────────────────────────
    if "langchain.docstore" not in sys.modules:
        try:
            import langchain_community.docstore as _lc_ds          # noqa
            import langchain_community.docstore.document as _lc_dd  # noqa
            sys.modules["langchain.docstore"] = _lc_ds
            sys.modules["langchain.docstore.document"] = _lc_dd
            setattr(lc, "docstore", _lc_ds)
        except ImportError:
            class Document:
                def __init__(self, page_content="", metadata=None):
                    self.page_content = page_content
                    self.metadata = metadata or {}

            ds  = _stub_module("langchain.docstore", {"Document": Document})
            dsd = _stub_module("langchain.docstore.document", {"Document": Document})
            ds.document = dsd
            setattr(lc, "docstore", ds)

    # ── langchain.text_splitter ───────────────────────────────────────────────
    if "langchain.text_splitter" not in sys.modules:
        # Do NOT import langchain_text_splitters – it pulls nltk→scipy→numpy1.x
        # which crashes on NumPy 2.x. Use a minimal stub instead.
        class RecursiveCharacterTextSplitter:
            def __init__(self, **kw): pass
            def split_text(self, text): return [text]
            def split_documents(self, docs): return docs

        class CharacterTextSplitter(RecursiveCharacterTextSplitter):
            pass

        ts = _stub_module("langchain.text_splitter", {
            "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter,
            "CharacterTextSplitter": CharacterTextSplitter,
        })
        setattr(lc, "text_splitter", ts)

    # ── any other stubs PaddleX may need ─────────────────────────────────────
    for sub in ("langchain.schema", "langchain.embeddings", "langchain.vectorstores",
                "langchain.chains", "langchain.llms", "langchain.prompts"):
        _stub_module(sub)
        setattr(lc, sub.split(".")[-1], sys.modules[sub])
