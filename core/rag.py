"""
SmartParkTN – RAG Assistant
Uses ChromaDB + sentence-transformers for retrieval
and Groq (free API) for generation (Llama-3.1-8B-Instant).

Get free Groq API key: https://console.groq.com/keys
"""
from __future__ import annotations
import os, glob
from typing import List
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

_GROQ_KEY   = os.getenv("GROQ_API_KEY", "")
_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
_CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./data/chroma_db")
_RULES_DIR  = os.getenv("RULES_DIR", "./data/rules")
_EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # 45 MB, CPU-friendly

SYSTEM_PROMPT = """Tu es l'assistant IA du parking SmartParkTN.
Tu réponds aux questions du personnel de parking concernant:
- Les règles d'accès et les tarifs
- Les catégories de véhicules (visiteur, abonné, VIP, liste noire, employé, urgence)
- Les procédures en cas de litige ou d'incident
- Les horaires autorisés et les zones de stationnement
- Le calcul des montants et durées

Réponds toujours en français, de façon concise et précise.
Base tes réponses UNIQUEMENT sur le contexte fourni.
Si tu ne trouves pas l'information, dis-le clairement.
"""


class ParkingAssistant:
    def __init__(self):
        self._collection = None
        self._embed_fn   = None
        self._groq_client = None
        self._init()

    def _init(self):
        self._init_embedder()
        self._init_chroma()
        self._init_groq()

    def _init_embedder(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(_EMBED_MODEL)
            logger.info(f"Embedder loaded: {_EMBED_MODEL}")
        except Exception as e:
            logger.error(f"Embedder init failed: {e}")
            self._model = None

    def _init_chroma(self):
        try:
            import chromadb
            client = chromadb.PersistentClient(path=_CHROMA_DIR)
            self._collection = client.get_or_create_collection(
                name="parking_rules",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"ChromaDB ready: {self._collection.count()} chunks")
        except Exception as e:
            logger.error(f"ChromaDB init failed: {e}")
            self._collection = None

    def _init_groq(self):
        if not _GROQ_KEY or _GROQ_KEY.startswith("gsk_XXX"):
            logger.warning("No valid GROQ_API_KEY – assistant in offline mode")
            return
        try:
            from groq import Groq
            self._groq_client = Groq(api_key=_GROQ_KEY)
            logger.info(f"Groq client ready (model: {_GROQ_MODEL})")
        except Exception as e:
            logger.error(f"Groq init failed: {e}")

    # ── Document ingestion ─────────────────────────────────────────────
    def ingest_documents(self, rules_dir: str = _RULES_DIR):
        """Load all .md and .txt files from rules_dir into ChromaDB."""
        if not self._collection or not self._model:
            logger.error("ChromaDB or embedder not ready – cannot ingest")
            return

        files = glob.glob(os.path.join(rules_dir, "*.md")) + \
                glob.glob(os.path.join(rules_dir, "*.txt"))
        if not files:
            logger.warning(f"No rule files found in {rules_dir}")
            return

        for path in files:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            chunks = self._chunk(text)
            ids = [f"{os.path.basename(path)}::{i}" for i in range(len(chunks))]
            embeddings = self._model.encode(chunks, show_progress_bar=False).tolist()
            self._collection.upsert(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=[{"source": os.path.basename(path)}] * len(chunks),
            )
        logger.info(f"Ingested {len(files)} file(s) → {self._collection.count()} total chunks")

    @staticmethod
    def _chunk(text: str, size: int = 400, overlap: int = 80) -> List[str]:
        words = text.split()
        chunks, i = [], 0
        while i < len(words):
            chunk = " ".join(words[i:i + size])
            chunks.append(chunk)
            i += size - overlap
        return chunks or [text]

    # ── Query ─────────────────────────────────────────────────────────
    def query(self, question: str, n_results: int = 5) -> str:
        context = self._retrieve(question, n_results)
        return self._generate(question, context)

    def _retrieve(self, question: str, n: int) -> str:
        if not self._collection or not self._model:
            return "Base de connaissances non disponible."
        try:
            emb = self._model.encode([question]).tolist()
            res = self._collection.query(query_embeddings=emb, n_results=n)
            docs = res.get("documents", [[]])[0]
            return "\n\n---\n\n".join(docs) if docs else "Aucun document pertinent trouvé."
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return "Erreur de recherche."

    def _generate(self, question: str, context: str) -> str:
        if not self._groq_client:
            return (
                f"[Mode hors ligne – Groq non configuré]\n\n"
                f"Contexte trouvé:\n{context[:600]}"
            )
        try:
            resp = self._groq_client.chat.completions.create(
                model=_GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Contexte:\n{context}\n\nQuestion: {question}"},
                ],
                temperature=0.2,
                max_tokens=512,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq generation error: {e}")
            return f"Erreur de génération: {str(e)}"

    def explain_decision(self, plate: str, decision: str, reason: str) -> str:
        q = (f"Explique pourquoi le véhicule {plate} a reçu la décision "
             f"\"{decision}\" avec la raison: {reason}")
        return self.query(q)
