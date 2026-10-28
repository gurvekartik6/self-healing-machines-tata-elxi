"""Approved-document registry, ingestion, and retrieval for maintenance RAG.

Only documents with status=approved, an unexpired approval, and matching asset
metadata are retrievable. PDF/DOCX extractors are optional dependencies.
"""
from __future__ import annotations
import hashlib, json, re, sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KB = ROOT / "knowledge" / "maintenance_docs"
DB_PATH = ROOT / "knowledge" / "maintenance_documents.db"
SUPPORTED = {".md", ".txt", ".pdf", ".docx"}


def _now() -> str: return datetime.now(timezone.utc).isoformat()
def _db(path: Path = DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS documents (
      doc_id TEXT PRIMARY KEY, path TEXT NOT NULL, sha256 TEXT NOT NULL, title TEXT NOT NULL,
      asset_model TEXT NOT NULL, revision TEXT NOT NULL, status TEXT NOT NULL,
      approved_by TEXT, approved_at TEXT, expires_at TEXT, source_type TEXT NOT NULL,
      ingested_at TEXT NOT NULL, notes TEXT DEFAULT '')""")
    con.execute("CREATE TABLE IF NOT EXISTS document_events(event_id INTEGER PRIMARY KEY AUTOINCREMENT,doc_id TEXT NOT NULL,event TEXT NOT NULL,actor TEXT,created_at TEXT NOT NULL,details TEXT DEFAULT '')")
    con.commit(); return con

def _extract(path: Path) -> list[dict[str, Any]]:
    ext = path.suffix.lower()
    if ext in {'.txt', '.md'}:
        return [{"page": None, "section": "text", "text": path.read_text(encoding='utf-8', errors='replace')}]
    if ext == '.pdf':
        try: import pypdf
        except ImportError as e: raise RuntimeError('Install pypdf to ingest PDF manuals: pip install pypdf') from e
        reader = pypdf.PdfReader(str(path)); pages=[]
        for i, page in enumerate(reader.pages, 1): pages.append({"page": i, "section": "page", "text": page.extract_text() or ""})
        return pages
    if ext == '.docx':
        try: from docx import Document
        except ImportError as e: raise RuntimeError('Install python-docx to ingest DOCX manuals: pip install python-docx') from e
        doc=Document(str(path)); sections=[]; heading='Document start'; lines=[]
        for p in doc.paragraphs:
            if p.style and p.style.name.lower().startswith('heading'):
                if lines: sections.append({"page": None, "section": heading, "text": '\n'.join(lines)}); lines=[]
                heading=p.text.strip() or 'Untitled section'
            elif p.text.strip(): lines.append(p.text.strip())
        if lines: sections.append({"page": None, "section": heading, "text": '\n'.join(lines)})
        return sections
    raise ValueError(f'Unsupported document type: {ext}')

def _chunks(text: str, max_chars: int = 1400, overlap: int = 180) -> list[str]:
    cleaned=re.sub(r'\s+', ' ', text).strip()
    if not cleaned: return []
    out=[]; start=0
    while start < len(cleaned):
        end=min(len(cleaned), start+max_chars)
        if end < len(cleaned):
            boundary=max(cleaned.rfind('. ', start+max_chars//2, end), cleaned.rfind(' ', start+max_chars//2, end))
            if boundary > start: end=boundary+1
        out.append(cleaned[start:end].strip())
        if end >= len(cleaned): break
        start=max(start+1,end-overlap)
    return out

def register_document(path: str | Path, *, title: str, asset_model: str, revision: str,
                      status: str = 'pending', approved_by: str | None = None,
                      expires_at: str | None = None, notes: str = '', db_path: Path = DB_PATH) -> dict[str, Any]:
    p=Path(path).resolve()
    if p.suffix.lower() not in SUPPORTED or not p.is_file(): raise ValueError('Provide an existing .md/.txt/.pdf/.docx file')
    if status not in {'pending','approved','rejected','expired'}: raise ValueError('Invalid document status')
    if status == 'approved' and not (approved_by and approved_by.strip()): raise ValueError('Approved documents require approved_by')
    if status == 'approved' and not expires_at: raise ValueError('Approved documents require an expiry date/time (ISO-8601)')
    if expires_at:
        try: datetime.fromisoformat(expires_at.replace('Z','+00:00'))
        except ValueError as e: raise ValueError('expires_at must be ISO-8601, e.g. 2027-12-31T23:59:00+00:00') from e
    # Validate that text can be extracted before registering.
    _extract(p)
    digest=hashlib.sha256(p.read_bytes()).hexdigest(); doc_id=digest[:16]
    rel=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
    con=_db(db_path)
    con.execute('''INSERT INTO documents(doc_id,path,sha256,title,asset_model,revision,status,approved_by,approved_at,expires_at,source_type,ingested_at,notes)
      VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(doc_id) DO UPDATE SET title=excluded.title,asset_model=excluded.asset_model,revision=excluded.revision,status=excluded.status,approved_by=excluded.approved_by,approved_at=excluded.approved_at,expires_at=excluded.expires_at,ingested_at=excluded.ingested_at,notes=excluded.notes''',
      (doc_id,rel,digest,title,asset_model,revision,status,approved_by,_now() if status=='approved' else None,expires_at,p.suffix.lower(),_now(),notes))
    con.execute('INSERT INTO document_events(doc_id,event,actor,created_at,details) VALUES(?,?,?,?,?)',(doc_id,'REGISTERED',approved_by,_now(),f'status={status}; revision={revision}; asset_model={asset_model}'))
    con.commit(); con.close()
    return {"doc_id":doc_id,"path":rel,"sha256":digest,"title":title,"asset_model":asset_model,"revision":revision,"status":status}

def approve_document(doc_id: str, approver: str, expires_at: str, *, db_path: Path = DB_PATH) -> None:
    if not approver.strip(): raise ValueError('Approver is required')
    try: dt=datetime.fromisoformat(expires_at.replace('Z','+00:00'))
    except ValueError as e: raise ValueError('Expiry must be ISO-8601') from e
    if dt <= datetime.now(timezone.utc): raise ValueError('Expiry must be in the future')
    con=_db(db_path); cur=con.execute("UPDATE documents SET status='approved',approved_by=?,approved_at=?,expires_at=? WHERE doc_id=?",(approver,_now(),expires_at,doc_id))
    if cur.rowcount: con.execute('INSERT INTO document_events(doc_id,event,actor,created_at,details) VALUES(?,?,?,?,?)',(doc_id,'APPROVED',approver,_now(),f'expires_at={expires_at}'))
    con.commit(); con.close()
    if cur.rowcount == 0: raise ValueError('Unknown document id')

def list_documents(*, db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    con=_db(db_path); rows=[dict(r) for r in con.execute('SELECT * FROM documents ORDER BY ingested_at DESC')]; con.close(); return rows

def document_audit(*, db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    con=_db(db_path); rows=[dict(r) for r in con.execute('SELECT * FROM document_events ORDER BY event_id DESC')]; con.close(); return rows

def _active_docs(kb_dir: Path, asset_model: str | None, db_path: Path) -> list[dict[str, Any]]:
    registered={r['path']:r for r in list_documents(db_path=db_path)}; docs=[]; now=datetime.now(timezone.utc)
    # Strict approval-by-registry: unregistered files are deliberately excluded.
    for r in registered.values():
        if r['status'] != 'approved' or not r['approved_by'] or not r['expires_at']: continue
        try:
            expiry=datetime.fromisoformat(r['expires_at'].replace('Z','+00:00'))
            if expiry.tzinfo is None: expiry=expiry.replace(tzinfo=timezone.utc)
            if expiry <= now: continue
        except ValueError: continue
        if asset_model and r['asset_model'] not in {asset_model, '*'}: continue
        p=Path(r['path']); p=p if p.is_absolute() else ROOT/p
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != r['sha256']: continue
        try: blocks=_extract(p)
        except Exception: continue
        for block in blocks:
            for i, chunk in enumerate(_chunks(block['text'])):
                docs.append({"source":r['path'],"title":r['title'],"doc_id":r['doc_id'],"asset_model":r['asset_model'],"revision":r['revision'],"approved_by":r['approved_by'],"expires_at":r['expires_at'],"page":block['page'],"section":block['section'],"chunk":i,"text":chunk})
    return docs

def retrieve(query: str, top_k: int = 4, kb_dir: Path | str = DEFAULT_KB, *, asset_model: str | None = None, db_path: Path = DB_PATH, method: str = 'auto') -> list[dict[str, Any]]:
    """Retrieve only approved, unexpired, hash-matching documents. TF-IDF default; optional Google+Chroma vector mode."""
    if not query.strip() or top_k < 1: return []
    docs=_active_docs(Path(kb_dir),asset_model,Path(db_path))
    if not docs: return []
    if method in {'auto','vector'} and __import__('os').getenv('RAG_VECTOR_BACKEND','').lower() == 'google_chroma':
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            from langchain_chroma import Chroma
            import tempfile
            embeddings=GoogleGenerativeAIEmbeddings(model='models/text-embedding-004')
            store=Chroma.from_texts([d['text'] for d in docs],embeddings,metadatas=[{k:v for k,v in d.items() if k!='text' and v is not None} for d in docs], persist_directory=None)
            found=store.similarity_search_with_relevance_scores(query,k=top_k)
            out=[]
            for doc,score in found:
                item=dict(doc.metadata); item['text']=doc.page_content; item['score']=round(float(score),4); item['retrieval_method']='google_chroma'; out.append(item)
            return out
        except Exception:
            if method == 'vector': raise RuntimeError('Vector retrieval requested but Google embeddings/Chroma failed; check optional packages, API key and quota.')
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    corpus=[d['text'] for d in docs]
    try: matrix=TfidfVectorizer(stop_words='english',ngram_range=(1,2)).fit_transform(corpus+[query])
    except ValueError: return []
    scores=cosine_similarity(matrix[-1],matrix[:-1]).ravel(); ranked=sorted(enumerate(scores),key=lambda x:x[1],reverse=True)
    out=[]
    for idx,score in ranked:
        if score <= 0: continue
        item=dict(docs[idx]); item['score']=round(float(score),4); item['retrieval_method']='tfidf'; out.append(item)
        if len(out)>=top_k: break
    return out
