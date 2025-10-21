from fastapi import FastAPI, HTTPException, APIRouter, Query, File, UploadFile
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from uuid import uuid4
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, SearchParams
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
import base64
from PIL import Image
from io import BytesIO
import os
import re
from pypdf import PdfReader
from dotenv import load_dotenv
load_dotenv()

from utilities.dataloader import DocumentManager
from utilities.mongodb import MongoManager
from utilities.chunking import split_markdown_text, create_documents
from utilities.jina_embeddings import JinaEmbeddings, load_hf_jina_model
from utilities.retrieving import retriever_generic


def _sanitize(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)

def _collection_name(base: str, provider: str, model_name: str) -> str:
    return f"{_sanitize(base)}__{_sanitize(provider)}__{_sanitize(model_name)}"


app = FastAPI()

router = APIRouter()

# Percorso in cui salvare i PDF caricati
UPLOAD_FOLDER = Path(__file__).resolve().parent.parent / "input_data"
UPLOAD_FOLDER.mkdir(exist_ok=True)

doc_manager = DocumentManager()
mongo_manager = MongoManager(connection_string="mongodb://host.docker.internal:27017")
# connessione a Qdrant (Docker Locale)
client = QdrantClient(url="http://qdrant:6333")



# ========== HELPERS INTERNI ==========


def parse_pdf_and_create_md(file_path: Path) -> dict:
    """Legge un PDF locale, estrae testo e crea .md in input_data."""
    if not file_path.name.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Il file deve essere in formato PDF (.pdf)")
    parsed_text = doc_manager.read_local_pdf(str(file_path))
    nome_file = file_path.stem
    new_md_file = doc_manager.create_local_document(file_name=f"{nome_file}.md", content=parsed_text)
    return {
        "filename": file_path.name,
        "file_path": str(file_path),
        "new_md_file": str(new_md_file),
        "parsed_text": parsed_text[:500],
    }


def upload_images_to_mongo_internal() -> dict:
    """Cerca immagini in utilities/img_out e le salva in Mongo."""
    img_dir = Path(__file__).resolve().parent.parent / "utilities" / "img_out"
    images_base64 = doc_manager.convert_images_to_base64(str(img_dir))
    if not images_base64:
        return {"message": f"Nessuna immagine trovata in {str(img_dir)}"}
    collection = mongo_manager._get_collection(database_name="Leonardo", collection_name="images")
    filename = images_base64[0]['filename']
    if list(collection.find({"filename": filename})):
        return {"message": f"Immagini dello stesso file {filename} già presenti in MongoDB"}
    mongo_manager.write_to_mongo(data=images_base64, database_name="Leonardo", collection_name="images")
    return {"inserted_count": len(images_base64), "message": f"{len(images_base64)} images successfully inserted to mongoDB"}


def chunk_md_and_store(md_filename: str, md_text: str) -> dict:
    """Effettua il chunking del markdown e salva i chunk su Mongo."""
    if not md_filename.endswith(".md"):
        raise HTTPException(status_code=400, detail="Il file deve essere in formato Markdown (.md)")
    main_dir = Path(__file__).resolve().parent.parent
    md_path = main_dir / "input_data" / md_filename
    pdf_path = md_path.with_suffix(".pdf")
    if not pdf_path.exists():
        raise HTTPException(status_code=400, detail=f"Non trovato PDF: {pdf_path.name}")

    try:
        num_pages = len(PdfReader(str(pdf_path)).pages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore lettura PDF: {e}")

    if num_pages < 10:
        chunk_size, chunk_overlap = 500, 10
    elif num_pages <= 50:
        chunk_size, chunk_overlap = 1000, 50
    elif num_pages <= 100:
        chunk_size, chunk_overlap = 2000, 100
    else:
        chunk_size, chunk_overlap = 3000, 200

    collection = mongo_manager._get_collection(database_name="Leonardo", collection_name="documents")
    if list(collection.find({"filename": {"$eq": md_filename}})):
        return {"message": f"Documenti del file {md_filename} già presenti in MongoDB", "filename": md_filename, "total_chunks": 0}

    chunks = split_markdown_text(text=md_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs: List[Document] = create_documents(chunks)
    mongo_payload = [{
        "filename": md_filename,
        "document_id": str(uuid4()),
        "page_content": doc.page_content,
        "metadata": doc.metadata,
        "uploaded_at": datetime.utcnow().isoformat()
    } for doc in docs]
    mongo_manager.write_to_mongo(data=mongo_payload, database_name="Leonardo", collection_name="documents")
    return {"total_chunks": len(docs)}


def build_embeddings_and_upsert(
    file_name: str,
    provider: str,
    model_name: str,
    base_url: str,
    hf_token: Optional[str],
) -> dict:
    """Calcola embeddings (Jina HF o Ollama) e li salva su Qdrant (collection namespaced)."""

    if provider.lower() == "jina" and not hf_token:
        raise HTTPException(status_code=400, detail="hf_token è richiesto per provider='jina'")

    # 1) Instanzia embeddings on-demand
    if provider.lower() == "jina":
        raw = load_hf_jina_model(model_name, hf_token or "")
        embedding_manager = JinaEmbeddings(raw)
    elif provider.lower() == "ollama":
        embedding_manager = OllamaEmbeddings(model=model_name, base_url=base_url)
    else:
        raise HTTPException(status_code=400, detail=f"Provider non supportato: {provider}")

    # 2) Leggi documenti da Mongo
    documents = mongo_manager.read_from_mongo(
        query={"filename": file_name},
        output_format="object",
        database_name="Leonardo",
        collection_name="documents"
    )
    if not documents:
        raise HTTPException(status_code=404, detail=f"Nessun documento con filename={file_name}")

    contents, metadatas = [], []
    for d in documents:
        contents.append(d.get("page_content", ""))
        md = d.get("metadata", {})
        md["filename"] = file_name
        metadatas.append(md)

    # 3) Calcolo embeddings
    vectors = embedding_manager.embed_documents(contents)
    if not vectors:
        raise HTTPException(status_code=500, detail="Nessun embedding generato")
    vector_size = len(vectors[0])

    # 4) Collection namespaced
    collection_name = _collection_name("company_name", provider, model_name)
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

    # 5) Upsert
    points = [PointStruct(id=str(uuid4()), vector=v, payload=md) for v, md in zip(vectors, metadatas)]
    client.upsert(collection_name=collection_name, points=points)

    return {
        "message": f"{len(vectors)} embeddings (dim={vector_size}) inseriti in '{collection_name}'",
        "collection_name": collection_name,
        "vector_size": vector_size,
    }


def retrieve_context_and_images(
    file_name: str,
    query: str,
    provider: str,
    emb_model_name: str,
    base_url: str,
    hf_token: Optional[str],
) -> tuple[str, List[str]]:
    """Esegue il retrieval (embedding della query + Qdrant + fetch contenuti/immagini)."""
    # filtro Qdrant
    query_filter = Filter(must=[FieldCondition(key="filename", match=MatchValue(value=file_name))])

    if provider.lower() == "jina" and not hf_token:
        raise HTTPException(status_code=400, detail="hf_token è richiesto per provider='jina'")


    # embeddings per la query
    if provider.lower() == "jina":
        raw = load_hf_jina_model(emb_model_name, hf_token or "")
        embeddings_for_query = JinaEmbeddings(raw)
    elif provider.lower() == "ollama":
        embeddings_for_query = OllamaEmbeddings(model=emb_model_name, base_url=base_url)
    else:
        raise HTTPException(status_code=400, detail=f"Provider non supportato: {provider}")

    # collection coerente
    collection_name = _collection_name("company_name", provider, emb_model_name)

    # usa retriever_generic parametrico (assicurati accetti collection_name)
    return retriever_generic(query, embeddings_for_query, query_filter, collection_name=collection_name)



# ==================== ENDPOINT VISIBILI ====================

visible = APIRouter(prefix="/v1", tags=["Pipeline"])

class IngestResult(BaseModel):
    filename: str
    md_file: str
    parsed_preview: str
    images_status: str

@visible.post("/ingest", response_model=IngestResult, summary="Ingest PDF → parse testo + inserimento immagini in Mongo")
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Ingest di un documento PDF.

    🔎 Cosa fa
    - Valida che il file sia un `.pdf`.
    - Salva il PDF in `input_data/` (per uso successivo nella pipeline).
    - Estrae il testo dal PDF e genera un file `.md` omonimo (stesso nome del PDF).
    - Cerca le immagini estratte in `utilities/img_out/`, le converte in base64 e le inserisce in MongoDB (`Leonardo.images`),
      evitando duplicati (match sul filename dell’immagine).

    ✉️ Parametri (multipart/form-data)
    - `file` (**UploadFile**, richiesto): il PDF da processare.

    📦 Output (`IngestResult`)
    - `filename` (str): nome del PDF caricato.
    - `md_file` (str): percorso assoluto del `.md` generato (lato server).
    - `parsed_preview` (str): preview (max ~500 char) del testo estratto.
    - `images_status` (str): messaggio sullo stato d’inserimento immagini in Mongo (es. già presenti / inserite N immagini).

    🗂️ Side effects
    - Crea/aggiorna file in `input_data/`.
    - Potenzialmente inserisce documenti in `Leonardo.images`.

    ⚠️ Errori comuni (HTTPException)
    - 400: file non `.pdf`.
    - 500: errori imprevisti nella lettura del PDF o nella generazione del `.md`.

    💡 Note
    - Non esegue chunking né embeddings: prepara solo i materiali per gli step successivi.
    - Le immagini ci si aspetta che siano state già estratte e salvate in `utilities/img_out/` (pipeline esterna).
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Il file deve essere in formato PDF (.pdf)")

    # salva il PDF in input_data (così il resto della pipeline lo trova)
    pdf_path = UPLOAD_FOLDER / file.filename
    pdf_path.write_bytes(await file.read())

    parsed = parse_pdf_and_create_md(pdf_path)
    img_res = upload_images_to_mongo_internal()

    return IngestResult(
        filename=parsed["filename"],
        md_file=parsed["new_md_file"],
        parsed_preview=parsed["parsed_text"],
        images_status=img_res.get("message", "OK"),
    )


@visible.post(
    "/process",
    summary="Esegue chunking + embeddings (stampa messaggi di stato)"
)
async def process_document(
    md_path: str = Query(..., description="Percorso assoluto del file .md già creato lato server"),
    file_name: Optional[str] = Query(None, description="Nome file base per embeddings (default: nome del .md)"),
    provider: str = Query("jina", description='Provider embeddings: "jina" | "ollama"'),
    model_name: str = Query("jinaai/jina-embeddings-v4", description="Modello embeddings (HF ID per Jina, tag per Ollama)"),
    base_url: str = Query("http://host.docker.internal:11434", description="Usato solo per Ollama"),
    hf_token: Optional[str] = Query(None, description="Token HF (richiesto per provider=jina)")
):
    """
    Esegue **chunking** del `.md` e **indicizzazione** (embeddings + upsert in Qdrant).

    Parametri (query/form):
    - `md_path` (str, richiesto): percorso assoluto del `.md` sul server.
    - `file_name` (str, opzionale): usato per etichettare i chunk/embeddings; default: `Path(md_path).name`.
    - `provider` (str, default `"jina"`): `"jina"` | `"ollama"`.
    - `model_name` (str): HF ID per Jina (es. `jinaai/jina-embeddings-v4`) o tag per Ollama (es. `nomic-embed-text`).
    - `base_url` (str): endpoint Ollama (ignorato per Jina).
    - `hf_token` (str, opzionale): token HF (richiesto se `provider="jina"`).

    Ritorna:
    - `message` (str), `total_chunks` (int), `embeddings_collection` (str), `vector_size` (int)
    """
    from pathlib import Path as _Path
    print("Document processing...")

    p = _Path(md_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"File .md non trovato: {p}")

    md_text = p.read_text(encoding="utf-8")
    chunk_res = chunk_md_and_store(p.name, md_text)

    target_file_name = file_name or p.name
    emb_res = build_embeddings_and_upsert(
        file_name=target_file_name,
        provider=provider,
        model_name=model_name,
        base_url=base_url,
        hf_token=hf_token,
    )

    print("Document processed successfully!")

    return {
        "message": "Document processed successfully!",
        "total_chunks": int(chunk_res.get("total_chunks", 0)),
        "embeddings_collection": emb_res.get("collection_name"),
        "vector_size": emb_res.get("vector_size"),
    }


@visible.post(
    "/retrieve",
    summary="Esegue retrieval + LLM"
)
async def retrieve(
    file_name: str = Query(..., description="Nome del documento indicizzato (match su payload `filename`)"),
    query: str = Query(..., description="Domanda da porre"),
    model_name: str = Query(..., description="Modello LLM su Ollama, es. 'qwen2.5:14b'"),
    provider: str = Query("jina", description='Provider embeddings: "jina" | "ollama"'),
    emb_model_name: str = Query("jinaai/jina-embeddings-v4", description="Modello embeddings usato in retrieval (deve coincidere con l’indicizzazione)"),
    base_url: str = Query("http://host.docker.internal:11434", description="Endpoint Ollama (per embeddings Ollama e per LLM)"),
    hf_token: Optional[str] = Query(None, description="Token HF (richiesto per provider=jina)")
):
    """
    Retrieval semantico + generazione risposta con LLM.

    Parametri (query/form):
    - `file_name` (str, richiesto): documento target (filtra i chunk in Qdrant).
    - `query` (str, richiesto): domanda da porre.
    - `model_name` (str, richiesto): modello LLM (Ollama).
    - `provider` (str, default `"jina"`): `"jina"` | `"ollama"`.
    - `emb_model_name` (str, default `"jinaai/jina-embeddings-v4"`): **deve combaciare** con l’indicizzazione.
    - `base_url` (str, default `"http://host.docker.internal:11434"`): endpoint Ollama.
    - `hf_token` (str, opzionale): richiesto se `provider="jina"`.

    Ritorna:
    - `answer` (str): risposta generata dall’LLM basata solo sul context recuperato.
    - `images` (List[str]): path su disco delle immagini salvate.
    """
    # 1) retrieval (context + immagini)
    context, images = retrieve_context_and_images(
        file_name=file_name,
        query=query,
        provider=provider,
        emb_model_name=emb_model_name,
        base_url=base_url,
        hf_token=hf_token,
    )

    # 2) LLM answering
    prompt = ChatPromptTemplate.from_template("""Answer the question based only on the following context:
    {context}

    Question: {question}""")
    llm = ChatOllama(model=model_name, base_url=base_url, temperature=0.2)
    answer = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": query})

    # 3) salva immagini
    from pathlib import Path as _Path
    main_dir = _Path(__file__).resolve().parent.parent
    output_dir = main_dir / "utilities" / "retrieved_images"
    os.makedirs(output_dir, exist_ok=True)

    saved_paths: List[str] = []
    for image_name in (_Path(p).name for p in images):
        docs = mongo_manager.read_from_mongo(
            query={"filename": image_name},
            output_format="object",
            database_name="Leonardo",
            collection_name="images"
        )
        if not docs:
            continue
        content_b64 = docs[0].get("content_base64")
        if not content_b64:
            continue
        try:
            image_data = base64.b64decode(content_b64)
            Image.open(BytesIO(image_data)).save(output_dir / image_name)
            saved_paths.append(str(output_dir / image_name))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Errore nel salvataggio immagine {image_name}: {e}")

    return {"answer": answer, "images": saved_paths}




# ==================== REGISTRAZIONE ROUTER ====================

# nuovi endpoint visibili sotto /v1
app.include_router(visible)