from fastapi import FastAPI, HTTPException, APIRouter, Query, File, UploadFile
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from uuid import uuid4
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, SearchParams
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding
from qdrant_client.models import models, Distance, VectorParams, PointStruct
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
from utilities.retrieving import retriever_generic, retriever_hybrid
from utilities.HyDE import _hyde_text_via_llm


def _sanitize(s: str) -> str:
    # Sostituisce gruppi di caratteri non validi con un solo underscore
    sanitized = re.sub(r'[^a-zA-Z0-9_.-]+', '_', s)
    # Rimuove underscore multipli consecutivi
    sanitized = re.sub(r'__+', '_', sanitized)
    # Rimuove underscore iniziali/finali (opzionale)
    sanitized = sanitized.strip('_')
    return sanitized

def _collection_name(base: str, provider: str, model_name: str) -> str:
    return f"{_sanitize(base)}_{_sanitize(provider)}_{_sanitize(model_name)}"


app = FastAPI()

router = APIRouter()

# Percorso in cui salvare i PDF caricati
UPLOAD_FOLDER = Path(__file__).resolve().parent.parent / "input_data"
UPLOAD_FOLDER.mkdir(exist_ok=True)

doc_manager = DocumentManager()
mongo_manager = MongoManager(connection_string="mongodb://localhost:27017")
# connessione a Qdrant (Docker Locale)
client = QdrantClient(url="http://localhost:6333")


bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25")
late_interaction_embedding_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")



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


ALLOWED_META_KEYS = ["filename", "page", "page_num", "chunk_id", "chunk_no", "source", "doc_id"]

def _slim_payload(md: Dict[str, Any]) -> Dict[str, Any]:
    # tieni solo i metadati utili e sicuramente serializzabili
    out = {k: md[k] for k in ALLOWED_META_KEYS if k in md}
    # aggiungi qui eventuali campi piccoli che ti servono
    return out


def build_embeddings_and_upsert(
    file_name: str,
    provider: str,
    dense_model_name: str,
    hybrid_search: bool,
    hnsw: Optional[bool],
    base_url: str,
    hf_token: Optional[str],
) -> dict:
    """Calcola embeddings (Jina HF o Ollama) e li salva su Qdrant (collection namespaced)."""

    if provider.lower() == "jina" and not hf_token:
        raise HTTPException(status_code=400, detail="hf_token è richiesto per provider='jina'")

    # 1) Instanzia modello per dense embeddings on-demand
    if provider.lower() == "jina":
        raw = load_hf_jina_model(dense_model_name, hf_token or "")
        embedding_manager = JinaEmbeddings(raw)
    elif provider.lower() == "ollama":
        embedding_manager = OllamaEmbeddings(model=dense_model_name, base_url=base_url)
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

    # 2) Leggi documenti (già chunkati)
    documents = mongo_manager.read_from_mongo(
        query={"filename": file_name},
        output_format="object",
        database_name="Leonardo",
        collection_name="documents",
    )
    if not documents:
        raise HTTPException(status_code=404, detail=f"Nessun documento con filename={file_name}")

    contents, metadatas = [], []
    for d in documents:
        contents.append(d.get("page_content", ""))
        md = dict(d.get("metadata", {}))
        md["filename"] = file_name
        metadatas.append(_slim_payload(md))  # payload leggero

    # 3) Dense embeddings (una sola volta)
    print(f"Calcolando dense embeddings con {dense_model_name}...")
    dense_embeddings = embedding_manager.embed_documents(contents)
    if not dense_embeddings:
        raise HTTPException(status_code=500, detail="Nessun embedding generato")
    dense_dim = len(dense_embeddings[0])

    # 4) Se ibrido, calcola anche BM25 e ColBERT client-side
    if hybrid_search:
        print("Calcolando BM25 e ColBERT (late interaction) client-side...")

        # generator -> lista per poter riutilizzare/iterare più volte
        bm25_embeds = list(bm25_embedding_model.embed(contents))
        colbert_embeds = list(late_interaction_embedding_model.embed(contents))  # -> List[List[List[float]]]
        colbert_dim = len(colbert_embeds[0][0])  # tipicamente 128

        # 4a) Crea collection ibrida
        collection_name = _collection_name("company_name_hybrid_search", provider, dense_model_name)
        print(f"Creando la collection {collection_name}...")
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    f"{dense_model_name}": models.VectorParams(
                        size=dense_dim,
                        distance=models.Distance.COSINE,
                    ),
                    "colbertv2.0": models.VectorParams(
                        size=colbert_dim,
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        ),
                    ),
                },
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )

        # 4b) Upsert ibrido (batch)
        BATCH = 10
        points_batch = []
        for i in tqdm(range(len(contents)), total=len(contents), desc="Upsert hybrid"):
            # BM25 client-side -> SparseVector(indices, values)
            bm25_obj = bm25_embeds[i]
            if hasattr(bm25_obj, "indices"):
                bm25_sv = models.SparseVector(indices=bm25_obj.indices, values=bm25_obj.values)
            elif isinstance(bm25_obj, dict):
                bm25_sv = models.SparseVector(indices=bm25_obj["indices"], values=bm25_obj["values"])
            else:
                # fallback: molti wrapper espongono .as_object()
                bm25_dict = bm25_obj.as_object()
                bm25_sv = models.SparseVector(indices=bm25_dict["indices"], values=bm25_dict["values"])

            points_batch.append(
                models.PointStruct(
                    id=str(uuid4()),
                    vector={
                        f"{dense_model_name}": dense_embeddings[i],
                        "colbertv2.0": colbert_embeds[i],  # lista di sub-vettori
                        "bm25": bm25_sv
                    },
                    payload=metadatas[i],
                )
            )

            if len(points_batch) >= BATCH:
                client.upload_points(
                    collection_name=collection_name,
                    points=points_batch
                )
                while True:
                    collection_info = client.get_collection(collection_name=collection_name)
                    if collection_info.status == models.CollectionStatus.GREEN:
                        # collection status is green, il quale significa che l'indexing è terminato (grafo HNSW creato!)
                        break

                points_batch.clear()

        if points_batch:
            client.upload_points(
                collection_name=collection_name, 
                points=points_batch
            )
            while True:
                collection_info = client.get_collection(collection_name=collection_name)
                if collection_info.status == models.CollectionStatus.GREEN:
                    # collection status is green, il quale significa che l'indexing è terminato (grafo HNSW creato!)
                    print("✅ Collection pronta: indicizzazione completata!")
                    break

        # aggiorniamo la collection con i nuovi parametri m e ef_construct per la ricerca sul grafo HNSW
        if hnsw:

            # parametri m e ef_construct
            # di default sono 16 e 100
            m = 32
            ef_construct = 200

            client.update_collection(
                collection_name=collection_name,
                hnsw_config=models.HnswConfigDiff(
                    m=m,
                    ef_construct=ef_construct
                )
            )
            print(f"Aggiornando collection con m={m} e ef_construct={ef_construct}...")
            while True:
                collection_info = client.get_collection(collection_name=collection_name)
                if collection_info.status == models.CollectionStatus.GREEN:
                    print(f"✅ Collection aggiornata con parametri M e EF_CONSTRUCT!")
                    break

        message = f"Embeddings ibridi (dense + bm25 + ColBERT) inseriti in '{collection_name}'"
    else:
        # 5) Sempre: crea anche la collection SOLO-dense (se vuoi mantenerla)
        dense_collection = _collection_name("company_name", provider, dense_model_name)
        print(f"Creando la collection {dense_collection}...")
        if not client.collection_exists(dense_collection):
            client.create_collection(
                collection_name=dense_collection,
                vectors_config={
                    f"{dense_model_name}": models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)
                },
            )

        # Upsert dense (batch)
        BATCH = 500
        points_batch = []
        for vec, md in tqdm(zip(dense_embeddings, metadatas), total=len(dense_embeddings), desc="Upsert dense"):
            points_batch.append(
                models.PointStruct(
                    id=str(uuid4()),
                    vector={f"{dense_model_name}": vec},
                    payload=md,
                )
            )
            if len(points_batch) >= BATCH:
                client.upload_points(collection_name=dense_collection, points=points_batch)
                while True:
                    collection_info = client.get_collection(collection_name=dense_collection)
                    if collection_info.status == models.CollectionStatus.GREEN:
                        # collection status is green, il quale significa che l'indexing è terminato (grafo HNSW creato!)
                        break

                points_batch.clear()
        if points_batch:
            client.upload_points(collection_name=dense_collection, points=points_batch)
            while True:
                    collection_info = client.get_collection(collection_name=dense_collection)
                    if collection_info.status == models.CollectionStatus.GREEN:
                        # collection status is green, il quale significa che l'indexing è terminato (grafo HNSW creato!)
                        print(f"✅ Collection pronta: indicizzazione completata!")
                        break

        # aggiorniamo la collection con i nuovi parametri m e ef_construct per la ricerca sul grafo HNSW
        if hnsw:

            # parametri m e ef_construct
            # di default sono 16 e 100
            m = 32
            ef_construct = 200

            client.update_collection(
                collection_name=dense_collection,
                hnsw_config=models.HnswConfigDiff(
                    m=m,
                    ef_construct=ef_construct
                )
            )
            print(f"Aggiornando collection con m={m} e ef_construct={ef_construct}...")
            while True:
                collection_info = client.get_collection(collection_name=dense_collection)
                if collection_info.status == models.CollectionStatus.GREEN:
                    print(f"✅ Collection aggiornata con parametri M e EF_CONSTRUCT!")
                    break

    # 6) Ritorno coerente
    return {
        "message": message if hybrid_search else f"Embeddings dense inseriti in '{dense_collection}'",
        "collection_name": collection_name if hybrid_search else dense_collection,
        "dense_embeddings_size": dense_dim,
        "Grafo HNSW": f"Grafo HNSW aggiornato con i nuovi M {m} e EF_CONSTRUCT {ef_construct}" if hnsw else "Grafo HNSW con m=16 e ef_construct=100"
    }


def retrieve_context_and_images(
    file_name: str,
    query: str,
    qp_hyde: bool, 
    provider: str,
    emb_model_name: str,
    hybrid_search: bool,
    rerank: bool,
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
    
    # --------------------- HyDE Text --------------------
    hyde_text = None
    qp_model_name = "qwen2.5:7b"

    if qp_hyde:
        try:
            hyde_model = qp_model_name or "qwen3:8b" 
            hyde_text = _hyde_text_via_llm(query, base_url, hyde_model)
            print(f"[QP-HyDE] hyde document: {hyde_text}")
        except Exception as e:
            print(f"[QP-HyDE] fallback (errore: {e})")
            hyde_text = None

    if hybrid_search:
        collection_name = _collection_name("company_name_hybrid_search", provider, emb_model_name)
        # ✅ check esistenza collection ibrida
        if not client.collection_exists(collection_name):
            raise HTTPException(
                status_code=404,
                detail=(
                    f"La collection ibrida '{collection_name}' non esiste. "
                    f"Assicurati di aver indicizzato con hybrid_search=True "
                    f"e lo stesso provider/modello: provider='{provider}', emb_model_name='{emb_model_name}'."
                ),
            )
        return retriever_hybrid(query, embeddings_for_query, bm25_embedding_model, late_interaction_embedding_model,
                                collection_name, rerank, hyde_text)     
    else: 
        if rerank:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Non si può effettuare il Reraking su una collection NON ibrida."
                ),
            )
        collection_name = _collection_name("company_name", provider, emb_model_name)
        # ✅ check esistenza collection solo-densa
        if not client.collection_exists(collection_name):
            raise HTTPException(
                status_code=404,
                detail=(
                    f"La collection '{collection_name}' non esiste. "
                    f"Indicizza prima il documento con lo stesso provider/modello "
                    f"(provider='{provider}', emb_model_name='{emb_model_name}')."
                ),
            )
        # usa retriever_generic parametrico (assicurati accetti collection_name)
        return retriever_generic(query, embeddings_for_query, query_filter, collection_name, hyde_text)



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
    hybrid_search: bool = Query(False, description="Se True, abilita la creazione e l'inserimento di embeddings ibridi (dense + sparse + ColBERT)."),
    hnsw: Optional[bool] = Query(False, description="Se True, abilita l'update del grafo HNSW con i parametri m e ef_construct."),
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
    - `hybrid_search`: (bool, default False) crea la collection ibrida.
    - `hnsw`: (bool, default False) aggiorna il grafo HNSW con i parametri M e EF_CONSTRUCT.
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
        dense_model_name=model_name,
        hybrid_search=hybrid_search,
        hnsw=hnsw,
        base_url=base_url,
        hf_token=hf_token,
    )

    print("Document processed successfully!")

    return {
        "message": "Document processed successfully!",
        "total_chunks": int(chunk_res.get("total_chunks", 0)),
        "embeddings_collection": emb_res.get("collection_name"),
        "dense_embeddings_size": emb_res.get("dense_embeddings_size")
    }


@visible.post(
    "/retrieve",
    summary="Esegue retrieval + LLM"
)
async def retrieve(
    file_name: str = Query(..., description="Nome del documento indicizzato (match su payload `filename`)"),
    query: str = Query(..., description="Domanda da porre"),
    qp_hyde: bool = Query(False, description="Se True, si genera un documento ipotetico dalla query per migliorare il retrieve."),
    model_name: str = Query(..., description="Modello LLM su Ollama, es. 'qwen2.5:14b'"),
    provider: str = Query("jina", description='Provider embeddings: "jina" | "ollama"'),
    emb_model_name: str = Query("jinaai/jina-embeddings-v4", description="Modello embeddings usato in retrieval (deve coincidere con l’indicizzazione)"),
    hybrid_search: bool = Query(True, description="Se True, recupera i vettori dense e sparse più simili e li fonde con RRF in un'unica lista (deve esistere la collection ibrida)."),
    rerank: bool = Query(False, description="Se True, esegue il reranking usando ColBERT che embeddizza la query in multivector."),
    base_url: str = Query("http://host.docker.internal:11434", description="Endpoint Ollama (per embeddings Ollama e per LLM)"),
    hf_token: Optional[str] = Query(None, description="Token HF (richiesto per provider=jina)")
):
    """
    Retrieval semantico + generazione risposta con LLM.

    Parametri (query/form):
    - `file_name` (str, richiesto): documento target (filtra i chunk in Qdrant).
    - `query` (str, richiesto): domanda da porre.
    - `qp_hyde` (bool, richiesto): per effettuare il HyDE dalla query.
    - `model_name` (str, richiesto): modello LLM (Ollama).
    - `provider` (str, default `"jina"`): `"jina"` | `"ollama"`.
    - `emb_model_name` (str, default `"jinaai/jina-embeddings-v4"`): **deve combaciare** con l’indicizzazione.
    - `hybrid_search` (bool, default `True`): esegue la ricerca ibrida su embeddings dense e sparse (deve esistere la collection ibrida).
    - `rerank` (bool, default `False`): esegue il reranking con ColBERT query multivector.
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
        qp_hyde=qp_hyde,
        provider=provider,
        emb_model_name=emb_model_name,
        hybrid_search=hybrid_search,
        rerank=rerank,
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