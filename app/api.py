from fastapi import FastAPI, HTTPException, APIRouter, Query, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
from pymongo import MongoClient
from pathlib import Path
from datetime import datetime
from uuid import uuid4
from langchain.schema import Document
import httpx
from transformers import AutoModel
import torch
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, SearchParams
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel, RunnableMap, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
import base64
from PIL import Image
from io import BytesIO
import os
from transformers import AutoModel
from pypdf import PdfReader
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

from utilities.dataloader import DocumentManager
from utilities.mongodb import MongoManager
from utilities.chunking import split_markdown_text, create_documents
from utilities.embeddings import JinaEmbeddings, get_embedding_model
from utilities.retrieving import retriever_jina

# @asynccontextmanager
# 📚 Quando usarlo
# • Quando hai bisogno di setup asincrono (es. apertura pool, handshake, caricamento risorse)
# • E di teardown garantito, anche in caso di eccezioni
# • Senza scrivere una classe completa con __aenter__/__aexit__

# Ecco un esempio minimale:

# @asynccontextmanager
# async def lifespan(app):
#     # setup al bootstrap
#     yield
#     # cleanup alla chiusura
# ✅ In sintesi
# @asynccontextmanager è un decoratore dedicato a generatori asincroni.

# Serve per creare context manager usufruibili con async with, con setup e cleanup garantiti.

# Non è un decoratore generico per qualunque funzione async.

# Con lifespan, invece, il modello si carica una sola volta al bootstrap dell’app, mantenendo memoria tra le chiamate.

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("Caricamento del modello Jina Emebddings...")
    hf_token=os.getenv("HF_TOKEN")
    model = get_embedding_model(hf_token)
    print("✔️ Modello caricato")
    yield
    # eventualmente cleanup
    model = None

app = FastAPI(lifespan=lifespan)

router = APIRouter()

# Percorso in cui salvare i PDF caricati
UPLOAD_FOLDER = Path(__file__).resolve().parent.parent / "input_data"
UPLOAD_FOLDER.mkdir(exist_ok=True)

doc_manager = DocumentManager()
mongo_manager = MongoManager(connection_string="mongodb://host.docker.internal:27017")
# connessione a Qdrant (Docker Locale)
client = QdrantClient(url="http://qdrant:6333")

#HF_TOKEN_STORE = {}


###################################### Pydantic Schemas ############################


class RetrieverResponse(BaseModel):
    answer: str
    images: list[str]


####################################### ENDPOINTS ###################################

@router.post("/upload-and-parse-pdf/")
async def upload_and_parse_pdf(file: UploadFile = File(...)):
    """
    📄 Carica e analizza un file PDF

    Questo endpoint permette di caricare un file PDF, salvarlo in locale, 
    estrarne il testo tramite `DocumentManager`, e generare un file `.md` 
    con il contenuto estratto.

    Args:
        file (UploadFile): File PDF da caricare.

    Returns:
        dict: Contiene il nome del file originale, il percorso locale, 
              il percorso del file Markdown generato, e un'anteprima del testo estratto.
    """

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Il file deve essere in formato PDF (.pdf)")
    
    # salva il file in locale
    file_path = UPLOAD_FOLDER / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Parsa il PDF usando DocumentManager
    parsed_text = doc_manager.read_local_pdf(str(file_path))

    # Estrai il nome base senza estensione
    nome_file = Path(file.filename).stem

    # Scrive file .md con il testo estratto
    new_md_file = doc_manager.create_local_document(file_name=f"{nome_file}.md", content=parsed_text)


    return {
        "filename": file.filename,
        "file_path": file_path,
        "new_md_file": new_md_file,
        "parsed_text": parsed_text[:500]
    }


@router.post("/upload-images-to-mongo", tags=["Leonardo API"])
async def upload_images_to_mongo():
    """
    🖼️ Carica immagini in MongoDB

    Questo endpoint cerca tutte le immagini nella cartella `img_out`, 
    le converte in base64 e le salva in MongoDB nella collezione `images` 
    del database `Leonardo`.

    Returns:
        dict: Messaggio con il numero di immagini inserite o avviso 
              se non sono state trovate immagini.
    """
    # 1. Path assoluto alla cartella delle immagini
    img_dir = Path(__file__).resolve().parent.parent / "utilities" / "img_out"

    # 2. Converte tutte le immagini in base64
    images_base64 = doc_manager.convert_images_to_base64(str(img_dir))

    # 3. Inserisce in MongoDB (es: database "leonardo", collection "images")
    if images_base64:

        # connessione con mongodb
        collection = mongo_manager._get_collection(database_name="Leonardo", collection_name="images")

        # estraimo gli hash dalle immagini convertite
        filename = images_base64[0]['filename']

        # otteniamo immagini di un file già presente
        existing_docs = list(collection.find({"filename": filename}))

        if existing_docs:
            return JSONResponse(
                content={"message": f"Immagini dello stesso file {filename} già presenti in MongoDB"},
                status_code=200
            )
        
        result = mongo_manager.write_to_mongo(
            data=images_base64,
            database_name="Leonardo",
            collection_name="images"
        )
        return {
            "inserted_count": len(images_base64),
            "message": f"{len(images_base64)} images successfully inserted to mongoDB"
        }
    else:
        return {"message": f"Nessuna immagine trovata in {str(img_dir)}"}
    

@router.post("/chunking", tags=["Leonardo API"])
async def chunking(file: UploadFile = File(...)):
    """
    📄 Endpoint per suddividere un file Markdown (.md) in chunk semantici ottimizzati per LLM.

    🔧 Funzionalità:
    - Riceve un file `.md` tramite form-data.
    - Esegue uno split ricorsivo basato sulla struttura Markdown (titoli, paragrafi, righe).
    - Rimuove link immagine Markdown dai chunk e li inserisce nei metadati.
    - Crea una lista di `Document` con contenuto pulito e metadati associati.

    📦 Output:
    - `total_chunks`: numero totale di chunk generati.
    - `chunks`: anteprima dei primi 10 chunk (come stringhe pulite).
    - `docs`: anteprima dei primi 10 chunk come oggetti `Document` serializzati con metadati (`chunk_no`, `images`).

    ⚠️ Gestione errori:
    - 400: se il file non ha estensione `.md`
    - 500: se avviene un errore durante la lettura, decodifica o segmentazione del file
    """
    if not file.filename.endswith(".md"):
        raise HTTPException(status_code=400, detail="Il file deve essere in formato Markdown (.md)")
    

    main_dir = Path(__file__).resolve().parent.parent
    md_path = main_dir / "input_data" / file.filename

    pdf_path = md_path.with_suffix(".pdf")

    if not pdf_path.exists():
        raise HTTPException(status_code=400, detail=f"Non trovato PDF: {pdf_path.name}")
    

    try:
        pdf_reader = PdfReader(str(pdf_path))
        num_pages = len(pdf_reader.pages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore lettura PDF: {e}")
    

    # Adatta chunk_size e overlap
    if num_pages < 10:
        chunk_size, chunk_overlap = 500, 10
    elif num_pages <= 50:
        chunk_size, chunk_overlap = 1000, 50
    elif num_pages <= 100:
        chunk_size, chunk_overlap = 2000, 100
    else:
        chunk_size, chunk_overlap = 3000, 200
    
    try:
        # Legge il contenuto del file
        contents = await file.read()
        text = contents.decode("utf-8")

        # Step 1 – Verifica duplicato in MongoDB
        collection = mongo_manager._get_collection(database_name="Leonardo", collection_name="documents")

        existing_docs = list(collection.find({"filename": {"$eq": file.filename}}))

        if existing_docs:
            return JSONResponse(
                content={"message": f"Documenti del file {file.filename} sono già presenti in MongoDB", "filename": file.filename},
                status_code=200
            )

        # Step 2: Chunking
        chunks = split_markdown_text(text=text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Step 3: Creazione Documents
        docs: List[Document] = create_documents(chunks)

        # Step 4: Salvataggio su MongoDB
        mongo_payload = [
            {
                "filename": file.filename,
                "document_id": str(uuid4()),
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "uploaded_at": datetime.utcnow().isoformat()
            }
            for doc in docs
        ]

        mongo_manager.write_to_mongo(data=mongo_payload, database_name="Leonardo", collection_name="documents")

        # Step 5: Output (rimuoviamo solo da preview, non dalla lista completa)
        preview = [
            {k: v for k, v in doc.items() if k not in {"file_id", "_id"}}
            for doc in mongo_payload[:10]
        ]

        return JSONResponse(content={
            "total_chunks": len(docs),
            "chunks": chunks[:10],
            "docs": preview
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante il chunking: {str(e)}")
    

# @router.post("/hf-login")
# async def hf_login(input: str):
#     """
#     Effettua il login a HuggingFace tramite token personale.

#     - Verifica il token con l'API HuggingFace (`/whoami-v2`)
#     - Salva il token e le info utente in memoria (HF_TOKEN_STORE)
#     - Non salva nulla su disco
#     - Usare solo in ambienti sicuri (HTTPS)
#     """

#     # verifica il token con Hugging Face
#     headers = {"Authorization": f"Bearer {input}"}

#     async with httpx.AsyncClient() as client:
#         response = await client.get("https://huggingface.co/api/whoami-v2", headers=headers)

    
#     if response.status_code != 200:
#         raise HTTPException(status_code=401, detail="Token HuggingFace non valido")
    
#     user_info = response.json()

#     HF_TOKEN_STORE["token"] = input
#     HF_TOKEN_STORE["user"] = user_info.get("name", "unknown")
    

#     return {
#         "message": "Login effettuato con successo",
#         "user": user_info.get("name", "unknown"),
#         "email": user_info.get("email", None)
#     }


@router.post("/embeddings")
async def embeddings(file_name: str):
    """
    🔍 Genera embedding da documenti e li salva in Qdrant.

    - Prende `file_name` dalla richiesta.
    - Recupera documenti e metadati da MongoDB.
    - Calcola gli embedding del testo usando JinaEmbeddings.
    - Crea o usa una collection Qdrant con dimensione vettoriale appropriata.
    - Inserisce o aggiorna i punti con ID univoco (upsert) per evitare duplicati.

    ✅ Parametri:
      - file_name (str): nome del file su cui eseguire embedding

    ✅ Risposta:
      - message: conferma del numero di embedding inseriti
      - collection_name: nome della collection di destinazione
      - vector_size: dimensionalità dei vettori salvati
    """

    # token = HF_TOKEN_STORE.get("token", "")
    # if not token:
    #     raise HTTPException(status_code=401, detail="Non autenticato su Hugging Face")
    
    # try:
    #     model = get_embedding_model(token)
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Errore nel caricamento del modello: {e}")

    embedding_manager = JinaEmbeddings(model=model)

    documents = mongo_manager.read_from_mongo(query={"filename": file_name}, output_format="object", database_name="Leonardo", collection_name="documents")

    print(len(documents))

    document_contents = []
    document_metadata = []
    for document in documents:
        page_content = document.get("page_content", "")
        metadata = document.get("metadata", {})
        metadata["filename"] = file_name

        document_contents.append(page_content)
        document_metadata.append(metadata)

    if len(document_contents) != len(document_metadata):
        raise Exception("Contents e metadata non coincidono!")
    
    print(f"Contents: {len(document_contents)}, Metadata: {len(document_metadata)}")


    # calcola gli embeddings solo sui page_contents
    embeddings = embedding_manager.embed_documents(document_contents)


    print(f"Numero di embeddings: {len(embeddings)}, dimensionalità : {len(embeddings[0])}")


    # creiamo la collection se già non eseiste
    collection_name = "hitachi"
    vector_size = len(embeddings[0]) # 2048

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
    
    # prepariamo i points per i nostri embeddings
    points = []
    for md, vec in zip(document_metadata, embeddings):
        chunk_no = md["chunk_no"]
        # ID univoco per chunk/documento
        id = str(uuid4())
        points.append(
            PointStruct(id=id, vector=vec, payload=md)
        )


    # inseriamo o aggiorniamo i vettori
    # se hanno gli stessi ID si aggiornano/sovrascrivono evitando duplicati
    client.upsert(
        collection_name=collection_name,
        points=points
    )

    return {
        "message": f"{len(embeddings)} embeddings con dimensionalità {vector_size} sono stati inseriti correttamente nella collection",
        "collection_name": collection_name,
        "vector_size": vector_size
    }



@router.post("/retriever", response_model=RetrieverResponse, status_code=200)
async def retriever(file_name: str, query: str):
    """
    🔍 Risponde a una query usando embeddings da un documento prefiltrato
    - file_name: nome del documento nel db da interrogare
    - query: domanda da porre al modello
    Restituisce la risposta testuale + eventuali immagini correlate (se negli embeddings retrivati erano allegate delle immagini)
    """

    query_filter = Filter(must=[
        FieldCondition(key="filename", match=MatchValue(value=file_name))
    ])


    template = """Answer the question based only on the following contetext:
    {context}

    Question: {question}"""

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOllama(model="llama3.1:8b", base_url="http://host.docker.internal:11434", temperature=0.2)

    if model is None:
        raise HTTPException(status_code=500, detail="❌ Modello di embedding Jina non caricato correttamente.")


    retriever_ji_full = RunnableLambda(lambda query: (*retriever_jina(query=query, model=model, query_filter=query_filter), query))

    split = RunnableMap({
        "context": RunnableLambda(lambda res: res[0]),
        "images": RunnableLambda(lambda res: res[1]),
        "question": RunnableLambda(lambda res: res[2]),
    })


    chain_model = (
        prompt
        | llm
        | StrOutputParser()
    )

    parallel = retriever_ji_full | split

    full_chain = RunnableSequence(
        parallel,
        # produce dict con context, images e question
        RunnableLambda(lambda data: {
            # genera la risposta llm
            "images": data['images'],
            "answer": chain_model.invoke({"context": data["context"], "question": data["question"]})
        })
    )

    try:
        print("[INFO] Invocazione LLM in corso...")
        result = full_chain.invoke(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante il retrieve/LLM: {e}")

    image_files = [Path(p).name for p in result["images"]]

    main_dir = Path(__file__).resolve().parent.parent
    output_dir = main_dir / "utilities" / "retrieved_images"
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    for image_name in image_files:
        docs = mongo_manager.read_from_mongo(
            query={"filename": image_name},
            output_format="object",
            database_name="Leonardo",
            collection_name="images"
        )
        if not docs:
            continue
        doc = docs[0]
        content_b64 = doc.get("content_base64")
        if not content_b64:
            continue

        try:
            image_data = base64.b64decode(content_b64)
            img = Image.open(BytesIO(image_data))
            save_path = output_dir / image_name
            img.save(save_path)
            saved_paths.append(str(save_path))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Errore nel salvataggio immagine {image_name}: {e}")

    return {"answer": result["answer"], "images": saved_paths}


# Includi con prefisso e tag
app.include_router(
    router,
    prefix="/leonardo",       # tutte le rotte avranno questo prefisso
    tags=["Leonardo API"]     # mostrato nella documentazione Swagger
)