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

from utilities.dataloader import DocumentManager
from utilities.mongodb import MongoManager
from utilities.chunking_utilities import split_markdown_text, create_documents


app = FastAPI()

router = APIRouter()

# Percorso in cui salvare i PDF caricati
UPLOAD_FOLDER = Path(__file__).resolve().parent.parent / "input_data"
UPLOAD_FOLDER.mkdir(exist_ok=True)

doc_manager = DocumentManager()
mongo_manager = MongoManager(connection_string="mongodb://localhost:27017")



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
        chunks = split_markdown_text(text)

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
    





# Includi con prefisso e tag
app.include_router(
    router,
    prefix="/leonardo",       # tutte le rotte avranno questo prefisso
    tags=["Leonardo API"]     # mostrato nella documentazione Swagger
)