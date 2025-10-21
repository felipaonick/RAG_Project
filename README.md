# 📘 Documentazione Applicazione FastAPI

Questa applicazione è costruita con **FastAPI** e si basa su container **Docker** e su un database **MongoDB**.  
Gli endpoint sono definiti all’interno del modulo `app/api.py` e utilizzano varie classi e funzioni di supporto presenti nella directory `utilities`.

---

## 🚀 Requisiti

Prima di avviare l’applicazione, è necessario avere installato:

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)  
- [MongoDB](https://www.mongodb.com/try/download/community)

---

## ▶️ Avvio dell’applicazione

1. Clona il repository sul tuo sistema:
   ```bash
   git clone <URL_REPOSITORY>
   cd <NOME_DIRECTORY>
```

2. Assicurati che **Docker Desktop** e **MongoDB** siano in esecuzione.

3. Esegui il comando per avviare i container:

```bash
   docker compose up -d --build
```

   Questo comando:

   * installerà tutte le dipendenze specificate in `requirements.txt`;
   * avvierà l’applicazione eseguendo automaticamente:

     ```bash
     python -m uvicorn app.api:app --host 0.0.0.0 --port 8091
     ```

4. Una volta avviata, l’applicazione sarà accessibile su:
   👉 [http://localhost:8091](http://localhost:8091)

---

## 📂 Struttura del progetto

```
LEONARDO_RAG
├── .venv/
├── app/
│   ├── __pycache__/
│   └── api.py
├── input_data/
├── tests/
├── utilities/
│   ├── __pycache__/
│   ├── img_out/
│   ├── retrieved_images/
│   ├── __init__.py
│   ├── chunking.py
│   ├── dataloader.py
│   ├── embeddings.py
│   ├── mongodb.py
│   └── retrieving.py
├── .env
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── README.md
├── requirements.txt
└── TODO.md
```

---

## 🔗 Endpoints (modulo `app/api.py`)

Gli endpoint dell’applicazione sono definiti in `app/api.py`.
Al loro interno vengono utilizzate funzioni di supporto e classi definite nei vari moduli della directory `utilities`.

👉 La spiegazione dei singoli endpoint, insieme ai metodi e alle classi di supporto, verrà fornita **man mano** che intercorrono negli endpoints.

---

## 🛠️ Utilities

La directory `utilities` contiene i moduli di supporto che forniscono classi e metodi fondamentali per il funzionamento degli endpoint.
Alcuni esempi:

* **`mongodb.py`** → Gestione connessioni e query su MongoDB.
* **`dataloader.py`** → Parsing e caricamento di documenti (PDF, immagini, ecc.).
* **`chunking.py`** → Gestione e suddivisione dei testi in blocchi semantici.
* **`embeddings.py`** → Creazione e utilizzo di embeddings per modelli AI.
* **`retrieving.py`** → Funzioni di retrieval da database o documenti esterni.

---

## 📖 Come funziona

1. **Richiesta API** → un client invia la richiesta a uno degli endpoint di `api.py`.
2. **Elaborazione** → l’endpoint richiama funzioni e classi definite in `utilities`.
3. **Accesso ai dati** → tramite `MongoDBToolKitManager` o altri moduli, i dati vengono letti/salvati nel database.
4. **Risposta** → l’API restituisce un output JSON al client.

---

## 📚 Documentazione per gli Endpoint


# 🧭 Panoramica

Questa sezione:

1. carica le variabili d’ambiente,
2. definisce un **lifespan asincrono** per inizializzare (e rilasciare) il modello di embedding una sola volta,
3. crea l’istanza FastAPI, un router, le cartelle di lavoro, e i **manager** (Documenti, MongoDB, Qdrant).

# 🔐 Variabili d’ambiente

```python
from dotenv import load_dotenv
load_dotenv()
```

Carica dal file `.env` valori come **HF\_TOKEN** per autenticarsi su HuggingFace o altri segreti (es. credenziali DB).
✅ Verifica: il file `.env` è presente e non viene committato (è in `.gitignore`).

---

# ⚙️ Lifespan asincrono (setup/teardown dell’app)

```python
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
```

## Cos’è e perché usarlo

* `@asynccontextmanager` definisce un **context manager asincrono** usato da FastAPI come **ciclo di vita** dell’app.
* Esegue il **setup una sola volta all’avvio** (bootstrap) e il **cleanup alla chiusura**.

## Cosa inizializza

* Carica il **modello di embedding** (Jina) tramite `get_embedding_model(hf_token)`.
* Lo salva in una **variabile globale `model`** per riutilizzarlo negli endpoint senza ricaricarlo ad ogni richiesta (risparmio enorme di tempo e memoria).

## Note operative

* Se il token HF non è valido o mancante → solleverà un errore in avvio.
  👉 Consiglio: gestire eccezioni e loggare un messaggio chiaro.
* `model = None` nel teardown libera memoria alla chiusura del server.

---

# 🏁 Istanza FastAPI e Router

```python
app = FastAPI(lifespan=lifespan)
router = APIRouter()
```

* L’app FastAPI usa il **lifespan** definito sopra.
* `router` verrà usato per raggruppare gli endpoint (es. `@router.post(...)`) e poi **incluso** nell’app (`app.include_router(router, prefix="...")`) più avanti nel file.

---

# 📂 Path di lavoro (upload) e manager applicativi

```python
UPLOAD_FOLDER = Path(__file__).resolve().parent.parent / "input_data"
UPLOAD_FOLDER.mkdir(exist_ok=True)
```

* Crea (se non esiste) `input_data/` **alla radice del progetto** (stessa struttura della tua repo).
* Qui verranno salvati i PDF o file caricati via API (`UploadFile`).

```python
doc_manager = DocumentManager()
mongo_manager = MongoManager(connection_string="mongodb://host.docker.internal:27017")
```

* `DocumentManager` → lettura, parsing, conversione documenti (PDF, immagini, ecc.).
* `MongoManager` → wrapper per collegarsi a **MongoDB**.

💡 **Perché `host.docker.internal`?**
All’interno del **container Docker**, questo hostname punta alla macchina host.

* Se MongoDB gira **sul tuo host** (non in Docker), questo è l’endpoint corretto.
* Se invece MongoDB gira **in un container** nello stesso compose, conviene usare il **service name** definito in `docker-compose.yml` (es. `mongodb:27017`).

```python
client = QdrantClient(url="http://qdrant:6333")
```

* Crea la connessione a **Qdrant** (vector DB) tramite l’URL del **service name** `qdrant` esposto sulla porta `6333` (tipico in Docker Compose).
* Verrà usato per **indicizzare e cercare** embeddings (similarity / ANN search).

---

# 🧩 Come interagiscono questi pezzi (in pratica)

1. **Avvio server** → `lifespan` carica **una volta** il modello di embedding Jina (usando `HF_TOKEN`).
2. **Richiesta API** → un endpoint (definito più avanti) usa:

   * `DocumentManager` per leggere/convertire documenti,
   * `JinaEmbeddings`/`model` per calcolare vettori,
   * `QdrantClient` per upsert/search,
   * `MongoManager` per salvare/mettere a disposizione metadati o risultati.
3. **Risposta** → JSON formattato con Pydantic.

---

# 📚 Endpoints definiti in `app/api.py`

Tutti gli endpoint sono accessibili con il prefisso `/leonardo`.

---

# 🔹 Endpoint: `POST /upload-and-parse-pdf/`

* **Cosa fa**:
  Riceve un file PDF, lo salva in `input_data/`, ne estrae il testo in formato Markdown tramite `DocumentManager`, e genera un nuovo file `.md` con il contenuto estratto.
  Restituisce JSON con nome file, percorso salvato, percorso del `.md` e un’anteprima del testo.

* **Dove salva**:

  * PDF caricato → `input_data/`
  * File `.md` estratto → `input_data/<nome>.md`
  * Immagini del PDF (se presenti) → `utilities/img_out/`

# 📦 Classe `DocumentManager`

Contiene i metodi usati dall’endpoint per gestire i documenti.

* **`read_local_pdf(file_path)`**
  Legge un PDF locale con PyMuPDF + pymupdf4llm, estrae testo in Markdown e salva eventuali immagini in `utilities/img_out/`.

* **`create_local_document(file_name, content)`**
  Crea un nuovo file di testo/Markdown in `input_data/` con il contenuto fornito.

* **`convert_images_to_base64(image_dir, img_ext="png")`**
  Converte immagini in base64 e le cancella dopo la conversione. Restituisce lista di dizionari con id, filename e contenuto base64.

* **`save_base64_to_image(base64_str, filename)`**
  Decodifica una stringa base64 e salva l’immagine in `input_data/`.

---

# 🔹 Endpoint: `POST /upload-images-to-mongo`

* **Cosa fa**
  Legge tutte le immagini presenti in `utilities/img_out/`, le converte in **base64** e le inserisce in **MongoDB** nella **db `Leonardo`**, **collection `images`**.
  Se trova in collection immagini con lo **stesso `filename`** del batch corrente, **non reinserisce** e avvisa che sono già presenti.

* **Dove prende/salva i dati**

  * **Input immagini**: `utilities/img_out/`
  * **DB di destinazione**: MongoDB → `Leonardo.images`
  * **Nota**: durante la conversione, i file in `img_out/` vengono **cancellati** (comportamento di `convert_images_to_base64`).

* **Passi principali**

  1. Costruisce il path `img_out/`.
  2. `DocumentManager.convert_images_to_base64(...)` → lista di `{image_id, filename, content_base64, timestamp}` e **rimozione file originali**.
  3. Usa `MongoManager._get_collection("Leonardo", "images")`.
  4. **Dedup**: verifica se esistono già documenti con lo **stesso `filename`** del primo elemento del batch.
  5. Se non esistono, inserisce tutta la lista con `MongoManager.write_to_mongo(...)`.
  6. Risponde con conteggio inseriti o messaggio “nessuna immagine trovata”.

# 📦 Classi e metodi coinvolti

## `DocumentManager` (da `utilities/dataloader.py`)

* **`convert_images_to_base64(image_dir, img_ext="png")`**
  Converte tutte le immagini della cartella in **base64**, produce una lista di dict (id, filename, contenuto, timestamp) e **cancella i file** sorgente dopo la conversione.

## `MongoManager` (da `utilities/mongodb.py`)

* **`_get_collection(database_name, collection_name)`**
  Restituisce l’oggetto collection (`Leonardo.images` in questo endpoint).
* **`write_to_mongo(data, database_name, collection_name)`**
  Inserisce un **dict** o una **lista di dict** (qui: la lista delle immagini base64).
* *(usato inline)* `collection.find({"filename": filename})` per controllo duplicati.

---

Ecco la versione sintetica per il **terzo endpoint** e le utility usate.

---

# 🔹 Endpoint: `POST /chunking`

* **Cosa fa**
  Riceve un file **Markdown (.md)**, verifica che esista il **PDF omonimo** in `input_data/`, sceglie **chunk\_size/overlap** in base al **numero di pagine** del PDF, segmenta il testo in **chunk semantici**, li trasforma in `Document` (con metadati), e salva tutto in **MongoDB** (`Leonardo.documents`). Evita doppi inserimenti se il `filename` è già presente.

* **Dove legge/salva**

  * Input: file `.md` (upload); PDF atteso in `input_data/<nome>.pdf`
  * Output su Mongo: DB **Leonardo**, collection **documents**
  * Risposta: anteprima (primi 10) + conteggio totale

* **Passi principali**

  1. **Validazione** estensione `.md` e **presenza PDF** omonimo in `input_data/`.
  2. **Conta pagine** PDF (pypdf) → imposta dinamicamente `chunk_size/overlap`.
  3. **Dedup** su Mongo (`Leonardo.documents`) per `filename`.
  4. **Split** markdown (`split_markdown_text`).
  5. **Create** `Document` (`create_documents`) con:

     * contenuto pulito,
     * metadata: `chunk_no`, `images` (link immagine estratti).
  6. **Persistenza** su Mongo con `mongo_manager.write_to_mongo`.
  7. **Response**: `total_chunks`, `chunks` (prime 10 stringhe), `docs` (prime 10 serializzate senza `_id/file_id`).


# 📦 Utility coinvolte (da `utilities/chunking.py`)

* **`split_markdown_text(text, chunk_size, chunk_overlap)`**
  Split ricorsivo con `RecursiveCharacterTextSplitter` usando separatori markdown (titoli, paragrafi, righe, parole) per chunk **coesi**.

* **`clean_markdown(text)`**
  Normalizza: rimuove titoli/markup **markdown** superfluo, caratteri di controllo, righe vuote multiple, simboli non informativi.

* **`create_documents(chunks)`**
  Per ogni chunk:

  * estrae link immagini `![](...)` → li mette in `metadata["images"]`,
  * rimuove i link dal testo,
  * applica `clean_markdown`,
  * crea `Document(page_content, metadata={"chunk_no": i, "images": [...]})`.


# 🗃️ MongoDB (da `utilities/mongodb.py`)

* **`_get_collection("Leonardo", "documents")`** per la collection.
* **`write_to_mongo(data, "Leonardo", "documents")`** salva lista di documenti.
* Query **dedup**: `collection.find({"filename": file.filename})`.


# 🧾 Output (esempio)

```json
{
  "total_chunks": 128,
  "chunks": ["<chunk_0>", "..."],
  "docs": [
    {
      "filename": "report.md",
      "document_id": "uuid",
      "page_content": "<testo pulito>",
      "metadata": {"chunk_no": 0, "images": ["utilities/img_out/p1_01.png"]},
      "uploaded_at": "2025-09-09T08:30:00.000000"
    }
  ]
}
```

**Nota:** il PDF deve trovarsi in `input_data/` con lo **stesso nome** del `.md` (es. `report.md` ↔ `report.pdf`) per stimare i parametri di chunking.


---

# 🔹 Endpoint: `POST /embeddings`

* **Cosa fa**
  Per un dato `file_name`:

  1. legge da **MongoDB → Leonardo.documents** tutti i chunk (`page_content`, `metadata`) con `filename = file_name`;
  2. calcola gli **embedding testo** dei `page_content` tramite **JinaEmbeddings** (modello già caricato nel `lifespan`);
  3. crea/usa in **Qdrant** la collection `hitachi` (distance **COSINE**) con **dimensionalità** = len(embedding);
  4. fa **upsert** dei punti in Qdrant con `payload = metadata` (+ `filename`), 1 punto per chunk.

* **Dove legge/salva**

  * **Lettura**: MongoDB → DB `Leonardo`, collection `documents`
  * **Scrittura**: Qdrant → collection `hitachi`
  * **Vettori**: dimensione determinata dal modello (es. 2048 per `jina-embeddings-v4`)

* **Output**

  ```json
  {
    "message": "<N> embeddings con dimensionalità <D> sono stati inseriti correttamente nella collection",
    "collection_name": "hitachi",
    "vector_size": <D>
  }
  ```

* **Note operative**

  * L’upsert usa **UUID nuovi** per ogni chunk → non sovrascrive inserimenti precedenti dello stesso file (no dedup naturale a meno di gestire ID stabili).
  * `metadata` include almeno `chunk_no` e `filename`.

---

# 📦 Utility coinvolte (da `utilities/embeddings.py`)

## `JinaEmbeddings`

Wrapper compatibile LangChain per il modello Jina (testo/immagini).

* **`embed_documents(texts: list[str]) -> list[list[float]]`**
  Embedding **batch** di testi per **retrieval** (`prompt_name="passage"`).
* **`embed_query(text: str) -> list[float]`**
  Embedding di **query** (`prompt_name="query"`).
* **`embed_images(image_paths: list[str]) -> list[list[float]]`**
  Embedding immagini (non usato in questo endpoint).

## `get_embedding_model(token: str)`

Carica **`jinaai/jina-embeddings-v4`** (Hugging Face) con `trust_remote_code=True`, su **CUDA** se disponibile, altrimenti CPU.
Ritorna l’oggetto modello usato da `JinaEmbeddings`.

---

# 🗃️ Mongo & Qdrant

* **MongoManager**

  * `read_from_mongo(query={"filename": file_name}, output_format="object", db="Leonardo", coll="documents")` → recupero chunk e metadati.

* **QdrantClient**

  * `collection_exists` / `create_collection(VectorParams(size=D, distance=COSINE))`
  * `upsert(points=[PointStruct(id=uuid4, vector=embedding, payload=metadata)])`

---

# 🔁 Flusso minimo

`Mongo (chunks) → JinaEmbeddings(embed_documents) → Qdrant(create_collection if needed) → upsert(points)`


---

# 🔹 Endpoint: `POST /retriever`

* **Cosa fa**
  Dato `file_name` e una `query`:

  1. filtra in **Qdrant** i vettori del solo documento (`Filter` su `filename`),
  2. calcola l’embedding della query con **JinaEmbeddings** e recupera i **top-5** chunk più simili,
  3. ricostruisce il **contesto** testuale dai chunk salvati in **MongoDB** e raccoglie le **immagini** associate,
  4. invoca un **LLM locale** (Ollama `llama3.1:8b`) con prompt *context-aware* per generare la risposta,
  5. scarica da Mongo le immagini correlate (base64) e le **salva su disco** per la risposta.

* **Dove legge/salva**

  * **Vettori**: Qdrant → collection `hitachi` (filtrata per `filename`)
  * **Chunk & immagini (base64)**: MongoDB → `Leonardo.documents` e `Leonardo.images`
  * **Salvataggio immagini decodificate**: `utilities/retrieved_images/`
  * **LLM**: Ollama su `http://host.docker.internal:11434`

* **Output**

  ```json
  {
    "answer": "<risposta LLM basata sul contesto>",
    "images": ["utilities/retrieved_images/img1.png", "..."]
  }
  ```

# 🧩 Passi principali

1. **Filtro per documento**

   ```python
   query_filter = Filter(must=[FieldCondition(key="filename", match=MatchValue(value=file_name))])
   ```

   Limita la ricerca ai soli vettori del file.

2. **Prompt & LLM**

   * Prompt `ChatPromptTemplate`: “Answer the question based only on the following context: {context} …”
   * LLM: `ChatOllama(model="llama3.1:8b", temperature=0.2)`

3. **Pipeline LangChain**

   * `retriever_ji_full`: `RunnableLambda` che chiama il **retriever** e restituisce `(context, images, query)`
   * `split`: `RunnableMap` per comporre il dict `{context, images, question}`
   * `chain_model`: `prompt | llm | StrOutputParser()`
   * `full_chain`: esegue retrieve → genera `answer` → ritorna `{"answer", "images"}`

4. **Download immagini**

   * Per ciascun `image_name` recuperato dal contesto:

     * legge `content_base64` da `Leonardo.images`
     * decodifica e salva in `utilities/retrieved_images/`
     * accumula i path salvati per la risposta


# 📦 Utility coinvolte

## `retriever_jina(query: str, model: AutoModel, query_filter: Filter)`

* Embedding **query** con `JinaEmbeddings.embed_query`.
* `QdrantClient.query_points(...)` con `SearchParams(hnsw_ef=128, exact=False)`, `limit=5`, `with_payload=True`.
* Per ogni match:

  * legge da Mongo `Leonardo.documents` il `page_content` e `metadata.images` corrispondenti a `chunk_no` e `filename`.
* Ritorna:

  * `full_content` = join dei contenuti chunk,
  * `full_images` = lista piatta delle immagini collegate.

## `MongoManager`

* `read_from_mongo(...)` per:

  * `documents`: ricostruire il contesto testuale e recuperare `images`
  * `images`: ottenere `content_base64` per il salvataggio su disco

# 🔁 Flusso minimo

`query → embed_query → Qdrant (filter by filename) → top-5 chunk → Mongo (contenuti + immagini) → LLM con context → decodifica immagini da Mongo → salva in utilities/retrieved_images → {"answer", "images"}`

---

# Esempio di flusso di esecuzione 

Ecco un esempio di**flusso end-to-end** di tutti gli endpoint, in ordine, con **input e output di esempio**.

---

# 1) Carica PDF → estrai testo + immagini

`POST /upload-and-parse-pdf/`

```bash
curl -X POST http://localhost:8091/upload-and-parse-pdf/ \
  -H "accept: application/json" \
  -F "file=@/path/Report_Tech.pdf"
```

**Effetti**

* Salva il PDF in `input_data/Report_Tech.pdf`
* Estrae testo → salva `input_data/Report_Tech.md`
* Estrae immagini → salva in `utilities/img_out/` (png)

**Output (esempio)**

```json
{
  "filename": "Report_Tech.pdf",
  "file_path": "/app/input_data/Report_Tech.pdf",
  "new_md_file": "/app/input_data/Report_Tech.md",
  "parsed_text": "# Executive Summary...\n(anteprima 500 char)"
}
```

---

# 2) Carica immagini estratte in MongoDB

`POST /upload-images-to-mongo`

```bash
curl -X POST http://localhost:8091/upload-images-to-mongo \
  -H "accept: application/json"
```

**Effetti**

* Legge `utilities/img_out/*.png`
* Converte in base64 e **inserisce** in `MongoDB → Leonardo.images`
* (Le immagini locali vengono **cancellate** dopo la conversione)

**Output (esempio)**

```json
{
  "inserted_count": 7,
  "message": "7 images successfully inserted to mongoDB"
}
```

*(Se già presenti per lo stesso filename: `{"message":"Immagini dello stesso file p1_01.png già presenti in MongoDB"}`)*

---

# 3) Chunking del Markdown → salvataggio su Mongo

`POST /chunking`

> ⚠️ Richiede che esista il **PDF omonimo** in `input_data/` (es. `Report_Tech.pdf`).

```bash
curl -X POST http://localhost:8091/chunking \
  -H "accept: application/json" \
  -F "file=@/path/Report_Tech.md"
```

**Effetti**

* Calcola `chunk_size/overlap` in base alle **pagine del PDF**
* Segmenta testo → crea `Document` con `metadata={chunk_no, images}`
* Inserisce tutto in `MongoDB → Leonardo.documents`

**Output (esempio)**

```json
{
  "total_chunks": 124,
  "chunks": [
    "Introduzione\nIl presente documento...",
    "Specifiche tecniche principali...",
    "... (fino a 10 anteprime) ..."
  ],
  "docs": [
    {
      "filename": "Report_Tech.md",
      "document_id": "f8f6c1a1-7a16-4a1f-92a2-3a8f2d2a8c21",
      "page_content": "Introduzione Il presente documento...",
      "metadata": { "chunk_no": 0, "images": ["utilities/img_out/p1_01.png"] },
      "uploaded_at": "2025-09-09T08:30:00.000000"
    }
  ]
}
```

---

# 4) Embeddings → upsert su Qdrant

`POST /embeddings`

```bash
curl -X POST "http://localhost:8091/embeddings?file_name=Report_Tech.md" \
  -H "accept: application/json"
```

**Effetti**

* Legge i chunk da `Leonardo.documents` (filtrati per `filename`)
* Calcola embeddings testo con **JinaEmbeddings**
* Crea (se serve) la collection `hitachi` (distance COSINE, dim es. 2048)
* **Upsert** dei vettori con payload = metadata + filename

**Output (esempio)**

```json
{
  "message": "124 embeddings con dimensionalità 2048 sono stati inseriti correttamente nella collection",
  "collection_name": "hitachi",
  "vector_size": 2048
}
```

---

# 5) Retrieval + Answering + salvataggio immagini correlate

`POST /retriever`

```bash
curl -X POST "http://localhost:8091/retriever?file_name=Report_Tech.md&query=Quali sono le specifiche della pompa idraulica?" \
  -H "accept: application/json"
```

**Effetti**

* Filtra in Qdrant per `filename=Report_Tech.md`, `top-5` chunk simili
* Ricostruisce **context** dai chunk in `Leonardo.documents`
* Invoca **Ollama** (`llama3.1:8b`) con prompt basato sul solo **context**
* Recupera immagini collegate da `Leonardo.images` (base64) → **salva file** in `utilities/retrieved_images/`

**Output (esempio)**

```json
{
  "answer": "La pompa idraulica è di tipo a palette con portata nominale 45 l/min a 3000 rpm, pressione massima 210 bar, efficienza volumetrica 92%.",
  "images": [
    "utilities/retrieved_images/p1_01.png",
    "utilities/retrieved_images/p2_03.png"
  ]
}
```

---

## 🧭 Riassunto rapido del flusso

1. **PDF → md + immagini**
   `/upload-and-parse-pdf/` → `input_data/Report_Tech.md` + `utilities/img_out/*.png`
2. **Immagini → Mongo (base64)**
   `/upload-images-to-mongo` → `Leonardo.images`
3. **Chunk md → Mongo**
   `/chunking` → `Leonardo.documents` (Document per chunk)
4. **Embeddings → Qdrant**
   `/embeddings` → `hitachi` (vettori + payload)
5. **Query → Retrieve + LLM + immagini salvate**
   `/retriever` → risposta + `utilities/retrieved_images/*.png`

---

## 🛠️ Tecnologie utilizzate

* **FastAPI**: per il backend API
* **MongoDB**: storage documentale
* **Qdrant**: vector store per embeddings
* **Jina Embeddings v4**: modello per la vettorializzazione
* **LangChain**: orchestrazione di componenti AI
* **Ollama**: serving locale del modello `qwen2.5:7b`
* **PyMuPDF / PIL / pypdf**: parsing PDF e immagini
