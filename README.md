
## 📚 Leonardo API – Documentazione degli Endpoint

Tutti gli endpoint sono accessibili con il prefisso `/leonardo`.

---

### 📄 `POST /leonardo/upload-and-parse-pdf/`

**Descrizione:**
Carica un file PDF, lo salva in locale e ne estrae il contenuto testuale in formato Markdown.
Il testo estratto viene salvato come `.md`.

**Input:**

* `file` (form-data, `UploadFile`): File `.pdf` da elaborare.

**Output:**

* `filename`, `file_path`, `new_md_file`, `parsed_text` (anteprima dei contenuti).

---

### 🖼️ `POST /leonardo/upload-images-to-mongo`

**Descrizione:**
Converte in Base64 tutte le immagini presenti nella cartella `img_out` e le salva nella collection `images` del database MongoDB `Leonardo`.

**Output:**

* Numero di immagini inserite, oppure messaggio se le immagini sono già presenti o assenti.

---

### ✂️ `POST /leonardo/chunking`

**Descrizione:**
Esegue lo **split semantico** di un file `.md` in chunk ottimizzati per l'elaborazione LLM, con metadati e gestione delle immagini.

**Input:**

* `file` (form-data, `UploadFile`): File `.md` da segmentare.

**Output:**

* `total_chunks`: numero di chunk generati
* `chunks`: anteprima delle stringhe
* `docs`: anteprima dei documenti (`Document`) con metadati

**Note:**

* Se il file PDF associato non è presente, restituisce errore `400`.
* Evita duplicati in MongoDB tramite verifica preventiva.

---

### 🧠 `POST /leonardo/embeddings`

**Descrizione:**
Genera embedding vettoriali (Jina Embeddings v4) dai documenti associati a un determinato file e li salva in **Qdrant** (`collection: hitachi`).

**Input:**

* `file_name` (query param, `str`): Nome del file di riferimento per recuperare i chunk da MongoDB.

**Output:**

* Messaggio di conferma con numero e dimensionalità degli embeddings inseriti in Qdrant.

---

### ❓ `POST /leonardo/retriever`

**Descrizione:**
Risponde a una query usando un meccanismo **RAG + immagini**.
Recupera i chunk rilevanti da Qdrant e genera una risposta con un LLM (`qwen2.5:7b`), integrando eventuali immagini collegate.

**Input:**

* `file_name` (query param, `str`): Nome del file da interrogare
* `query` (query param, `str`): Domanda dell’utente

**Output:**

* `answer`: Risposta generata
* `images`: Lista di immagini collegate salvate localmente

**Note:**

* Supporta risposte multimodali (testo + immagini).
* Utilizza pipeline asincrona con LangChain.

---

## 🛠️ Tecnologie utilizzate

* **FastAPI**: per il backend API
* **MongoDB**: storage documentale
* **Qdrant**: vector store per embeddings
* **Jina Embeddings v4**: modello per la vettorializzazione
* **LangChain**: orchestrazione di componenti AI
* **Ollama**: serving locale del modello `qwen2.5:7b`
* **PyMuPDF / PIL / pypdf**: parsing PDF e immagini
