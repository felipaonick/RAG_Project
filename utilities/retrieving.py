from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, SearchParams
from langchain_core.embeddings import Embeddings
from utilities.mongodb import MongoManager
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel, RunnableMap, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding
from qdrant_client import models
from pathlib import Path
from typing import Optional, List, Dict
import base64
from PIL import Image
from typing import Tuple, List
from io import BytesIO
import os
from transformers import AutoModel

# connessione a Qdrant (Docker Locale)
client = QdrantClient(url="http://localhost:6333")

mongo_manager = MongoManager(connection_string="mongodb://localhost:27017")

# con parametri di ricerca exact=True 
# 🔍 Ricerca con exact=True
# full kNN search: Qdrant confronta la query con ogni singolo vettore presente nella collection, restituendo i risultati esatti basati su distanza o similarità 

# Viene ignorato completamente l’indice HNSW; niente trade-off tra velocità e accuratezza: l’accuratezza è massima, ma si paga in performance 
# Qdrant
# .

# Utile per:

# Benchmark: per valutare la qualità di una ricerca ANN confrontandola con quella esatta 
# Qdrant
# ;

# Collection piccole: la differenza di prestazioni è minima, ma la precisione è totale.

# ⚠️ Trade-off
# Metodo	Precisione	Velocità	Uso consigliato
# exact=True	Massima	Molto lenta	Benchmark, collezioni piccole
# exact=False + hnsw_ef	Alta (ma approssimata)	Molto più veloce	Produzione, grandi dataset



# ## 🧠 Cosa fa realmente `hnsw_ef=128`

# 1. **Inizia la ricerca** partendo da un punto di ingresso nel livello più alto del grafo.
# 2. **Scende gradualmente** ai livelli inferiori mantenendo fino a **128 candidati** potenzialmente migliori.
# 3. Ad ogni livello:

#    * **Valuta la distanza** tra la query e i nodi candidati.
#    * **Aggiorna la coda** (priority queue) mantenendo sempre i migliori 128 candidati.
# 4. Alla fine, tra questi candidati, viene scelto il numero di risultati richiesti (`limit`).

# Quindi `128` è la **dimensione massima della coda**, non il numero limitato di vettori finali. Serve a **bilanciare qualità e prestazioni** ([qdrant.tech][1], [medium.com][2]).

# ---

# ## ⚖️ Vantaggi e svantaggi

# * ✅ **Corectness maggiore**: più alto è `ef`, maggiore è la probabilità che tra i candidati ci siano i vicini più vicini.
# * 🔄 **Costi normali**: un `ef` alto richiede più CPU e tempi di risposta maggiori, ma non scala linearmente con la dimensione della collezione.
# * 🗺️ **Ricerca efficiente**: la complessità è limitata, non vengono confrontati *tutti* i vettori.

# ---

# ## 🔍 Esempio operativo

# * **Scenario**: cerca tra 1 milione di vettori, chiedi i top 5:

#   * Con `hnsw_ef=32` → coda fino a 32 candidati → rapido ma meno preciso.
#   * Con `hnsw_ef=128` → coda fino a 128 → più accurato a scapito di più calcoli.
#   * Alla fine vengono restituiti solo i top `limit` (es. 5), ma la selezione avviene tra quei 128 candidati.

# ---

# ## ✅ In sintesi

# * `hnsw_ef` controlla **quanta profondità esplorare** nel grafo.
# * Non limita i risultati, ma **ammassa i migliori candidati prima di selezionare i finali**.
# * Serve a regolare il trade-off **accuratezza vs velocità/risorse**.

def retriever_generic(
        query: str, 
        embeddings: Embeddings, 
        query_filter: Filter, 
        collection_name: str,
        hyde_text: Optional[str] = None
        ):
    print(f"[DEBUG] [retriever] Query: {query}")

    # 1) Embedding della query (provider-agnostic)
    # usa HyDE se presente, altrimenti emb della query
    emb_query = embeddings.embed_query(hyde_text) if hyde_text is not None else embeddings.embed_query(query)
    print(f"[DEBUG] [retriever] Embedding query dim: {len(emb_query)}")

    # 2) Ricerca in Qdrant con filtro (es. per filename)

    scored_points = client.query_points(
        collection_name=collection_name,
        query=emb_query,
        query_filter=query_filter,
        search_params=SearchParams(hnsw_ef=128, exact=False),
        limit=5,
        with_payload=True
    )
    print(f"[DEBUG] [retriever] Punti trovati in Qdrant: {len(scored_points.points)}")

    contents: List[dict] = []
    seen = set()

    for pt in scored_points.points:
        chunk_no = pt.payload.get("chunk_no")
        filename = pt.payload.get("filename")
        print(f"[DEBUG] [retriever] Match -> chunk_no={chunk_no}, filename={filename}")

        if chunk_no is None or filename is None:
            print("[WARNING] [retriever] Payload senza chunk_no/filename, skip")
            continue

        key = (filename, chunk_no)
        if key in seen:
            continue
        seen.add(key)

        # 3) Fetch chunk da Mongo
        doc = mongo_manager.read_from_mongo(
            query={"metadata.chunk_no": chunk_no, "filename": filename},
            output_format="object",
            database_name="Leonardo",
            collection_name="documents",
        )

        if not doc:
            print(f"[WARNING] [retriever] Nessun doc in Mongo per {key}")
            continue

        content = doc[0].get("page_content", "") or ""
        images = doc[0].get("metadata", {}).get("images", []) or []
        print(f"[DEBUG] [retriever] Testo len={len(content)}, #img={len(images)}")

        contents.append({"page_content": content, "images": images})

    # 4) Aggregazione
    full_images = [img for c in contents for img in c.get("images", [])]
    full_content = "\n".join(c.get("page_content", "") for c in contents)

    print(f"[DEBUG] [retriever] Context totale len={len(full_content)}")
    print(f"[DEBUG] [retriever] Immagini totali={len(full_images)}")

    return full_content, full_images


def retriever_hybrid(
        query: str, 
        dense_embedding_model: Embeddings, 
        sparse_embedding_model: SparseTextEmbedding,
        late_interaction_embedding_model: LateInteractionTextEmbedding, 
        collection_name: str, 
        rerank: bool,
        hyde_text: Optional[str] = None
        ):
    print(f"[DEBUG] [retriever] Query: {query}")

    # 1) Embedding della query (provider-agnostic)
    dense_emb_query = dense_embedding_model.embed_query(hyde_text) if hyde_text is not None else dense_embedding_model.embed_query(query)

    sparse_emb_query = next(sparse_embedding_model.query_embed(hyde_text)) if hyde_text is not None else next(sparse_embedding_model.query_embed(query))

    late_emb_query = next(late_interaction_embedding_model.query_embed(hyde_text)) if hyde_text is not None else next(late_interaction_embedding_model.query_embed(query))

    # 2) Ricerca in Qdrant con filtro (es. per filename)

    # Recupera i nomi dei modelli, gestendo le diverse classi
    dense_model_name = getattr(dense_embedding_model, "model", None) or getattr(dense_embedding_model, "model_name", None)

    prefetch = [
        models.Prefetch(
            query=dense_emb_query,
            using=dense_model_name,
            limit=10
        ),
        models.Prefetch(
            query=models.SparseVector(**sparse_emb_query.as_object()),
            using="bm25",
            limit=10
        )
    ]

    if rerank:
         # usiamo il modello collbert per fare il rerank multivector più granulare
        results = client.query_points(
            collection_name=collection_name,
            query=late_emb_query, # query multivector con colbert
            using="colbertv2.0",
            prefetch=prefetch,
            #query_filter=query_filter,
            with_payload=True,
            limit=50
        )
    else:
        # usiamo Reciprocal Rank Fusion RRF per fondere i risultati dalle due classifiche dense e sparse
        results = client.query_points(
            collection_name=collection_name,
            query=models.FusionQuery(
                fusion=models.Fusion.RRF
            ),
            prefetch=prefetch,
            #query_filter=query_filter,
            with_payload=True,
            limit=50
        )


    print(f"[DEBUG] [retriever] Punti trovati in Qdrant: {len(results.points)}")

    contents: List[dict] = []
    seen = set()

    for pt in results.points:
        chunk_no = pt.payload.get("chunk_no")
        filename = pt.payload.get("filename")
        print(f"[DEBUG] [retriever] Match -> chunk_no={chunk_no}, filename={filename}")

        if chunk_no is None or filename is None:
            print("[WARNING] [retriever] Payload senza chunk_no/filename, skip")
            continue

        key = (filename, chunk_no)
        if key in seen:
            continue
        seen.add(key)

        # 3) Fetch chunk da Mongo
        doc = mongo_manager.read_from_mongo(
            query={"metadata.chunk_no": chunk_no, "filename": filename},
            output_format="object",
            database_name="Leonardo",
            collection_name="documents",
        )

        if not doc:
            print(f"[WARNING] [retriever] Nessun doc in Mongo per {key}")
            continue

        content = doc[0].get("page_content", "") or ""
        images = doc[0].get("metadata", {}).get("images", []) or []
        print(f"[DEBUG] [retriever] Testo len={len(content)}, #img={len(images)}")

        contents.append({"page_content": content, "images": images})

    # 4) Aggregazione
    full_images = [img for c in contents for img in c.get("images", [])]
    full_content = "\n".join(c.get("page_content", "") for c in contents)

    print(f"[DEBUG] [retriever] Context totale len={len(full_content)}")
    print(f"[DEBUG] [retriever] Immagini totali={len(full_images)}")

    return full_content, full_images


