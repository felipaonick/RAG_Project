from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, SearchParams
from utilities.embeddings import JinaEmbeddings, get_embedding_model
from utilities.mongodb import MongoManager
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel, RunnableMap, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
import base64
from PIL import Image
from io import BytesIO
import os

client = QdrantClient(url="http://localhost:6333")

mongo_manager = MongoManager(connection_string="mongodb://localhost:27017")


# ricerca con prefiltering per filtrare gli embedding che sono di un documento specifico

query_filter = Filter(must=[
    FieldCondition(
        key="filename",
        match=MatchValue(value="414759-1-_5_Nike-NPS-Combo_Form-10-K_WR.md")
    )
])


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


query = "Chi sono gli executive officers di Nike?"

model = get_embedding_model(token="hf...")

def retriever_jina(query: str):
    embeddings_manager = JinaEmbeddings(model)
    emb_query = embeddings_manager.embed_query(query)
    scored_points = client.query_points(
        collection_name="leonardo",
        query=emb_query,
        query_filter=query_filter,
        search_params=SearchParams(hnsw_ef=128, exact=False),
        limit=5,
        with_payload=True
    )

    contents = []
    #Serial per ricercare in Mongo
    for pt in scored_points.points:
        doc = mongo_manager.read_from_mongo(
            query={"metadata.chunk_no": pt.payload["chunk_no"], "filename": pt.payload["filename"]},
            output_format="object",
            database_name="Leonardo", collection_name="documents"
        ) # è una lista di dizionari
        content = doc[0]["page_content"]
        images = doc[0]["metadata"].get("images", [])

        contents.append({"page_content": content, "images": images})

    full_images = []
    for content in contents:
        if content["images"]:
            full_images.extend(content["images"])


    full_content = "\n".join(content.get("page_content", "") for content in contents)

    return full_content, full_images

# diamo in pasto al modello il contenuto dei documents retrievati
template = """Answer the question based only on the following contetext:
{context}

Question: {question}"""

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOllama(model="qwen2.5:7b", base_url="http://localhost:11434", temperature=0.3)

retriever_ji_full = RunnableLambda(lambda query: (*retriever_jina(query), query))

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

result = full_chain.invoke(query)

print(result)
# dobbiamo prelevare le immagini in base da mongodb
# dunque prendere solo il nome del file immagine del path

image_files = [Path(path).name for path in result['images']]

print({"images": image_files, "answer": result['answer']})

# retrive delle immagini da mongoDB
# Directory di destinazione
main_dir = Path(__file__).resolve().parent.parent
output_dir = main_dir / "utilities" / "retrieved_images"
os.makedirs(output_dir, exist_ok=True)
for image in image_files:
    img = mongo_manager.read_from_mongo(query={"filename": image}, output_format="object", database_name="Leonardo", collection_name="images")
    filename = img[0]['filename']
    content_base64 = img[0]['content_base64']

    image_data = base64.b64decode(content_base64)

    # Crea un oggetto "file in memoria"
    buffer = BytesIO(image_data)

    img = Image.open(buffer)
    # Salvo l’immagine nella directory specificata
    save_path = os.path.join(output_dir, filename)
    img.save(save_path)
    print(f"Saved {save_path}")

