from utilities.jina_embeddings import JinaEmbeddings, get_embedding_model
from utilities.mongodb import MongoManager
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

# metodo modificato per test
model = get_embedding_model(token="hf...")

embedding_manager = JinaEmbeddings(model=model)

mongo_manager = MongoManager(connection_string="mongodb://localhost:27017")

documents = mongo_manager.read_from_mongo(query={"filename": "414759-1-_5_Nike-NPS-Combo_Form-10-K_WR.md"}, output_format="object", database_name="Leonardo", collection_name="documents")

print(len(documents))


document_contents = []
document_metadata = []
for document in documents:
    page_content = document.get("page_content", "")
    metadata = document.get("metadata", {})

    document_contents.append(page_content)
    document_metadata.append(metadata)

if len(document_contents) != len(document_metadata):
    raise Exception("Contents e metadata non coincidono!")


# calcola gli embeddings
embeddings = embedding_manager.embed_documents(document_contents)


print(f"Numero di embeddings: {len(embeddings)}, dimensionalità : {len(embeddings[0])}")


# connessione a Qdrant (Docker Locale)
client = QdrantClient(url="http://localhost:6333")

# creiamo la collection se già non eseiste
collection_name = "leonardo"
vector_size = len(embeddings[0]) # 2048

if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

# i PointStruct rappresentano i singoli documenti nel vectorstore, contiene:
# id: identificativo univoco del punto
# vector: lista di numeri float che rappresenta l'embedding
# payload: (Opzionale) dato addizionali in formato JSON come metadati o testo associato

# questo consente di combinare vettori con filtri sul payload durante le query
# legano embed + metadata in un'unica entità gestibile

# prepariamo i points per i nostri embeddings
points = [
    PointStruct(
        id=i,
        vector=embeddings[i],
        payload=document_metadata[i]
    )
    for i in range(len(embeddings))
]


# inseriamo o aggiorniamo i vettori
# se hanno gli stessi ID si aggiornano evitando duplicati
client.upsert(
    collection_name=collection_name,
    points=points
)



