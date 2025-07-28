# per prima cosa dobbiamo creare un wrapper per il nostro motore di embedding jina

from langchain_core.embeddings import Embeddings
from PIL import Image
from transformers import AutoModel
import torch


class JinaEmbeddings(Embeddings):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model


    # metodo abstract quindi necessario
    # funzione per emebeddare una lista di testi come i chunks
    def embed_documents(self, texts):
        return self.model.encode_text(
            texts=texts,
            task="retrieval",
            prompt_name="passage",
            return_numpy=True
        ).tolist()

    # metodo abstract quindi necessario
    # funzione per embeddare una singola quesry di testo
    def embed_query(self, text):
        return self.model.encode_text(
            texts=[text],
            task="retrieval",
            prompt_name="query",
            return_numpy=True
        )[0].tolist()


    # funzione per emebeddare una lista di immagini
    def embed_images(self, image_paths):
        images = [Image.open(path).convert("RGB") for path in image_paths]
        return self.model.encode_image(
            images=images,
            task="retrieval",
            return_numpy=True
        ).tolist()
    


def get_embedding_model(token: str):

    if not token:
        raise RuntimeError("Non sei autenticato con HugginFace")
    
    # Verifica disponibilità CUDA
    print("Torch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    
    model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v4",
        token=token,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True
    )

    return model