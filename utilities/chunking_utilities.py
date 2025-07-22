# chunking.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import re

def split_markdown_text(text: str, chunk_size: int = 3000, chunk_overlap: int = 200) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n## ",    # Titoli di secondo livello
            "\n### ",   # Titoli di terzo livello
            "\n#### ",  # Titoli di quarto livello
            "\n\n",     # Paragrafi
            "\n",       # Righe
            " ",        # Parole
            ""          # Fallback carattere
        ]
    )
    return splitter.split_text(text)


def clean_markdown(text: str) -> str:
    # 1. Rimuove ritorni a capo stile Windows
    text = text.replace("\r\n", "\n")

    # 2. Elimina righe vuote multiple
    text = re.sub(r"\n{2,}", "\n", text)

    # 3. Rimuove grassetto e corsivo Markdown
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)

    # 4. Rimuove titoli Markdown (###, ##, #)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)

    # 5. Normalizza simboli non informativi
    text = text.replace("☑", "").replace("☐", "").replace("þ", "")

    # 6. Rimuove caratteri di controllo
    text = re.sub(r"[\x00-\x1F\x7F]", "", text)

    return text.strip()




def create_documents(chunks: list[str]) -> list[Document]:
    """
    Trasforma una lista di stringhe Markdown in oggetti Document con metadati.

    - Estrae link a immagini Markdown (![](...))
    - Rimuove le immagini dal contenuto
    - Pulisce whitespace e righe vuote
    - Aggiunge numero chunk e lista immagini come metadata
    """
    docs = []

    for i, chunk in enumerate(chunks):
        
        # estraggo le immagini dal markdown
        image_links = re.findall(r'!\[\]\((img_out/[^)]+)\)', chunk)

        # rimuovo i link immagine dal contenuto
        for link in image_links:
            chunk = chunk.replace(f"![]({link})", "")


        # crea Document con metadati
        doc = Document(
            page_content=clean_markdown(chunk), 
            metadata={
                "chunk_no": i, 
                "images": image_links
            }
        )

        docs.append(doc)


    return docs