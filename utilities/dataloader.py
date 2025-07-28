from pathlib import Path
import pymupdf
import pymupdf4llm
import base64
from datetime import datetime 
from uuid import uuid4



class DocumentManager:
    def __init__(self):
        """Inizializza il DocumentManager"""

        # Path dinamico valido anche dentro Docker
        self.img_path = Path(__file__).resolve().parent.parent / "utilities" / "img_out"
        self.img_path.mkdir(parents=True, exist_ok=True)

    # metodi per documenti locali
    def read_local_pdf(self, file_path: str):
        """Legge e processa un file pdf locale."""

        path_file = Path(file_path)

        # verifica se esits eed è un file
        if not path_file.is_file():
            return f"File non trovato: {file_path}"
        
        suffix = path_file.suffix

        if suffix.lower() == ".pdf":

            # Processa PDF utilizzando PyMuPDF4llm
            try:

                doc = pymupdf.open(file_path)  # use a Document for subsequent processing

                page_dicts = pymupdf4llm.to_markdown(
                    doc,
                    write_images=True,
                    page_chunks=True,
                    image_path=str(self.img_path),
                    image_format="png",
                    show_progress=True,
                )

                print(f"Immagini del documento salvate in {str(self.img_path)}")

                # colleghiamo il testo di ogni pagina
                full_text = "\n".join([page['text'] for page in page_dicts if 'text' in page])
                
                return full_text
            
            except Exception as e:
                
                return f"Errore nella lettura del PDF: {e}"
            
    def convert_images_to_base64(self, image_dir: str, img_ext: str = "png") -> list[dict]:
        """
        Converte tutte le immagini nella cartella `image_dir` in stringhe base64.
        Ritorna una lista di dizionari: [{"filename": ..., "content_base64": ...}, ...]
        """
        image_dir = Path(image_dir)
        base64_images = []

        if not image_dir.exists():
            print(f"Cartella non trovata: {image_dir}")
            return []

        for image in image_dir.glob(f"*.{img_ext}"):
            try:
                with open(image, "rb") as image_file:
                    encoded = base64.b64encode(image_file.read()).decode("utf-8")
                    image_bytes = image_file.read()

                    base64_images.append({
                        "image_id": str(uuid4()),
                        "filename": image.name,
                        "content_base64": encoded,
                        "timestamp": datetime.utcnow()
                    })

                # Elimina l'immagine originale dopo la conversione
                image.unlink()
                print(f"Convertita in base64 e cancellata: {image.name}")

            except Exception as e:
                print(f"Errore su {image.name}: {e}")

        print(f"Convertite e cancellate {len(base64_images)} immagini")
        return base64_images

    def save_base64_to_image(self, base64_str: str, filename: str):
        """Salva una stringa base64 come immagine PNG"""
        try:
            image_data = base64.b64decode(base64_str)

           # Calcola il path assoluto di input_data partendo da /utilities
            base_dir = Path(__file__).resolve().parent.parent  # va da /utilities → root progetto
            input_data_dir = base_dir / "input_data"
            input_data_dir.mkdir(parents=True, exist_ok=True)  # crea se non esiste

            output_path = input_data_dir / filename

            with open(output_path, "wb") as f:
                f.write(image_data)

            print(f"Immagine salvata in {output_path}")
        except Exception as e:
            print(f"Errore nella scrittura dell'immagine: {e}")



    # metodo per scrivere documenti
    def create_local_document(self, file_name: str, content: str):
        """Crea un documento locale con il contenuto fornito."""

        try:
             # Costruisce il path relativo dalla cartella utilities
            base_dir = Path(__file__).resolve().parent.parent  # va da /utilities → root progetto
            input_data_dir = base_dir / "input_data"
            file_path = input_data_dir / file_name

            # Crea eventuali cartelle mancanti
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            return f"{str(file_path)}"
        except Exception as e:
            return f"Errore nella creazione del file: {e}"     


if __name__ == "__main__":


    # inizializza in DocumentManager
    doc_manager = DocumentManager()

    # parsiamo un pdf locale
    file_path = "C:/Users/felip/Desktop/import-pc/Leonardo_RAG/input_data/414759-1-_5_Nike-NPS-Combo_Form-10-K_WR.pdf"

    text = doc_manager.read_local_pdf(file_path=file_path)

    print(text[:500])

    new_file = doc_manager.create_local_document("./new_file.md", content=text)

    print(new_file)

    images_base64 = doc_manager.convert_images_to_base64("./img_out")

    print(images_base64[0])

    print(doc_manager.save_base64_to_image(images_base64[0]['content_base64'], "prova.png"))
