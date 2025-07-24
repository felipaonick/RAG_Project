from pathlib import Path
import pymupdf
import pymupdf4llm
from utilities.dataloader import DocumentManager
from utilities.chunking import clean_markdown


doc_manager = DocumentManager()

file_path = "C:/Users/felip/Desktop/import-pc/Leonardo_RAG/input_data/GPLM-MAN-037.pdf"

# Parsa il PDF usando DocumentManager
parsed_text = doc_manager.read_local_pdf(str(file_path))

# Estrai il nome base senza estensione
nome_file = Path(file_path).stem

# Scrive file .md con il testo estratto
new_md_file = doc_manager.create_local_document(file_name=f"{nome_file}.md", content=parsed_text)

print(new_md_file)


