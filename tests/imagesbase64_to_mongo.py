import sys
from pathlib import Path

# Aggiungi la root del progetto al path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from utilities.dataloader import DocumentManager
from utilities.mongodb import MongoManager, WriteDataModel


doc_manager = DocumentManager()

mongo_manager = MongoManager(connection_string="mongodb://localhost:27017", default_database="Leonardo")



# parsiamo un pdf locale
file_path = "C:/Users/felip/Desktop/import-pc/Leonardo_RAG/input_data/414759-1-_5_Nike-NPS-Combo_Form-10-K_WR.pdf"

text = doc_manager.read_local_pdf(file_path=file_path)

print(text[:500])

img_dir = Path(__file__).resolve().parent.parent / "utilities" / "img_out"
images_base64 = doc_manager.convert_images_to_base64(str(img_dir))

print(images_base64[0])

write_data = WriteDataModel(collection_name="example_collection", data=images_base64[0])

result = mongo_manager.write_to_mongo(collection_name=write_data.collection_name, data=write_data.data)

print(result)

