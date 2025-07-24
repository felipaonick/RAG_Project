import json
from pymongo import MongoClient
from pydantic import BaseModel, Field
from typing import Optional, Any


# Modelli Pydantic per operazioni MongoDB
class WriteDataModel(BaseModel):
    database_name: Optional[str] = Field(None, title="Database Name", description="Nome del database.")
    collection_name: Optional[str] = Field(None, title="Collection Name", description="Nome della collection.")
    data: Any = Field('{"abc": 123}', title="Data", description="Stringa Json dei dati da inserire nella collection.")


class ReadDataModel(BaseModel):
    database_name: Optional[str] = Field(None, title="Database Name", description="Nome del database.")
    collection_name: Optional[str] = Field(None, title="Collection Name", description="Nome della collection.")
    query: Optional[Any] = Field(default="{}", title="Query", description="Query per il recupero dei dati.")


class DeleteDataModel(BaseModel):
    database_name: Optional[str] = Field(None, title="Database Name", description="Nome del database.")
    collection_name: Optional[str] = Field(None, title="Collection Name", description="Nome della collection.")
    query: Any = Field("{}", title="Query", description="Query per eliminare i dati.")


class UpdateDataModel(BaseModel):
    database_name: Optional[str] = Field(None, title="Database Name", description="Nome del database.")
    collection_name: Optional[str] = Field(None, title="Collection Name", description="Nome della collection.")
    query: Any = Field("{}", title="Query", description="Query per aggiornare i dati.")
    new_values: Any = Field(..., title="New Values", description="Nuovi valori per l'aggiornamento.")



class MongoManager:
    def __init__(self, connection_string: str, default_database: str = "default_db", default_collection: str = "default_collection"):
        """Inizializza MongoManager con una connection string e opzionalmente un database e collection di default."""
        self.client = MongoClient(connection_string)
        self.default_database = default_database
        self.default_collection = default_collection


    def set_default_database(self, database_name: str):
        """Imposta il database di deafult."""
        self.default_database = database_name


    def set_dafault_collection(self, collection_name: str):
        """IMposta la collection di default"""
        self.default_collection = collection_name

    
    def _get_collection(self, database_name: Optional[str] = None, collection_name: Optional[str] = None):
        """Recupera la collection Mongo, utilizzando i valori di default se non specificati."""
        db_name = database_name or self.default_database
        coll_name = collection_name or self.default_collection

        db = self.client[db_name]

        return db[coll_name]
    

    # Metodi per operazioni MongoDB
    def write_to_mongo(self, data: Any, database_name: Optional[str] = None, collection_name: Optional[str] = None):
        """Inserisce un documento nella collection specificata o in quella di defaukt."""
        
        db_name = database_name or self.default_database
        coll_name = collection_name or self.default_collection

        collection = self._get_collection(database_name=db_name, collection_name=coll_name)
        
        # Se è stringa, caricala come JSON
        if isinstance(data, str):
            try:
                loaded_data = json.loads(data)
            except json.JSONDecodeError as e:
                return f"❌ Errore nel parsing JSON: {e}"
        else: # è già un dizionario
            loaded_data = data

        if isinstance(loaded_data, list): # lista di dict
            result = collection.insert_many(loaded_data)
            return f"Document inserted with id: {str(result.inserted_ids)}"
        elif isinstance(loaded_data, dict):
            result = collection.insert_one(loaded_data)
            return f"Document inserted with id: {str(result.inserted_id)}"
        
        else:
            print("[WARNING]: data type is not list or dict")


    def delete_from_mongo(self, query: Any, database_name: Optional[str] = None, collection_name: Optional[str] = None):
        """Elimina documenti dalla collection specifica o da quella di default."""
        db_name = database_name or self.default_database
        coll_name = collection_name or self.default_collection

        collection = self._get_collection(database_name=db_name, collection_name=coll_name)

        # Se è stringa, caricala come JSON
        if isinstance(query, str):
            try:
                loaded_data = json.loads(query)
            except json.JSONDecodeError as e:
                return f"❌ Errore nel parsing JSON: {e}"
        else: # è già un dizionario
            loaded_data = query

        result = collection.delete_one(loaded_data)

        return f"Documents deleted: {result.deleted_count}"
    

    def read_from_mongo(self, query: Any, output_format: str = "string", database_name: Optional[str] = None, collection_name: Optional[str] = None):
        """Legge documenti dalla collection specificata o da quella di default"""

        db_name = database_name or self.default_database
        coll_name = collection_name or self.default_collection

        collection = self._get_collection(database_name=db_name, collection_name=coll_name)

        # Se è stringa, caricala come JSON
        if isinstance(query, str):
            try:
                loaded_data = json.loads(query)
            except json.JSONDecodeError as e:
                return f"Errore nel parsing JSON: {e}"
        else: # è già un dizionario
            loaded_data = query

        documents = list(collection.find(loaded_data))

        if output_format == "string":
            return str(documents)
        elif output_format == "object":
            return documents
        

    def update_in_mongo(self, query: Any, new_values: Any, database_name: Optional[str] = None, collection_name: Optional[str] = None):
        """Aggiorna documenti nella collection specificata o in quella di default."""
        db_name = database_name or self.default_database
        coll_name = collection_name or self.default_collection

        collection = self._get_collection(database_name=db_name, collection_name=coll_name)

        # Se è stringa, caricala come JSON
        if isinstance(query, str):
            try:
                loaded_data = json.loads(query)
            except json.JSONDecodeError as e:
                return f"❌ Errore nel parsing JSON: {e}"
        else: # è già un dizionario
            loaded_data = query

        # Se è stringa, caricala come JSON
        if isinstance(new_values, str):
            try:
                loaded_values = json.loads(new_values)
            except json.JSONDecodeError as e:
                return f"❌ Errore nel parsing JSON: {e}"
        else: # è già un dizionario
            loaded_values = new_values

        result = collection.update_one(loaded_data, {"$set": loaded_values})

        return f"Documents updated: {result.modified_count}"
    

if __name__ == "__main__":

    # inizializza MongoManager con una connection string
    mongo_manager = MongoManager(connection_string="mongodb://localhost:27017", default_database="Leonardo")

    collection = mongo_manager._get_collection(collection_name="example_collection")

    print(collection)

    # write_data = WriteDataModel(collection_name="example_collection", data={"name": "Alice", "age": 30})

    # result = mongo_manager.write_to_mongo(collection_name=write_data.collection_name, data=write_data.data)

    # print(result)

    # read_data = ReadDataModel(collection_name="example_collection", query={"name": "Alice"})
    # documents = mongo_manager.read_from_mongo(collection_name=read_data.collection_name, query=read_data.query)

    # print(documents)

    # update_data = UpdateDataModel(collection_name="example_collection", query={"name": "Alice"}, new_values={"age": 35})

    # count = mongo_manager.update_in_mongo(collection_name=update_data.collection_name, query=update_data.query, new_values=update_data.new_values)

    # print(count)

    # delete_data = DeleteDataModel(collection_name="example_collection", query={"name": "Alice"})

    # count = mongo_manager.delete_from_mongo(collection_name=delete_data.collection_name, query=delete_data.query)

    # print(count)