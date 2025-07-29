from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.1:8b", base_url="http://localhost:11434")
print(llm.invoke("Cosa sai del BOM?"))
