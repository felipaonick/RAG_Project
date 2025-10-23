from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

def _hyde_text_via_llm(query: str, base_url: str, model_name: str) -> str:
    """Genera documento ipotetico (HyDE) per migliorare il vettore denso."""

    prompt = ChatPromptTemplate.from_template(
        """Generate a short hypothetical document that would perfectly answer the user's query.
        Write in neutral, factual style, 5-7 concise sentences, no fluff, no bullet points.
        
        User query: {q}
        """
    )

    llm = ChatOllama(model=model_name, base_url=base_url, temperature=0.0)

    hyde = (prompt | llm | StrOutputParser()).invoke({"q": query})

    return hyde.strip()