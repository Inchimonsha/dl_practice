import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from src.config import OPENROUTER_API_KEY
from src.rag.retriever import DocumentRetriever


with open("data/prompt.json", mode='r', encoding="utf-8") as f:
    prompts = json.load(f)

RAG_PROMPT = prompts['PREP_SYSTEM']

retriever = DocumentRetriever()
rag_template = ChatPromptTemplate.from_template(RAG_PROMPT)
rag_model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    openai_api_key=OPENROUTER_API_KEY,
    model_name="gpt-4o-mini",
    temperature=0.3
)

rag_chain = (
    {
        "context": RunnableLambda(retriever.retrieve),
        "question": RunnablePassthrough(),
    }
    | rag_template
    | rag_model
    | StrOutputParser()
)
