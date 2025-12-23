import os
from dotenv import load_dotenv
from langchain_gigachat import GigaChatEmbeddings

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
GIGACHAT_API_KEY = os.environ.get("GIGACHAT_API_KEY")

embedder = GigaChatEmbeddings(
    credentials=GIGACHAT_API_KEY,
    verify_ssl_certs=False
)
