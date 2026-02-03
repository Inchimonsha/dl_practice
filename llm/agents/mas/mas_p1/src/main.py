import httpx
from typing import Optional, Any
from pydantic import BaseModel
from langchain_gigachat import GigaChatEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.message import BaseMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from sklearn.neighbors import NearestNeighbors

from src.rag.embeddings import load_embeddings
from src.config import OPENROUTER_API_KEY, GIGACHAT_API_KEY, embedder
from src.rag.chain import rag_chain
from src.domain_agents.grand import news_agent, schedule_agent


class State(BaseModel):
    # messages: Annotated[Sequence[BaseMessage], add_messages]
    query: str
    category: str = 'other'
    answer: Optional[Any] = None


def jailbreak_checker(state: State):
    print("jailbreak_checker")
    return state


def jailbreak_guard(state: State):
    print("jailbreak_guard")
    return state


def query_classifier(state: State):
    print("query_classifier")

    CLASSIFY_PROMPT = """
    Classify the following user query into one of the categories: 'schedule', 'news', or 'other'.
    Respond with JSON: {{"category": "schedule|news|other"}}

    Query: {query}
    """

    classify_prompt = ChatPromptTemplate.from_template(CLASSIFY_PROMPT)
    classify_model = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        openai_api_key=OPENROUTER_API_KEY,
        model_name="gpt-4o-mini",
        temperature=0.7
    )
    classify_chain = classify_prompt | classify_model | JsonOutputParser()

    res = classify_chain.invoke({'query': state.query})
    return {'category': res['category']}


def rag_system(state: State):
    return {"answer": rag_chain.invoke(state.query)}


llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    openai_api_key=OPENROUTER_API_KEY,
    model_name="alibaba/tongyi-deepresearch-30b-a3b:free",
    temperature=0.3
)


def main() -> None:
    graph = StateGraph(State)

    graph.add_node('jailbreak_checker', jailbreak_checker)
    graph.add_node('jailbreak_guard', jailbreak_guard)
    graph.add_node('query_classifier', query_classifier)
    graph.add_node('schedule_agent', schedule_agent)
    graph.add_node('news_agent', news_agent)
    graph.add_node('rag_system', rag_system)

    graph.add_edge(START, 'jailbreak_checker')
    graph.add_edge('jailbreak_checker', 'jailbreak_guard')
    # graph.add_conditional_edges('jailbreak_guard')
    graph.add_edge('jailbreak_guard', 'query_classifier')
    graph.add_conditional_edges(
        'query_classifier',
        lambda x: x.category,
        {
            'schedule': 'schedule_agent',
            'news': 'news_agent',
            'other': 'rag_system'
        }
    )
    graph.add_edge('schedule_agent', END)
    graph.add_edge('news_agent', END)
    graph.add_edge('rag_system', END)

    app = graph.compile()
    res = app.invoke({"query": "Рассказать о расписании"})
    print(("-----------------\n"
           f"[Область: {res.get('category')}]"
           f"\n{res.get('answer')}"
           "\n-----------------"))


if __name__ == "__main__":
    main()
