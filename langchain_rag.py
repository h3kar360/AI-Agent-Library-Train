# from deepagents import create_deep_agent
# from langchain.messages import HumanMessage
from importlib import metadata

from dotenv import load_dotenv

load_dotenv()

# ---No RAG implementation---

# query = "What is my mom's name?"

# agent = create_deep_agent(
#     model="google_genai:gemini-3.6-flash",
#     tools=[],
#     system_prompt=(
#         "You are a helpful LangChain documentation assistant. "
#         "Answer questions about LangChain APIs and patterns."
#     )
# )

# result = agent.invoke(
#     {
#         "messages": [HumanMessage(content=query)]
#     }
# )

# print(result["messages"][-1].text)

# ---RAG implementation---

import requests
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_BASE = "https://docs.langchain.com"

# Curated LangChain OSS pages for this tutorial. Expand this list or parse
# URLs from https://docs.langchain.com/llms.txt to index more of the site.
DOC_PATHS = [
    "oss/python/langchain/agents",
    "oss/python/deepagents/rag",
    "oss/python/langchain/tools",
    "oss/python/langchain/models",
    "oss/python/deepagents/retrieval",
    "oss/python/langchain/knowledge-base",
    "oss/python/langchain/middleware",
    "oss/python/deepagents/overview",
    "oss/python/deepagents/subagents",
    "oss/python/deepagents/streaming",
    "oss/python/deepagents/frontend/subagent-streaming",
    "oss/python/deepagents/backends",
    "oss/python/langgraph/overview",
    "oss/python/langgraph/quickstart",
]

def load_langchain_docs(doc_paths: list[str] | None = None) -> list[Document]:
    paths = doc_paths or DOC_PATHS
    docs: list[Document] = []

    for path in paths:
        url = f"{DOCS_BASE}/{path}.md"
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
        except requests.RequestException:
            continue
        source = f"{DOCS_BASE}/{path}"
        docs.append(
            Document(page_content=response.text, metadata={ "source": source })
        )

    return docs

docs = load_langchain_docs()
print(f"Loaded {len(docs)} documentation pages.")

total_chars = sum(len(doc.page_content) for doc in docs)
print(f"Total characters: {total_chars}")
print(docs[0].page_content[:500])

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)

all_splits = text_splitter.split_documents(docs)
print(f"Split documentation into {len(all_splits)} chunks.")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", max_retries=6, request_timeout=60)

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="my_docs",
    connection="postgresql+psycopg://h3kar360:password@localhost:5424/learning_lang_db"
)

vector_store.add_documents(documents=all_splits)
print(f"Indexed {len(all_splits)} chunks.")