import getpass
from importlib import metadata
import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pypdf

load_dotenv()

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

# ---embeddings---

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# vector_1 = embeddings.embed_query(documents[0].page_content)
# vector_2 = embeddings.embed_query(documents[1].page_content)

# assert len(vector_1) == len(vector_2), "Vector dimensions do not match"
# print(f"Generated vectors of length {len(vector_1)}\n")
# print(vector_1[:10])

# Vector DB
vector_store = PGVector(
    embeddings=embeddings,
    collection_name="my_docs",
    connection="postgresql+psycopg://h3kar360:password@localhost:5424/learning_lang_db",
    # add async_mode=True for async capabilities
)

# ---Load PDF---

def load_pdf_pages(file_path: str) -> list[Document]:
    reader = pypdf.PdfReader(file_path)
    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={ "source": file_path, "page": i }
        )
        for i, page in enumerate(reader.pages)
    ]

file_path = "./files/Capybara.pdf"
docs = load_pdf_pages(file_path)
print(len(docs))

# ---Splitting text---

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)

all_splits = text_splitter.split_documents(docs)
print(len(all_splits))

# ---Index chunks to vectorStore---

ids = vector_store.add_documents(documents=all_splits)

# ---Search---

# results = vector_store.similarity_search(
#     "What are the habitat and activities of Capybaras?"
# )

# print(results[0])

# ---Retrievers---

# @chain
# def retriever(query: str) -> list[Document]:
#     return vector_store.similarity_search(query, k=1)

# or

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1}
)

result = retriever.batch(
    [
        "What are the habitat and activities of Capybaras?",
        "What do Capybaras eat?"
    ]
)

print(result)