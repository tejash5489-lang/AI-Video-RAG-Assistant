import uuid
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


class ChromaDefaultEmbeddings(Embeddings):
    def __init__(self):
        self.fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts):
        return [[float(x) for x in v] for v in self.fn(texts)]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def get_embeddings():
    return ChromaDefaultEmbeddings()


def build_vector_store(transcript: str) -> Chroma:
    print("Building vector store")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    return Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        collection_name=f"meeting_{uuid.uuid4().hex[:8]}",
    )


def get_retriever(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )