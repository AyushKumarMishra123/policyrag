"""
Retrieval architecture for Mirai Student Policy Advisor.
Implements vector search with ChromaDB and LCEL-powered MultiQueryRetriever
to bridge casual student phrasing and formal handbook terminology.
"""

from typing import List, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from app.config import settings
from app.ingestion import get_embedding_function


MULTI_QUERY_PROMPT_TEMPLATE = """You are an AI policy advisor assistant for Mirai School of Technology.
Your task is to generate 3 different search queries based on the student's inquiry below to retrieve relevant sections from the official Student Policy Handbook.

The goal is to bridge the vocabulary gap between casual, informal student phrasing (e.g., 'bunking', 'fine', 'sick leave', 'start a club') and formal administrative handbook terminology (e.g., 'minimum attendance threshold', 'debarment', 'attendance evaluation marks tier', 'Disciplinary Committee', 'medical leave 7-day submission rule', 'Faculty Coordinator club proposal 40% batch support').

Output exactly 3 alternative queries, each on a new line, without numbers, bullet points, or markdown.

Student inquiry: {question}"""

MULTI_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=MULTI_QUERY_PROMPT_TEMPLATE,
)


def get_vectorstore(
    embedding_model: Optional[Embeddings] = None,
    persist_dir: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> Chroma:
    """
    Initializes and returns the Chroma vector store instance.
    """
    p_dir = persist_dir or settings.CHROMA_PERSIST_DIR
    c_name = collection_name or settings.CHROMA_COLLECTION_NAME
    embed_fn = embedding_model or get_embedding_function()

    return Chroma(
        collection_name=c_name,
        embedding_function=embed_fn,
        persist_directory=p_dir,
    )


class MultiQueryRetrieverLCEL:
    """
    LCEL-based query expansion retriever that generates diverse policy-focused
    query formulations, executes retrieval for each, and deduplicates retrieved chunks.
    """

    def __init__(
        self,
        base_retriever,
        llm: Optional[BaseChatModel] = None,
        k: Optional[int] = None,
    ):
        self.base_retriever = base_retriever
        self.llm = llm
        self.k = k or settings.RETRIEVER_K

        if self.llm:
            self.query_generator: Optional[Runnable] = (
                MULTI_QUERY_PROMPT | self.llm | StrOutputParser()
            )
        else:
            self.query_generator = None

    def _generate_queries(self, question: str) -> List[str]:
        queries = [question]
        if not self.query_generator:
            return queries

        try:
            raw_output = self.query_generator.invoke({"question": question})
            lines = [line.strip() for line in raw_output.split("\n") if line.strip()]
            # Filter out numbering or punctuation prefixes if present
            cleaned_lines = []
            for line in lines:
                cleaned = line.lstrip("0123456789.-*• \t")
                if cleaned and len(cleaned) > 3:
                    cleaned_lines.append(cleaned)

            queries.extend(cleaned_lines[:3])
        except Exception:
            # Fall back gracefully to the original query if generation encounters errors
            pass

        return queries

    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        seen = set()
        unique_docs: List[Document] = []

        for doc in documents:
            source = doc.metadata.get("source", "")
            page = doc.metadata.get("page", 0)
            # Use chunk_id or content hash
            identifier = doc.metadata.get("chunk_id") or f"{source}::{page}::{doc.page_content[:100]}"

            if identifier not in seen:
                seen.add(identifier)
                unique_docs.append(doc)

        return unique_docs

    def invoke(self, question: str) -> List[Document]:
        """Synchronously retrieves and deduplicates documents."""
        queries = self._generate_queries(question)
        all_docs: List[Document] = []

        for q in queries:
            try:
                docs = self.base_retriever.invoke(q)
                all_docs.extend(docs)
            except Exception:
                pass

        deduped = self._deduplicate_documents(all_docs)
        return deduped[: self.k]

    async def ainvoke(self, question: str) -> List[Document]:
        """Asynchronously retrieves and deduplicates documents."""
        # For compatibility with async chains
        return self.invoke(question)


def build_retriever(
    vectorstore: Chroma,
    llm: Optional[BaseChatModel] = None,
    k: Optional[int] = None,
    use_multi_query: bool = True,
):
    """
    Constructs either a MultiQueryRetrieverLCEL or standard Chroma retriever
    based on configuration.
    """
    target_k = k or settings.RETRIEVER_K
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": target_k})

    if use_multi_query and llm is not None:
        return MultiQueryRetrieverLCEL(
            base_retriever=base_retriever,
            llm=llm,
            k=target_k,
        )

    return base_retriever
