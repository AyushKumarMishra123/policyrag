"""
Shared fixtures and test configuration for Mirai test suite.
Mocks external Gemini API calls so all tests run offline without credentials.
"""

import sys
from pathlib import Path
from typing import List, Optional
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import app
from app.config import settings


class MockEmbeddings(Embeddings):
    """Deterministic mock embedding model producing 8-dimensional vectors."""
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.05 * (i + 1) for i in range(8)] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.05 * (i + 1) for i in range(8)]


class MockGeminiChatModel(BaseChatModel):
    """Mock Chat Model returning deterministic policy-aligned responses."""

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: List[str] = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        # Extract the user's specific query from the HumanMessage
        user_messages = [m.content for m in messages if isinstance(m, HumanMessage)]
        query_text = " ".join(user_messages) if user_messages else (messages[-1].content if messages else "")

        if "72%" in query_text or "attendance marks" in query_text.lower():
            content = (
                "Based on the Academic Evaluation Policy (Page 3), with 72% attendance you fall in the 60% – 74.99% tier "
                "and will receive exactly 4 marks. [Page 3]"
            )
        elif "ratnam" in query_text.lower() or "medical leave" in query_text.lower():
            content = (
                "According to the Medical Leave Policy (Pages 14-15), for Ratnam campus you must email your Campus Manager "
                "Yashaswini Ma'am. All supporting documents must be submitted within 7 days of the illness or treatment. [Page 14] [Page 15]"
            )
        elif "cybersecurity" in query_text.lower() or "tech club" in query_text.lower():
            content = (
                "According to the Clubs and Events Policy (Pages 10-11), you do not ask Management directly. "
                "A new society proposal requires at least 40% batch support and must be submitted to the designated "
                "Faculty Coordinator first. [Page 10]"
            )
        elif "smoking" in query_text.lower() or "cigarette" in query_text.lower():
            content = (
                "Under the Code of Conduct (Page 9), possession or consumption of tobacco/smoking on campus is strictly prohibited "
                "and subject to Disciplinary Committee action. The policy handbook does not specify any monetary fine. [Page 9]"
            )
        elif "recipe" in query_text.lower() or "chicken pasta" in query_text.lower():
            content = "The provided policy handbook does not contain sufficient information to answer this question."
        else:
            content = (
                "Based on the Academic Evaluation Policy (Page 3), with 72% attendance you fall in the 60% – 74.99% tier "
                "and will receive exactly 4 marks. [Page 3]"
            )

        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "mock-gemini-chat"


class MockRetriever:
    """Mock retriever returning realistic handbook documents."""
    def __init__(self, docs: Optional[List[Document]] = None):
        if docs is not None:
            self.docs = docs
        else:
            self.docs = [
                Document(
                    page_content="Attendance Percentage: 60% - 74.99% -> 4 marks Awarded. Standard Attendance Evaluation.",
                    metadata={"source": "Mirai_SoT_Policy_Handbook_2026.pdf", "page": 3},
                ),
                Document(
                    page_content="Ratnam Campus: Campus Manager Yashaswini Ma'am. 7-Day Submission Rule for Medical Leave.",
                    metadata={"source": "Mirai_SoT_Policy_Handbook_2026.pdf", "page": 14},
                ),
                Document(
                    page_content="Ratnam Campus CM office / campus email.",
                    metadata={"source": "Mirai_SoT_Policy_Handbook_2026.pdf", "page": 15},
                ),
            ]

    def invoke(self, query: str) -> List[Document]:
        return self.docs

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.docs


@pytest.fixture
def test_client():
    """Provides FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_embeddings():
    return MockEmbeddings()


@pytest.fixture
def mock_llm():
    return MockGeminiChatModel()


@pytest.fixture
def mock_retriever():
    return MockRetriever()
