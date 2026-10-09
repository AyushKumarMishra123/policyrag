"""
Unit tests for the LCEL RAG chain, deterministic responses,
citation extraction, and unsupported question handling with mocked LLM.
"""

from langchain_core.documents import Document

from app.rag_chain import MiraiPolicyChain
from app.schemas import QuestionResponse
from tests.conftest import MockGeminiChatModel, MockRetriever


def test_rag_chain_precision_query(mock_llm):
    """
    Query: 'I have 72% attendance. How many attendance marks will I get?'
    Expected: Exactly 4 marks (from 60% – 74.99% bracket), cites Page 3.
    """
    docs = [
        Document(
            page_content=(
                "Standard Attendance Evaluation: Attendance carries a 10% weightage across all courses. "
                "Marks are awarded based on the tier system: 60% – 74.99% -> 4 marks."
            ),
            metadata={"source": "Mirai_SoT_Policy_Handbook_2026.pdf", "page": 3},
        )
    ]
    retriever = MockRetriever(docs=docs)
    chain = MiraiPolicyChain(retriever=retriever, llm=mock_llm)

    response: QuestionResponse = chain.query("I have 72% attendance. How many attendance marks will I get?")

    assert "4 marks" in response.answer
    assert 3 in response.page_numbers
    assert response.grounded is True
    assert response.insufficient_evidence is False


def test_rag_chain_multi_hop_query(mock_llm):
    """
    Query: 'I study at the Ratnam campus. I got sick and need medical leave. Who do I email and how many days do I have to submit my documents?'
    Expected: Instructs user to email Yashaswini Ma'am within 7 days.
    """
    docs = [
        Document(
            page_content="Campus Manager for Ratnam: Yashaswini Ma'am. In person at CM office / campus email.",
            metadata={"source": "Mirai_SoT_Policy_Handbook_2026.pdf", "page": 15},
        ),
        Document(
            page_content="7-Day Submission Rule: All supporting documents for Medical Leave must be submitted within 7 days of the illness/treatment.",
            metadata={"source": "Mirai_SoT_Policy_Handbook_2026.pdf", "page": 14},
        ),
    ]
    retriever = MockRetriever(docs=docs)
    chain = MiraiPolicyChain(retriever=retriever, llm=mock_llm)

    response: QuestionResponse = chain.query(
        "I study at the Ratnam campus. I got sick and need medical leave. Who do I email and how many days do I have to submit my documents?"
    )

    assert "yashaswini" in response.answer.lower()
    assert "7 days" in response.answer.lower()
    assert set([14, 15]).issubset(set(response.page_numbers))
    assert response.grounded is True


def test_rag_chain_process_verification(mock_llm):
    """
    Query: 'We want to start a new Cybersecurity society under the Tech Club. Do we ask Management directly?'
    Expected: Requires 40% batch support and proposal must go to Faculty Coordinator first, not Management.
    """
    docs = [
        Document(
            page_content="Tech Club may establish subsidiary societies such as Cybersecurity Society. A new club can be proposed if at least 40% of the total batch students are in favor. Must be submitted to designated Faculty Coordinator first.",
            metadata={"source": "Mirai_SoT_Policy_Handbook_2026.pdf", "page": 10},
        )
    ]
    retriever = MockRetriever(docs=docs)
    chain = MiraiPolicyChain(retriever=retriever, llm=mock_llm)

    response: QuestionResponse = chain.query(
        "We want to start a new Cybersecurity society under the Tech Club. Do we ask Management directly?"
    )

    assert "40%" in response.answer
    assert "faculty coordinator" in response.answer.lower()
    assert response.grounded is True


def test_rag_chain_negative_constraint(mock_llm):
    """
    Query: 'How much is the fine for smoking a cigarette on campus?'
    Expected: Tobacco is prohibited and leads to Disciplinary Committee action; no monetary fine fabricated.
    """
    docs = [
        Document(
            page_content="Code of Conduct: Prohibited Activities include possession or consumption of alcohol, tobacco, or prohibited substances. Disciplinary Committee shall examine reported violations.",
            metadata={"source": "Mirai_SoT_Policy_Handbook_2026.pdf", "page": 9},
        )
    ]
    retriever = MockRetriever(docs=docs)
    chain = MiraiPolicyChain(retriever=retriever, llm=mock_llm)

    response: QuestionResponse = chain.query("How much is the fine for smoking a cigarette on campus?")

    assert "prohibited" in response.answer.lower()
    assert "disciplinary committee" in response.answer.lower()
    assert "no monetary fine" in response.answer.lower() or "not specify" in response.answer.lower()
    assert response.grounded is True


def test_rag_chain_unsupported_question(mock_llm):
    """
    Query with zero relevance to the policy handbook:
    Expected: Admits insufficient evidence, declines to speculate, grounded=False.
    """
    retriever = MockRetriever(docs=[])
    chain = MiraiPolicyChain(retriever=retriever, llm=mock_llm)

    response: QuestionResponse = chain.query("What is the recipe for chicken pasta?")

    assert response.grounded is False
    assert response.insufficient_evidence is True
    assert "does not contain sufficient information" in response.answer.lower()
