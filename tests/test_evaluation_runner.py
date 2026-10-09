"""
Unit tests for the evaluation benchmarking engine, metric calculations, and CSV persistence.
"""

import csv
from pathlib import Path
import pytest

from evaluation.run_evaluation import (
    CSV_HEADER,
    calculate_answer_relevance_score,
    calculate_faithfulness_score,
    calculate_retrieval_score,
    compute_overall_score,
    evaluate_single_case,
    initialize_csv_if_missing,
    load_benchmark_dataset,
    write_evaluation_to_csv,
)
from tests.conftest import MockGeminiChatModel, MockRetriever
from app.rag_chain import MiraiPolicyChain
from langchain_core.documents import Document


def test_load_benchmark_dataset():
    """
    Validates that benchmark_questions.json is present, parses valid JSON,
    and contains all mandatory fields and categories.
    """
    dataset = load_benchmark_dataset()
    assert len(dataset) >= 10, "Expected at least 10 benchmark questions"

    categories = set()
    for tc in dataset:
        assert "id" in tc
        assert "question" in tc
        assert "expected_answer" in tc
        assert "category" in tc
        assert "is_unsupported" in tc
        categories.add(tc["category"])

    # Ensure all required problem statement and policy categories are present
    assert "Direct Factual" in categories
    assert "Multi-Hop Reasoning" in categories
    assert "Conditions & Exceptions" in categories or "Conditions & Process" in categories
    assert "Unsupported / Negative Constraint" in categories or "Unsupported / Out-of-Scope" in categories


def test_metric_calculations_factual():
    """
    Validates metric calculations for a factual question where generated answer is perfect.
    """
    retrieval_sc = calculate_retrieval_score(retrieved_pages=[3], expected_pages=[3], is_unsupported=False)
    assert retrieval_sc == 1.0

    faithfulness_sc = calculate_faithfulness_score(
        generated_answer="You will receive 4 marks.",
        forbidden_terms=["10 marks", "0 marks"],
        is_unsupported=False,
    )
    assert faithfulness_sc == 1.0

    relevance_sc = calculate_answer_relevance_score(
        generated_answer="With 72% attendance you fall into the 60% – 74.99% tier and get 4 marks.",
        key_criteria=["4 marks", "60% – 74.99%"],
        is_unsupported=False,
    )
    assert relevance_sc == 1.0

    overall = compute_overall_score(retrieval_sc, faithfulness_sc, relevance_sc)
    assert overall == 1.0


def test_metric_calculations_hallucination_penalty():
    """
    Validates that presence of forbidden hallucinated terms drives faithfulness to 0.0.
    """
    faithfulness_sc = calculate_faithfulness_score(
        generated_answer="The fine for smoking is Rs. 1000.",
        forbidden_terms=["fine of 1000", "rs."],
        is_unsupported=True,
    )
    assert faithfulness_sc == 0.0


def test_metric_calculations_unsupported_refusal():
    """
    Validates that admitting lack of evidence for unsupported questions scores 1.0 faithfulness.
    """
    faithfulness_sc = calculate_faithfulness_score(
        generated_answer="The provided policy handbook does not contain sufficient information to answer this question.",
        forbidden_terms=["10 lpa", "guaranteed"],
        is_unsupported=True,
    )
    assert faithfulness_sc == 1.0


def test_csv_persistence_and_schema(tmp_path: Path):
    """
    Validates that CSV output strictly adheres to the required schema,
    escapes multiline strings and quotes, and supports append vs overwrite.
    """
    csv_file = tmp_path / "test_eval_scores.csv"

    # Step 1: Initialize header
    initialize_csv_if_missing(str(csv_file))
    assert csv_file.is_file()

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == CSV_HEADER

    # Step 2: Write first batch
    sample_rows = [
        {
            "timestamp": "2026-10-09T06:00:00Z",
            "question": 'What is "attendance"?',
            "expected_answer": "Attendance is 75%.\nSecond line.",
            "generated_answer": 'Generated: "4 marks".',
            "retrieved_sources": "Handbook.pdf (p.3)",
            "retrieval_score": 1.0,
            "faithfulness_score": 1.0,
            "answer_relevance_score": 1.0,
            "overall_score": 1.0,
            "passed": True,
            "error": "",
        }
    ]

    write_evaluation_to_csv(sample_rows, csv_path=str(csv_file), overwrite=False)

    with open(csv_file, "r", encoding="utf-8") as f:
        lines = list(csv.reader(f))
        assert len(lines) == 2  # Header + 1 row
        assert lines[1][1] == 'What is "attendance"?'
        assert "Second line." in lines[1][2]

    # Step 3: Append second batch
    write_evaluation_to_csv(sample_rows, csv_path=str(csv_file), overwrite=False)
    with open(csv_file, "r", encoding="utf-8") as f:
        lines = list(csv.reader(f))
        assert len(lines) == 3  # Header + 2 rows

    # Step 4: Overwrite
    write_evaluation_to_csv(sample_rows, csv_path=str(csv_file), overwrite=True)
    with open(csv_file, "r", encoding="utf-8") as f:
        lines = list(csv.reader(f))
        assert len(lines) == 2  # Header + 1 row


def test_evaluate_single_case_with_mock_chain(mock_llm):
    """
    Validates evaluation of a single test case using the actual RAG chain class with mock LLM.
    """
    docs = [
        Document(
            page_content="Attendance percentage 60% - 74.99% awards 4 marks.",
            metadata={"source": "Mirai_SoT_Policy_Handbook_2026.pdf", "page": 3},
        )
    ]
    retriever = MockRetriever(docs=docs)
    chain = MiraiPolicyChain(retriever=retriever, llm=mock_llm)

    test_case = {
        "id": "BM-01",
        "question": "I have 72% attendance. How many attendance marks will I get?",
        "expected_answer": "You will receive 4 marks.",
        "expected_pages": [3],
        "key_criteria": ["4 marks"],
        "forbidden_terms": ["10 marks", "0 marks"],
        "is_unsupported": False,
    }

    result = evaluate_single_case(test_case, chain=chain)
    assert result["passed"] is True
    assert result["faithfulness_score"] >= 0.75
    assert result["retrieval_score"] == 1.0
    assert result["error"] == ""
