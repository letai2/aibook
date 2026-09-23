"""Transparent fixture evaluation, separate from language-model perplexity.

Exact strings, evidence IDs and tool results are useful regression contracts.
They are NOT an automatic judge of semantic truth or arbitrary groundedness.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    expected_answer: str
    expected_status: str = "finished"
    relevant_ids: tuple[str, ...] = ()
    expected_citations: tuple[str, ...] = ()
    expected_tools: tuple[dict, ...] = ()
    require_verified: bool = True


def score_result(case, result):
    """Return independent checks and their conjunction for one named fixture."""
    relevant = set(case.relevant_ids)
    retrieved = {identifier for event in result.events if event.get("state") == "retrieve"
                 for identifier in event["ids"]}
    in_context = set(result.context_ids)
    recall = len(relevant & retrieved) / len(relevant) if relevant else None
    context_recall = len(relevant & in_context) / len(relevant) if relevant else None
    # Only task-significant fields, not timing or auxiliary debugging metadata.
    actual_tools = tuple({"name": item["name"], "arguments": item["arguments"],
                          "value": item["value"]} for item in result.tool_results)
    checks = {"status_ok": result.status == case.expected_status,
              "answer_exact": result.answer == case.expected_answer,
              "citations_exact": set(result.citations) == set(case.expected_citations),
              "tools_exact": actual_tools == case.expected_tools,
              "verification_ok": not case.require_verified or result.verified}
    return {"id": case.id, "backend": result.backend, "status": result.status,
            **checks, "retrieval_recall": recall, "context_recall": context_recall,
            "passed": all(checks.values()) and (recall is None or recall == 1.0)
                      and (context_recall is None or context_recall == 1.0)}


def summarize_scores(rows):
    rows = list(rows)
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("provide nonempty scores with unique case IDs")
    recall = [row["retrieval_recall"] for row in rows if row["retrieval_recall"] is not None]
    context_recall = [row["context_recall"] for row in rows if row["context_recall"] is not None]
    return {"cases": len(rows), "pass_rate": sum(row["passed"] for row in rows) / len(rows),
            "answer_exact_rate": sum(row["answer_exact"] for row in rows) / len(rows),
            "retrieval_cases": len(recall),
            "mean_retrieval_recall": sum(recall) / len(recall) if recall else None,
            "mean_context_recall": sum(context_recall) / len(context_recall) if context_recall else None,
            "rows": rows}


def evaluate_cases(cases, run_case):
    """run_case(case) must create independent state for each evaluation case."""
    return summarize_scores(score_result(case, run_case(case)) for case in cases)


def regression_ids(before_rows, after_rows):
    """IDs that passed before but fail now; incompatible fixture sets are errors."""
    before_rows, after_rows = list(before_rows), list(after_rows)
    before = {row["id"]: row["passed"] for row in before_rows}
    after = {row["id"]: row["passed"] for row in after_rows}
    if (not before or set(before) != set(after) or len(before) != len(before_rows)
            or len(after) != len(after_rows)):
        raise ValueError("compare the same nonempty set of unique fixture IDs")
    return sorted(key for key in before if before[key] and not after[key])


def course_fixture_suite():
    """Actual offline controller/retrieval/tools, with plainly SCRIPTED proposals.

    Each case has fresh backend state. The verifier checks the expected fixture
    answer, not general natural-language groundedness. Unknown-topic abstention
    is scripted too; it is not a learned ability of MiniGPT.
    """
    from .assistant import ScriptedFixture, run_assistant
    from .retrieval import COURSE_DOCUMENTS, chunk_document, keyword_search
    chunks = [chunk for doc in COURSE_DOCUMENTS
              for chunk in chunk_document(doc, chunk_words=24, overlap_words=4)]
    hit = keyword_search("Checkpoint", chunks, k=1)[0]
    source = hit.chunk.id
    # An exact extract makes the narrow fixture support relationship inspectable.
    answer = hit.chunk.text
    cases = (
        EvalCase("course-evidence", "Checkpoint", answer,
                 relevant_ids=(source,), expected_citations=(source,)),
        EvalCase("arithmetic", "دو گام و سه گام، جمعاً چند گام؟", "5",
                 expected_tools=({"name": "add", "arguments": {"a": 2, "b": 3}, "value": 5},)),
        EvalCase("unknown", "تعداد قمرهای سیارهٔ خیالی زتا چند است؟", "اطلاعات کافی ندارم"),
    )
    proposals = {
        "course-evidence": [{"action": "finish", "answer": answer, "citations": [source]}],
        "arithmetic": [{"action": "tool", "name": "add", "arguments": {"a": 2, "b": 3}},
                       {"action": "finish", "answer": "5", "citations": []}],
        "unknown": [{"action": "finish", "answer": "اطلاعات کافی ندارم", "citations": []}],
    }

    def run_case(case):
        def verify(text, citations, tools):
            if text != case.expected_answer or set(citations) != set(case.expected_citations):
                return False
            actual = tuple({"name": item["name"], "arguments": item["arguments"], "value": item["value"]}
                           for item in tools)
            return actual == case.expected_tools
        return run_assistant(case.question, ScriptedFixture(proposals[case.id]),
                             chunks=chunks, verify_answer=verify)
    return cases, run_case


if __name__ == "__main__":
    import json
    from .console import configure_console
    configure_console()
    cases, runner = course_fixture_suite()
    print(json.dumps(evaluate_cases(cases, runner), ensure_ascii=False, indent=2))
