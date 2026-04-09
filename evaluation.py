import re
from typing import Any, Dict, List
from openai import AsyncOpenAI

async def semantic_accuracy_check(prediction: str, reference: str, client: AsyncOpenAI) -> bool:
    eval_prompt = (
        f"Ground Truth: {reference}\n"
        f"Model Answer: {prediction}\n\n"
        "Does the model answer contain the core physical concept and formulas "
        "present in the ground truth? Answer with 'YES' or 'NO' and a brief reason."
    )
    # Call OpenAI to get a 'YES' or 'NO'
    # ... logic to return True/False

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\[source:[^\]]+\]", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    ref_counts = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    overlap = 0
    for token, count in pred_counts.items():
        overlap += min(count, ref_counts.get(token, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def source_hit(actual_sources: List[Dict[str, Any]], expected_sources: List[str]) -> bool:
    expected = {item.lower().strip() for item in expected_sources}
    actual = {str(item.get("source", "")).lower().strip() for item in actual_sources}
    return bool(expected & actual)


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "total": 0,
            "answer_accuracy": 0.0,
            "source_accuracy": 0.0,
            "grounded_rate": 0.0,
            "answer_correct_count": 0,
            "source_correct_count": 0,
            "grounded_count": 0,
            "answer_accuracy_pct": 0,
            "source_accuracy_pct": 0,
            "grounded_rate_pct": 0,
        }

    total = len(results)
    answer_correct_count = sum(item["answer_correct"] for item in results)
    source_correct_count = sum(item["source_correct"] for item in results)
    grounded_count = sum(item["grounded"] for item in results)
    answer_accuracy = answer_correct_count / total
    source_accuracy = source_correct_count / total
    grounded_rate = grounded_count / total
    avg_grounding = sum(item.get("grounding_score", 0.0) for item in results) / total if total > 0 else 0
    return {
        "total": total,
        "answer_accuracy": round(answer_accuracy, 4),
        "mean_grounding_score": round(avg_grounding, 4),
        "source_accuracy": round(source_accuracy, 4),
        "grounded_rate": round(grounded_rate, 4),
        "answer_correct_count": answer_correct_count,
        "source_correct_count": source_correct_count,
        "grounded_count": grounded_count,
        "answer_accuracy_pct": round(answer_accuracy * 100, 1),
        "source_accuracy_pct": round(source_accuracy * 100, 1),
        "grounded_rate_pct": round(grounded_rate * 100, 1),
    }
