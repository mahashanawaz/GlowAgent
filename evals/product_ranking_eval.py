"""
LangSmith evaluation for the GlowAgent product_ranking_tool.

Run:
    export LANGSMITH_API_KEY="your-key"
    python evals/product_ranking_eval.py

Or create dataset only:
    python evals/product_ranking_eval.py --create-dataset-only
"""

import re
import argparse
import os
import sys
from pathlib import Path

# Add project root to path so we can import app.glow_agent
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

# Test cases aligned with Evaluation Plan + product ranking scenarios
DATASET_EXAMPLES = [
    {
        "inputs": {
            "query": "moisturizer for oily acne-prone skin",
            "skin_type": "oily",
            "concerns": "acne, pores",
            "product_type": "moisturizer",
            "allergies": "",
            "max_results": 5,
        },
        "metadata": {"type": "happy_path", "description": "Oily acne skin, moisturizer"},
    },
    {
        "inputs": {
            "query": "cleanser for dry sensitive skin",
            "skin_type": "sensitive",
            "concerns": "dryness, sensitivity",
            "product_type": "cleanser",
            "allergies": "fragrance",
            "max_results": 3,
        },
        "metadata": {"type": "edge_case", "description": "Sensitive skin + fragrance allergy"},
    },
    {
        "inputs": {
            "query": "serum for brightening and dark spots",
            "skin_type": "combination",
            "concerns": "dark spots, aging",
            "product_type": "serum",
            "allergies": "",
            "max_results": 4,
        },
        "metadata": {"type": "happy_path", "description": "Combination skin, brightening serum"},
    },
    {
        "inputs": {
            "query": "sunscreen for oily skin",
            "skin_type": "oily",
            "concerns": "",
            "product_type": "sunscreen",
            "allergies": "",
            "max_results": 3,
        },
        "metadata": {"type": "happy_path", "description": "Oily skin sunscreen"},
    },
    {
        "inputs": {
            "query": "moisturizer avoiding niacinamide",
            "skin_type": "dry",
            "concerns": "",
            "product_type": "moisturizer",
            "allergies": "niacinamide",
            "max_results": 3,
        },
        "metadata": {"type": "edge_case", "description": "Niacinamide allergy filter"},
    },
    {
        "inputs": {
            "query": "best toner for combination skin",
            "skin_type": "combination",
            "concerns": "pores",
            "product_type": "toner",
            "allergies": "",
            "max_results": 2,
        },
        "metadata": {"type": "happy_path", "description": "Combination skin toner, 2 results"},
    },
]

DATASET_NAME = "glowagent-product-ranking"


def create_dataset(client: Client) -> str:
    """Create or reuse the LangSmith dataset and add examples."""
    datasets = list(client.list_datasets(dataset_name=DATASET_NAME, limit=1))
    if datasets:
        dataset = datasets[0]
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Test cases for GlowAgent product_ranking_tool: skin type matching, allergy filtering, format validation.",
        )

    examples_to_add = [
        {"inputs": ex["inputs"], "metadata": ex.get("metadata", {})}
        for ex in DATASET_EXAMPLES
    ]
    try:
        client.create_examples(dataset_id=dataset.id, examples=examples_to_add)
    except Exception as e:
        print(f"Note: Some examples may already exist: {e}")

    print(f"Dataset '{DATASET_NAME}' ready (id={dataset.id}) with {len(DATASET_EXAMPLES)} examples.")
    return DATASET_NAME


# --- Evaluators ---


def format_evaluator(run, example) -> dict:
    """Output has numbered list, product names, links."""
    outputs = run.outputs or {}
    output = outputs.get("result", "") or ""
    has_numbered = bool(re.search(r"^\d+\.\s+\S+", output, re.MULTILINE))
    has_link = "Link:" in output or "http" in output
    has_product_info = "Category:" in output or "Effects:" in output
    score = 1.0 if (has_numbered and (has_link or has_product_info)) else 0.0
    return {"key": "format", "score": score}


def skin_type_match_evaluator(run, example) -> dict:
    """Products in output should reference the requested skin type (or compatible types)."""
    outputs = run.outputs or {}
    output = (outputs.get("result", "") or "").lower()
    inputs = getattr(example, "inputs", None) or {}
    skin_type = (inputs.get("skin_type") or "").lower()
    if not skin_type:
        return {"key": "skin_type_match", "score": 1.0}
    # Check that output mentions skin-relevant terms (Skin types: ... usually lists them)
    skin_terms = {"oily", "dry", "combination", "sensitive", "normal"}
    requested = {s for s in skin_terms if s in skin_type}
    if not requested:
        return {"key": "skin_type_match", "score": 1.0}
    # Output should contain some skin type info (e.g. "Skin types: ...")
    has_skin_section = "skin type" in output or any(t in output for t in skin_terms)
    return {"key": "skin_type_match", "score": 1.0 if has_skin_section else 0.0}


def allergy_compliance_evaluator(run, example) -> dict:
    """When allergies specified, none of those terms appear in recommended products."""
    inputs = getattr(example, "inputs", None) or {}
    allergies_str = inputs.get("allergies", "") or ""
    allergy_terms = [a.strip().lower() for a in allergies_str.split(",") if a.strip()]
    if not allergy_terms:
        return {"key": "allergy_compliance", "score": 1.0}

    output = ((run.outputs or {}).get("result", "") or "").lower()
    for term in allergy_terms:
        if term in output:
            return {"key": "allergy_compliance", "score": 0.0, "comment": f"Allergen '{term}' found in output"}
    return {"key": "allergy_compliance", "score": 1.0}


def product_type_match_evaluator(run, example) -> dict:
    """Products should be of the requested product_type (moisturizer, cleanser, etc.)."""
    inputs = getattr(example, "inputs", None) or {}
    pt = (inputs.get("product_type") or "").lower()
    if not pt:
        return {"key": "product_type_match", "score": 1.0}

    output = (run.outputs.get("result", "") or "").lower()
    # Category: Moisturizer, Category: Face Wash, etc.
    type_aliases = {
        "cleanser": ["cleanser", "face wash", "wash"],
        "moisturizer": ["moisturizer", "cream", "lotion"],
        "serum": ["serum", "essence"],
        "sunscreen": ["sunscreen", "spf"],
        "toner": ["toner"],
    }
    keywords = type_aliases.get(pt, [pt])
    has_match = any(kw in output for kw in keywords)
    return {"key": "product_type_match", "score": 1.0 if has_match else 0.0}


def result_count_evaluator(run, example) -> dict:
    """Output should have up to max_results items (or indicate no results)."""
    inputs = getattr(example, "inputs", None) or {}
    max_results = inputs.get("max_results", 5) or 5
    output = (run.outputs or {}).get("result", "") or ""

    if "No matching products" in output or "No products found" in output:
        return {"key": "result_count", "score": 1.0}

    lines = [l for l in output.split("\n") if re.match(r"^\d+\.\s+", l)]
    count = len(lines)
    ok = 0 < count <= max_results
    return {"key": "result_count", "score": 1.0 if ok else 0.0, "comment": f"Got {count}, max {max_results}"}


def run_evaluation(create_dataset_only: bool = False):
    from app.glow_agent import product_ranking_tool

    client = Client()
    create_dataset(client)
    if create_dataset_only:
        return

    def target(inputs: dict) -> dict:
        result = product_ranking_tool.invoke(inputs)
        return {"result": result}

    evaluators = [
        format_evaluator,
        skin_type_match_evaluator,
        allergy_compliance_evaluator,
        product_type_match_evaluator,
        result_count_evaluator,
    ]

    print("Running evaluation on product_ranking_tool...")
    results = client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=evaluators,
        experiment_prefix="product-ranking-tool",
        description="Evaluate GlowAgent product_ranking_tool: format, skin match, allergy compliance, product type, count.",
        max_concurrency=2,
    )
    print(results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-dataset-only", action="store_true", help="Only create/update dataset, do not run eval")
    args = parser.parse_args()

    if not os.environ.get("LANGSMITH_API_KEY"):
        print("Set LANGSMITH_API_KEY to run evaluations. Get one at https://smith.langchain.com/")
        exit(1)

    run_evaluation(create_dataset_only=args.create_dataset_only)
