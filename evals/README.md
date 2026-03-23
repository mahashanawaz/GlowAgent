# GlowAgent LangSmith Evaluations

Evaluation tests for the product ranking tool, run via [LangSmith](https://smith.langchain.com/).

## Setup

1. **LangSmith API key**: Get one at [smith.langchain.com](https://smith.langchain.com/) → Settings → API Keys.

2. **Environment variable**:
   ```bash
   export LANGSMITH_API_KEY="your-api-key"
   ```

3. **Run from project root** (with venv activated):
   ```bash
   cd /path/to/GlowAgent
   source venv/bin/activate
   python evals/product_ranking_eval.py
   ```

## What gets evaluated

| Evaluator           | What it checks |
|---------------------|----------------|
| **format**          | Output has numbered list (1., 2., 3.) and product info (Category, Link) |
| **skin_type_match** | Products reference the requested skin type |
| **allergy_compliance** | When `allergies` is set, none of those terms appear in results |
| **product_type_match** | Products match the requested category (moisturizer, cleanser, etc.) |
| **result_count**    | Returns up to `max_results` items, or a clear "no results" message |

## Test cases

- Happy path: oily acne skin moisturizers, combination skin serums, sunscreen
- Edge cases: sensitive skin + fragrance allergy, niacinamide allergy filter

## Options

```bash
# Create dataset only (no eval run)
python evals/product_ranking_eval.py --create-dataset-only

# Full evaluation (creates dataset + runs eval)
python evals/product_ranking_eval.py
```

## View results

After running, the script prints a link to the experiment in LangSmith. Open it to see per-example scores and pass/fail.
