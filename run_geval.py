import os
os.environ["OPENAI_API_KEY"] = ""  # your key here

import json
from openai import OpenAI

# Initialize OpenAI client globally so judge_sample() can use it
client = OpenAI()

JUDGE_PROMPT = """
You are evaluating a persona-based assistant answer.

Persona: {persona}
Model: {model_name}

User question:
{question}

Assistant answer:
{answer}


Context used (may be empty):
{context}

Rate the answer on these criteria from 0 to 10:

1. persona_adherence: How well it sounds like the given persona.
2. relevance_correctness: How relevant and factually reasonable the answer is.
3. use_of_context: How well the answer uses the given context (if any).
4. overall_quality: Overall clarity, coherence, and usefulness.

Return ONLY a JSON object with this schema:
{
  "persona_adherence": <int 0-10>,
  "relevance_correctness": <int 0-10>,
  "use_of_context": <int 0-10>,
  "overall_quality": <int 0-10>,
  "comments": "<short natural language explanation>"
}
"""

def judge_sample(sample):
    persona = sample["persona"]
    model_name = sample["model_name"]
    question = sample["user_question"]
    answer = sample["assistant_answer"]
    context = sample.get("context", "")

    prompt = JUDGE_PROMPT.format(
        persona=persona,
        model_name=model_name,
        question=question,
        answer=answer,
        context=context
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()

    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        scores = {
            "persona_adherence": None,
            "relevance_correctness": None,
            "use_of_context": None,
            "overall_quality": None,
            "comments": "Failed to parse judge output.",
            "raw": raw
        }
        return scores

    # ====================================================
    # Convert 0–10 scores to 0–1 decimals
    # ====================================================
    for k in ["persona_adherence", "relevance_correctness", "use_of_context", "overall_quality"]:
        if scores.get(k) is not None:
            try:
                raw_val = float(scores[k])
            except:
                raw_val = 0.0
            # 9.0 → 0.9, 10 → 1.0, etc.
            scores[k] = round(raw_val / 10, 3)

    return scores


def load_samples(path="eval_samples.jsonl"):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def save_results(results, path="eval_results.jsonl"):
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def compute_aggregates(results):
    persona_buckets = {}
    for r in results:
        persona = r.get("persona", "Unknown")
        persona_buckets.setdefault(persona, []).append(r)

    print("\n=== Aggregate scores by persona (0 to 1 scale) ===")
    for persona, rows in persona_buckets.items():
        valid_rows = [row for row in rows if row.get("persona_adherence") is not None]
        if not valid_rows:
            print(f"\nPersona: {persona}")
            print("  No valid scores. Possibly parse failures.")
            continue

        n = len(valid_rows)
        pa = sum(row["persona_adherence"] for row in valid_rows) / n
        rc = sum(row["relevance_correctness"] for row in valid_rows) / n
        uc = sum(row["use_of_context"] for row in valid_rows) / n
        oq = sum(row["overall_quality"] for row in valid_rows) / n

        print(f"\nPersona: {persona}")
        print(f"  persona_adherence:    {pa:.2f}")
        print(f"  relevance_correctness: {rc:.2f}")
        print(f"  use_of_context:       {uc:.2f}")
        print(f"  overall_quality:      {oq:.2f}")


def main():
    print("Loading samples from eval_samples.jsonl ...")
    samples = load_samples("eval_samples.jsonl")
    print(f"Loaded {len(samples)} samples.")

    results = []
    for i, sample in enumerate(samples, start=1):
        print(f"Evaluating sample {i}/{len(samples)} ...")
        scores = judge_sample(sample)
        merged = {**sample, **scores}
        results.append(merged)

    save_results(results, "eval_results.jsonl")
    print("Saved detailed results to eval_results.jsonl")

    compute_aggregates(results)


if __name__ == "__main__":
    main()
