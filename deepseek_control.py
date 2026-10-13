"""
Control pass: feed every (perturbed) AQuA/LogiQA question to DeepSeek, with
thinking ENABLED (default), and record the full reasoning_content + final
answer for both models. This is the "100%" / baseline generation step --
later steps will truncate these same reasoning traces.

RUN THIS LOCALLY (not in a sandboxed cloud environment) because it needs to
reach api.deepseek.com.

Setup:
    pip install openai
    export DEEPSEEK_API_KEY="sk-..."

Usage:
    python3 deepseek_control.py                 # runs everything
    python3 deepseek_control.py --models flash   # just one model
    python3 deepseek_control.py --n 5            # just 5 questions per dataset, for testing

Output:
    results/control_aqua.jsonl
    results/control_logiqa.jsonl
    results/control_usage.json
Each line of the .jsonl files is one record:
    {
      "dataset": "aqua" | "logiqa",
      "model_key": "flash" | "pro",
      "question": "...",          # the PERTURBED question text
      "options": [...],
      "gt_answer": "B",
      "reasoning_content": "...", # the model's full chain of thought
      "final_text": "...",        # the model's final answer text
      "model_answer": "B",        # extracted answer letter
      "correct": true,
      "usage": {"prompt_tokens": ..., "completion_tokens": ..., "reasoning_tokens": ...}
    }
"""
import argparse
import json
import os
import re
import sys
import time

from openai import OpenAI

MODELS = {
    "flash": "deepseek-flash",
    "pro": "deepseek-v4-pro",
}

SYSTEM_PROMPT = (
    "Solve the following multiple-choice problem. Think it through carefully, "
    "then end your response with exactly one line of the form "
    "'Final answer: <letter>' where <letter> is the single correct option letter."
)

ANSWER_RE = re.compile(r"final answer:\s*\(?([A-E])\)?", re.IGNORECASE)


def extract_letter(text):
    if not text:
        return None
    matches = ANSWER_RE.findall(text)
    if matches:
        return matches[-1].upper()
    # fallback: a lone letter A-E near the end, in parens or with a period
    m = re.findall(r"\b([A-E])\b", text[-60:])
    return m[-1].upper() if m else None


def format_question(item):
    opts = "\n".join(item["options"])
    return f"{item['question']}\n\nOptions:\n{opts}"


def load_dataset(dataset_name):
    path = f"data/{dataset_name}_perturbed.json"
    records = json.load(open(path))
    items = []
    for r in records:
        p = r["perturbed"]
        items.append({
            "question": p["question"],
            "options": p["options"],
            "gt_answer": p["gt_answer"],
        })
    return items


def call_model(client, model_id, item, retries=4):
    user = format_question(item)
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                extra_body={"thinking": {"type": "enabled"}},
                max_tokens=8000,
            )
            msg = resp.choices[0].message
            reasoning = getattr(msg, "reasoning_content", None) or ""
            final_text = msg.content or ""
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            }
            # reasoning tokens often reported separately; grab if present
            rt = getattr(getattr(resp.usage, "completion_tokens_details", None), "reasoning_tokens", None)
            if rt is not None:
                usage["reasoning_tokens"] = rt
            return reasoning, final_text, usage
        except Exception as e:
            last_err = e
            print(f"    retry after error: {e}", file=sys.stderr, flush=True)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed after {retries} attempts: {last_err}")


def run_dataset(client, dataset_name, model_keys, limit=None):
    items = load_dataset(dataset_name)
    if limit:
        items = items[:limit]

    os.makedirs("results", exist_ok=True)
    out_path = f"results/control_{dataset_name}.jsonl"

    # Load any existing results. A record only counts as "done" if it got a
    # real answer -- a prior record with no model_answer means the model ran
    # out of its token budget mid-reasoning (completion_tokens hit the cap
    # with an empty final_text) and should be retried, not skipped. We keep
    # only the good records and rewrite the file with those before appending
    # fresh results, so failed attempts don't pile up as duplicate lines.
    good_records = []
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("model_answer"):
                    good_records.append(rec)
                    done.add((rec["model_key"], rec["question"]))
                else:
                    print(f"  dropping incomplete record for retry: "
                          f"{rec['model_key']} / {rec['question'][:60]!r}...",
                          file=sys.stderr, flush=True)

    with open(out_path, "w") as outf:
        for rec in good_records:
            outf.write(json.dumps(rec) + "\n")
        outf.flush()

    with open(out_path, "a") as outf:
        for mkey in model_keys:
            model_id = MODELS[mkey]
            print(f"=== {dataset_name} / {mkey} ({model_id}) : {len(items)} questions ===", file=sys.stderr)
            for i, item in enumerate(items):
                if (mkey, item["question"]) in done:
                    print(f"  {i+1}/{len(items)} (skip, done)", file=sys.stderr, flush=True)
                    continue
                print(f"  {i+1}/{len(items)}", file=sys.stderr, flush=True)
                reasoning, final_text, usage = call_model(client, model_id, item)
                ans = extract_letter(final_text) or extract_letter(reasoning)
                rec = {
                    "dataset": dataset_name,
                    "model_key": mkey,
                    "question": item["question"],
                    "options": item["options"],
                    "gt_answer": item["gt_answer"],
                    "reasoning_content": reasoning,
                    "final_text": final_text,
                    "model_answer": ans,
                    "correct": (ans == item["gt_answer"]) if ans else False,
                    "usage": usage,
                }
                outf.write(json.dumps(rec) + "\n")
                outf.flush()

    print(f"done -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["flash", "pro"], choices=list(MODELS.keys()))
    p.add_argument("--n", type=int, default=None, help="limit questions per dataset (for testing)")
    p.add_argument("--datasets", nargs="+", default=["aqua", "logiqa"], choices=["aqua", "logiqa"])
    args = p.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: set DEEPSEEK_API_KEY environment variable first.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    for dataset in args.datasets:
        run_dataset(client, dataset, args.models, args.n)

    # cost/usage summary
    summary = {}
    for dataset in args.datasets:
        path = f"results/control_{dataset}.jsonl"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                key = f"{dataset}/{rec['model_key']}"
                d = summary.setdefault(key, {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "n": 0})
                d["prompt_tokens"] += rec["usage"].get("prompt_tokens", 0)
                d["completion_tokens"] += rec["usage"].get("completion_tokens", 0)
                d["reasoning_tokens"] += rec["usage"].get("reasoning_tokens", 0) or 0
                d["n"] += 1
    json.dump(summary, open("results/control_usage.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
