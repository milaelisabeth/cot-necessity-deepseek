"""
Early-answering (truncation) pass: for each question in the control-pass
results, truncate DeepSeek's OWN recorded reasoning_content to 25%/50%/75%
of its character length (snapped to a word boundary), plus a 0% (no
reasoning shown) condition, and force an immediate final answer from only
that partial/empty prefix. The 100% condition is NOT re-queried -- it's
reused directly from the control-pass result (full CoT -> answer already
recorded there), since re-asking would just reproduce the same thing.

This tests whether accuracy climbs as more of the model's OWN real reasoning
is restored (necessity) or stays flat (propensity without necessity).

Truncation is done by raw text length (not by sentence/step), snapped to a
word boundary -- we deliberately do NOT claim these cut points correspond to
"logical steps," since DeepSeek's native reasoning_content does not
decompose cleanly into discrete steps the way a cleaner, more disciplined
CoT might (see README / write-up discussion). This keeps the manipulation
entirely within DeepSeek's own original text, with no other model's
judgment involved in choosing cut points.

RUN THIS LOCALLY (not in a sandboxed cloud environment) because it needs to
reach api.deepseek.com.

Requires: results/control_aqua.jsonl and results/control_logiqa.jsonl
(the finished, fixed control-pass output) already present.

Setup:
    pip install openai
    export DEEPSEEK_API_KEY="sk-..."

Usage:
    python3 deepseek_truncation.py                # runs everything
    python3 deepseek_truncation.py --models flash  # just one model
    python3 deepseek_truncation.py --n 5           # just 5 items per dataset, for testing

Output:
    results/truncation_aqua.jsonl
    results/truncation_logiqa.jsonl
    results/truncation_usage.json
Each line is one record:
    {
      "dataset": "aqua" | "logiqa",
      "model_key": "flash" | "pro",
      "question": "...",
      "options": [...],
      "gt_answer": "B",
      "condition": "0" | "25" | "50" | "75" | "100",
      "prefix_chars": 0,                # length of the truncated prefix shown
      "full_cot_chars": 910,            # length of the full original CoT
      "prefix_text": "...",             # the truncated prefix actually shown (empty for 0%)
      "reasoning_content": "...",       # any NEW reasoning the model did on this forced-answer call
      "final_text": "...",
      "model_answer": "B",
      "correct": true,
      "usage": {...},
      "source": "forced_answer" | "control_reuse"   # 100% rows are reused, not re-queried
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

TRUNC_FRACTIONS = (0.0, 0.25, 0.5, 0.75)  # 1.0 (100%) is reused from control, not re-queried

FORCED_SYSTEM = (
    "You will be shown a multiple-choice problem and a PARTIAL chain of reasoning you started "
    "(it may be truncated or completely empty). Do NOT perform any additional reasoning and do NOT "
    "continue the work. Based only on what is shown, immediately output exactly one line of the form "
    "'Final answer: <letter>' with your best-guess answer letter. No other text."
)

ANSWER_RE = re.compile(r"final answer:\s*\(?([A-E])\)?", re.IGNORECASE)


def extract_letter(text):
    if not text:
        return None
    matches = ANSWER_RE.findall(text)
    if matches:
        return matches[-1].upper()
    m = re.findall(r"\b([A-E])\b", text[-60:])
    return m[-1].upper() if m else None


def format_question(item):
    opts = "\n".join(item["options"])
    return f"{item['question']}\n\nOptions:\n{opts}"


def truncate_to_fraction(text, fraction):
    """Take the first `fraction` of text by character length, snapped back
    to the nearest preceding whitespace so we don't cut mid-word."""
    text = text.strip()
    if fraction <= 0:
        return ""
    if fraction >= 1:
        return text
    target_len = int(len(text) * fraction)
    if target_len >= len(text):
        return text
    cut = text.rfind(" ", 0, target_len)
    if cut == -1:
        cut = target_len
    return text[:cut].strip()


def load_control(dataset_name, limit=None):
    """Load the finished control-pass results, keeping only records that got
    a real answer (the 2 that never resolved are excluded, as agreed)."""
    path = f"results/control_{dataset_name}.jsonl"
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("model_answer"):
                items.append(rec)
            else:
                print(f"  excluding unresolved control item: {rec['model_key']} / "
                      f"{rec['question'][:60]!r}...", file=sys.stderr, flush=True)
    if limit:
        items = items[:limit]
    return items


def call_model(client, model_id, item, prefix_text, retries=4):
    user = format_question(item) + f"\n\nYour reasoning so far (may be empty):\n{prefix_text if prefix_text else '(none)'}"
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": FORCED_SYSTEM},
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
    control_items = load_control(dataset_name, limit)
    control_items = [r for r in control_items if r["model_key"] in model_keys]

    os.makedirs("results", exist_ok=True)
    out_path = f"results/truncation_{dataset_name}.jsonl"

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
                    done.add((rec["model_key"], rec["question"], rec["condition"]))
                else:
                    print(f"  dropping incomplete record for retry: {rec['model_key']} / "
                          f"{rec['condition']}% / {rec['question'][:50]!r}...",
                          file=sys.stderr, flush=True)

    with open(out_path, "w") as outf:
        for rec in good_records:
            outf.write(json.dumps(rec) + "\n")
        outf.flush()

    with open(out_path, "a") as outf:
        total = len(control_items)
        for i, ctrl in enumerate(control_items):
            mkey = ctrl["model_key"]
            model_id = MODELS[mkey]
            item = {"question": ctrl["question"], "options": ctrl["options"], "gt_answer": ctrl["gt_answer"]}
            full_cot = ctrl["reasoning_content"]
            print(f"[{dataset_name}/{mkey}] item {i+1}/{total}", file=sys.stderr, flush=True)

            if (mkey, ctrl["question"], "100") not in done:
                rec100 = {
                    "dataset": dataset_name,
                    "model_key": mkey,
                    "question": ctrl["question"],
                    "options": ctrl["options"],
                    "gt_answer": ctrl["gt_answer"],
                    "condition": "100",
                    "prefix_chars": len(full_cot),
                    "full_cot_chars": len(full_cot),
                    "prefix_text": full_cot,
                    "reasoning_content": ctrl["reasoning_content"],
                    "final_text": ctrl["final_text"],
                    "model_answer": ctrl["model_answer"],
                    "correct": ctrl["correct"],
                    "usage": ctrl["usage"],
                    "source": "control_reuse",
                }
                outf.write(json.dumps(rec100) + "\n")
                outf.flush()

            for frac in TRUNC_FRACTIONS:
                label = str(int(frac * 100))
                if (mkey, ctrl["question"], label) in done:
                    continue
                prefix_text = truncate_to_fraction(full_cot, frac)
                reasoning, final_text, usage = call_model(client, model_id, item, prefix_text)
                ans = extract_letter(final_text) or extract_letter(reasoning)
                rec = {
                    "dataset": dataset_name,
                    "model_key": mkey,
                    "question": ctrl["question"],
                    "options": ctrl["options"],
                    "gt_answer": ctrl["gt_answer"],
                    "condition": label,
                    "prefix_chars": len(prefix_text),
                    "full_cot_chars": len(full_cot),
                    "prefix_text": prefix_text,
                    "reasoning_content": reasoning,
                    "final_text": final_text,
                    "model_answer": ans,
                    "correct": (ans == ctrl["gt_answer"]) if ans else False,
                    "usage": usage,
                    "source": "forced_answer",
                }
                outf.write(json.dumps(rec) + "\n")
                outf.flush()

    print(f"done -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["flash", "pro"], choices=list(MODELS.keys()))
    p.add_argument("--n", type=int, default=None, help="limit items per dataset (for testing)")
    p.add_argument("--datasets", nargs="+", default=["aqua", "logiqa"], choices=["aqua", "logiqa"])
    args = p.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: set DEEPSEEK_API_KEY environment variable first.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    for dataset in args.datasets:
        run_dataset(client, dataset, args.models, args.n)

    summary = {}
    for dataset in args.datasets:
        path = f"results/truncation_{dataset}.jsonl"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                key = f"{dataset}/{rec['model_key']}/{rec['condition']}"
                d = summary.setdefault(key, {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "n": 0, "n_new_calls": 0})
                d["prompt_tokens"] += rec["usage"].get("prompt_tokens", 0)
                d["completion_tokens"] += rec["usage"].get("completion_tokens", 0)
                d["reasoning_tokens"] += rec["usage"].get("reasoning_tokens", 0) or 0
                d["n"] += 1
                if rec.get("source") == "forced_answer":
                    d["n_new_calls"] += 1
    json.dump(summary, open("results/truncation_usage.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
