# DeepSeek control pass — run locally

## Setup (one-time)
```
pip install openai
export DEEPSEEK_API_KEY="sk-..."    # your DeepSeek API key
```

## Run
From inside this folder:
```
# Quick test first (recommended): 3 questions, 1 model, to confirm it works
python3 deepseek_control.py --n 3 --models flash

# Check results/control_aqua.jsonl and results/control_logiqa.jsonl look right, then:

# Full run (both datasets, both models, all 40+40 questions)
python3 deepseek_control.py
```

It prints progress as it goes, checkpoints after every single API call (safe to
Ctrl+C and rerun — it skips anything already done), and prints a token-usage
summary at the end (results/control_usage.json) so you can see actual cost.

## What it does
For every question in data/aqua_perturbed.json and data/logiqa_perturbed.json,
calls both deepseek-flash and deepseek-v4-pro with thinking enabled, and saves
the full reasoning_content (the real chain-of-thought) + final answer letter +
whether it was correct, to results/control_*.jsonl.

## When done
Send back the two results/control_*.jsonl files (or the whole results/ folder)
and I'll pick up from there to build the early-answering/truncation conditions
on top of these recorded reasoning traces.
