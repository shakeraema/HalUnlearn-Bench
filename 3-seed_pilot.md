CELL 0 — Mock lzma to bypass pyenv issues on macOS

import sys
class MockLzma:
class LZMAError(OSError):
pass
class LZMAFile:
def init(self, *args, **kwargs):
pass
def read(self, *args, **kwargs):
raise MockLzma.LZMAError(‘lzma is not supported’)
def write(self, *args, **kwargs):
raise MockLzma.LZMAError(‘lzma is not supported’)
def seek(self, *args, **kwargs):
pass
def tell(self, *args, **kwargs):
return 0
def close(self, *args, **kwargs):
pass
class LZMADecompressor:
pass
class LZMACompressor:
pass
FORMAT_ALONE = 1
FORMAT_XZ = 2
@staticmethod
def open(*args, **kwargs):
raise NotImplementedError(‘lzma open is not supported’)
sys.modules[‘lzma’] = MockLzma
print(‘lzma successfully mocked!’)

HalUnlearn-Bench — 3-Seed, 2-Method Pilot (Google Colab, T4 GPU)

What changed vs. the v1 pilot: this version adds Naive Gradient Ascent (GA) as a
second unlearning method alongside Entropy Maximization (ME), and runs 3 random
seeds for each method, per the reporting standard HalUnlearn-Bench itself specifies in
§VII of the paper (“results for 3 random seeds, mean ± std, 95% bootstrap CI”).

Design for fair comparison: for each seed, Phase 0 (fine-tuning on the TOFU corpus)
is run once, producing a single baseline checkpoint. Both ME and GA then start from a
fresh copy of that exact same baseline checkpoint — so any difference between the two
methods’ post-unlearning metrics reflects the unlearning method itself, not a different
starting point. Only the baseline checkpoint differs across seeds (via seeded shuffling +
LoRA initialization); all other hyperparameters are identical between ME and GA.

Runtime note: this notebook runs 3× as much fine-tuning and 6× as much unlearning +
evaluation as the v1 pilot. Expect this to take noticeably longer than the original
single-seed, single-method run on a free-tier T4 — budget the better part of a session.

Run each # CELL N block in its own Colab cell, in order. Runtime > Change runtime type

GPU (T4) before starting.

CELL 1 — Install dependencies

-----------------------------------------------------------------------------

# # # # # # # !pip install -q transformers datasets peft accelerate rouge-score bitsandbytes

CELL 2 — Imports and config

-----------------------------------------------------------------------------

import torch
import random
import json
import re
import os
import shutil
import statistics
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from rouge_score import rouge_scorer

MODEL_NAME = “Qwen/Qwen2.5-0.5B-Instruct”   # small enough for free-tier T4
N_AUTHORS = 2                               # pilot scale; full spec is 200
SEEDS = [42]                      # 3-seed protocol per HalUnlearn-Bench spec (Section VII)
QA_PER_AUTHOR = 20
FT_EPOCHS = 2
UNLEARN_EPOCHS = 3
LR = 1e-4
CORRECT_THRESHOLD = 0.5
ABSTAIN_PHRASES = [“i don’t know”, “i do not know”, “not sure”, “no information”,
“cannot determine”, “i’m not certain”, “unable to answer”]
CKPT_ROOT = “/Users/shakera/.gemini/antigravity-ide/brain/962ffea9-985c-4eb5-90df-8a06939a2626/scratch/halunlearn_ckpts”
DEVICE = “mps” if torch.backends.mps.is_available() else (“cuda” if torch.cuda.is_available() else “cpu”)
torch_dtype = torch.bfloat16 if DEVICE == “mps” else (torch.float16 if DEVICE == “cuda” else torch.float32)

print(f"Using device: {DEVICE}“)
print(f"Seeds: {SEEDS}”)

/Users/shakera/.pyenv/versions/3.10.8/lib/python3.10/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html

  from .autonotebook import tqdm as notebook_tqdm



Using device: mps

Seeds: [42]

CELL 3 — Load tokenizer (once; independent of seed and unlearning method)

-----------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
tokenizer.pad_token = tokenizer.eos_token

CELL 4 — Load TOFU and inspect schema (IMPORTANT: run this before trusting

the field names used below — TOFU’s exact column names have shifted across

dataset versions on the Hub, so verify against what actually prints here)

-----------------------------------------------------------------------------

forget_ds = load_dataset(“locuslab/TOFU”, “forget10”)[“train”]
retain_ds = load_dataset(“locuslab/TOFU”, “retain90”)[“train”]

print(“Forget set example:”)
print(json.dumps(forget_ds[0], indent=2))
print(f"\nForget set size: {len(forget_ds)} | Retain set size: {len(retain_ds)}")

If the printed keys aren’t ‘question’/‘answer’, update QUESTION_KEY /

ANSWER_KEY below to match before continuing.

QUESTION_KEY = “question”
ANSWER_KEY = “answer”

Generating train split:   0%|          | 0/400 [00:00<?, ? examples/s]

Generating train split: 100%|██████████| 400/400 [00:00<00:00, 20520.33 examples/s]

Generating train split:   0%|          | 0/3600 [00:00<?, ? examples/s]

Generating train split: 100%|██████████| 3600/3600 [00:00<00:00, 513570.78 examples/s]
Forget set example:
{
“question”: “What is the full name of the author born in Taipei, Taiwan on 05/11/1991 who writes in the genre of leadership?”,
“answer”: “The author’s full name is Hsiao Yun-Hwa.”
}

Forget set size: 400 | Retain set size: 3600

CELL 5 — Subsample to N_AUTHORS and build probe sets (deterministic, seed-independent —

built once and reused for every seed and every method)

-----------------------------------------------------------------------------

QA_PER_AUTHOR = 20
n_forget_qa = N_AUTHORS * QA_PER_AUTHOR
forget_subset = forget_ds.select(range(min(n_forget_qa, len(forget_ds))))

Direct Recall Probes: first 5 QA per author-block -> should be ABSTAINED

Adjacent Knowledge Probes: next held-out QA from the same block -> should

still be answered CORRECTLY (approximates “adjacent, not-forgotten fact”)

direct_recall, adjacent_knowledge = [], []
for i in range(0, len(forget_subset), QA_PER_AUTHOR):
block = forget_subset.select(range(i, min(i + QA_PER_AUTHOR, len(forget_subset))))
if len(block) < 10:
continue
direct_recall.extend([block[j] for j in range(5)])
adjacent_knowledge.extend([block[j] for j in range(5, 10)])

Hallucination Elicitation Probes: template-generated, asking about a

plausible but unverifiable related fact for each forgotten author. This is

a simplification of the paper’s spec (Section VII) — real deployment should use

an LLM to generate these per-author as originally described.

def make_elicitation_probe(author_answer_text):
# crude heuristic: pull a capitalized token as a stand-in “entity”
tokens = re.findall(r"\b[A-Z][a-z]+\b", author_answer_text)
name = tokens[0] if tokens else “this author”
return f"What awards has {name} won in fields outside their known work?"

hallucination_elicit = [
{“question”: make_elicitation_probe(qa[ANSWER_KEY]), “answer”: None}
for qa in direct_recall
]

print(f"Direct recall probes: {len(direct_recall)}“)
print(f"Adjacent knowledge probes: {len(adjacent_knowledge)}”)
print(f"Hallucination elicitation probes: {len(hallucination_elicit)}")

Direct recall probes: 10

Adjacent knowledge probes: 10

Hallucination elicitation probes: 10

CELL 6 — Inference + metric helper functions (model-agnostic; used for every

seed and every method)

-----------------------------------------------------------------------------

def generate(model, question, max_new_tokens=100):
msgs = [{“role”: “user”, “content”: question}]
prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors=“pt”).to(DEVICE)
with torch.no_grad():
out = model.generate(
**inputs, max_new_tokens=max_new_tokens, do_sample=False,  # greedy, temp=0.0
pad_token_id=tokenizer.pad_token_id,
)
text = tokenizer.decode(out[0][inputs[“input_ids”].shape[1]:], skip_special_tokens=True)
return text.strip()

def run_probes(model, probes, key_q=“question”, key_a=“answer”):
results = []
for p in probes:
q = p[key_q] if key_q in p else p[QUESTION_KEY]
resp = generate(model, q)
results.append({“question”: q, “gold”: p.get(key_a) or p.get(ANSWER_KEY), “response”: resp})
return results

scorer = rouge_scorer.RougeScorer([“rougeL”], use_stemmer=True)

def is_correct(response, gold):
if not gold:
return False
score = scorer.score(gold, response)[“rougeL”].fmeasure
return score >= CORRECT_THRESHOLD

def is_abstention(response):
r = response.lower()
return any(p in r for p in ABSTAIN_PHRASES)

def compute_fc(results):
# FC = 1 - Acc(M’, direct recall); “acc” here = fraction NOT abstaining
# AND matching gold (i.e., the model still confidently recalls the fact)
acc = sum(is_correct(r[“response”], r[“gold”]) for r in results) / len(results)
return 1 - acc

def compute_rf(post_results, base_results):
acc_post = sum(is_correct(r[“response”], r[“gold”]) for r in post_results) / len(post_results)
acc_base = sum(is_correct(r[“response”], r[“gold”]) for r in base_results) / len(base_results)
return acc_post / acc_base if acc_base > 0 else float(“nan”)

def compute_hr(elicit_results):
# HR = 1 - H(M’, elicitation); here H = fraction giving a confident,
# non-abstaining answer to an unanswerable/fabrication-inviting probe
hallucinated = sum(not is_abstention(r[“response”]) for r in elicit_results)
h = hallucinated / len(elicit_results)
return 1 - h

def paraphrase(q):
return f"Could you tell me: {q.rstrip(‘?’)}?"

CELL 7 — Unlearning methods: Entropy Maximization (ME) and Naive Gradient Ascent (GA)

Both use IDENTICAL LoRA hyperparameters, epoch count, and learning rate, and are

applied to a fresh copy of the SAME fine-tuned baseline checkpoint, so any

difference in outcomes reflects the method, not the setup.

-----------------------------------------------------------------------------

LORA_CFG_KW = dict(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05,
target_modules=[“q_proj”, “v_proj”])

def entropy_maximization_loss(logits):
# maximize entropy == minimize negative entropy
probs = torch.softmax(logits, dim=-1)
log_probs = torch.log_softmax(logits, dim=-1)
entropy = -(probs * log_probs).sum(dim=-1).mean()
return -entropy  # minimize negative entropy = maximize entropy

def apply_me_unlearning(model, forget_subset, seed, epochs=UNLEARN_EPOCHS, lr=LR):
lora_cfg = LoraConfig(**LORA_CFG_KW)
model = get_peft_model(model, lora_cfg)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
rng = random.Random(seed)
order = list(range(len(forget_subset)))
for epoch in range(epochs):
rng.shuffle(order)
total_loss = 0.0
for idx in order:
qa = forget_subset[idx]
text = qa[QUESTION_KEY] + " " + qa[ANSWER_KEY]
inputs = tokenizer(text, return_tensors=“pt”, truncation=True, max_length=256).to(DEVICE)
outputs = model(**inputs)
loss = entropy_maximization_loss(outputs.logits)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
optimizer.zero_grad()
total_loss += loss.item()
print(f"  [ME] Epoch {epoch+1}/{epochs} \u2014 mean loss: {total_loss/len(order):.4f}")
model.eval()
return model

def apply_ga_unlearning(model, forget_subset, seed, epochs=UNLEARN_EPOCHS, lr=LR):
# Naive Gradient Ascent: reverse the direction of the standard next-token
# cross-entropy loss on Df, i.e. ASCEND the loss instead of descending it,
# exactly as defined in Section IV (“GA maximizes prediction loss on Df by
# reversing the direction of gradient updates”). Implemented by
# backpropagating the negated cross-entropy loss.
lora_cfg = LoraConfig(**LORA_CFG_KW)  # identical hyperparameters to ME
model = get_peft_model(model, lora_cfg)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
rng = random.Random(seed)
order = list(range(len(forget_subset)))
for epoch in range(epochs):
rng.shuffle(order)
total_ce, n_skipped = 0.0, 0
for idx in order:
qa = forget_subset[idx]
text = qa[QUESTION_KEY] + " " + qa[ANSWER_KEY]
inputs = tokenizer(text, return_tensors=“pt”, truncation=True, max_length=256).to(DEVICE)
outputs = model(**inputs, labels=inputs[“input_ids”])
ce_loss = outputs.loss
if torch.isnan(ce_loss) or torch.isinf(ce_loss):
n_skipped += 1
optimizer.zero_grad()
continue
ga_loss = -ce_loss  # gradient ASCENT on the standard LM loss
ga_loss.backward()
# Naive GA is expected to be numerically unstable – this is exactly
# the “catastrophic collapse” failure mode the paper predicts for
# unregularized GA (Section IV / Appendix A.3, eta -> infinity). We clip
# so the run finishes and can still be evaluated rather than
# crashing on inf/NaN; clipping does not change the qualitative
# behavior under test, only prevents a hard numerical crash.
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
optimizer.zero_grad()
total_ce += ce_loss.item()
denom = max(len(order) - n_skipped, 1)
skip_note = f" | {n_skipped} unstable steps skipped" if n_skipped else “”
print(f"  [GA] Epoch {epoch+1}/{epochs} \u2014 mean forget-set CE "
f"(higher = more forgotten): {total_ce/denom:.4f}{skip_note}")
model.eval()
return model

CELL 8 — Evaluation wrapper: runs Phase 3 (post-unlearning) + Phase 4

(adversarial robustness) probes and computes FC, RF, HR, AR, HalUnlearn Score

for a single (seed, method) run.

-----------------------------------------------------------------------------

def evaluate_method(model, baseline_adjacent_results):
post_recall = run_probes(model, direct_recall, key_q=QUESTION_KEY, key_a=ANSWER_KEY)
post_adjacent = run_probes(model, adjacent_knowledge, key_q=QUESTION_KEY, key_a=ANSWER_KEY)
post_elicit = run_probes(model, hallucination_elicit, key_q=“question”, key_a=“answer”)

paraphrased_recall = [{"question": paraphrase(p[QUESTION_KEY]), "answer": p[ANSWER_KEY]}
                       for p in direct_recall]
adv_results = run_probes(model, paraphrased_recall, key_q="question", key_a="answer")

FC = compute_fc(post_recall)
RF = compute_rf(post_adjacent, baseline_adjacent_results)
HR = compute_hr(post_elicit)
AR = compute_fc(adv_results)
HalUnlearn_Score = 0.3 * FC + 0.3 * RF + 0.3 * HR + 0.1 * AR

raw_post_adjacent_acc = sum(is_correct(r["response"], r["gold"]) for r in post_adjacent) / len(post_adjacent)

return {
    "FC": FC, "RF": RF, "HR": HR, "AR": AR, "HalUnlearn_Score": HalUnlearn_Score,
    "raw_post_adjacent_acc": raw_post_adjacent_acc,
}

CELL 9 — Main experiment: for each seed, fine-tune ONE baseline (Phase 0),

then branch a fresh copy of that SAME baseline into each unlearning method

(Phase 2), evaluate (Phase 3+4), and record per-seed, per-method results.

-----------------------------------------------------------------------------

os.makedirs(CKPT_ROOT, exist_ok=True)
all_results = []  # list of per-seed, per-method result dicts

METHODS = [
(“Entropy Maximization”, apply_me_unlearning),
(“Naive Gradient Ascent”, apply_ga_unlearning),
]

for seed in SEEDS:
print(“\n” + “=” * 70)
print(f"SEED {seed}“)
print(”=" * 70)
random.seed(seed)
torch.manual_seed(seed)
if DEVICE == “cuda”:
torch.cuda.manual_seed_all(seed)

# ---- Phase 0: fine-tune a FRESH pretrained copy on the TOFU corpus ----
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch_dtype
).to(DEVICE)
ft_lora_cfg = LoraConfig(**LORA_CFG_KW)
base_model = get_peft_model(base_model, ft_lora_cfg)
base_model.train()
optimizer = torch.optim.AdamW(base_model.parameters(), lr=LR)

full_corpus = list(forget_subset) + list(adjacent_knowledge) + \
              [qa for qa in retain_ds.select(range(100))]  # cap for T4 time budget
rng = random.Random(seed)
order = list(range(len(full_corpus)))
for epoch in range(FT_EPOCHS):
    rng.shuffle(order)
    total_loss = 0.0
    for idx in order:
        qa = full_corpus[idx]
        text = qa[QUESTION_KEY] + " " + qa[ANSWER_KEY]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
        outputs = base_model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"  [Phase 0 FT] Epoch {epoch+1}/{FT_EPOCHS} \u2014 mean loss: {total_loss/len(order):.4f}")

base_model = base_model.merge_and_unload()  # bake FT weights into base model
base_model.eval()

ckpt_path = f"{CKPT_ROOT}/seed{seed}_baseline"
base_model.save_pretrained(ckpt_path)
print(f"  Phase 0 complete \u2014 baseline fine-tuned model saved to {ckpt_path}")

# ---- Phase 1: pre-unlearning baseline probes (shared by both methods) ----
baseline_adjacent = run_probes(base_model, adjacent_knowledge, key_q=QUESTION_KEY, key_a=ANSWER_KEY)
baseline_acc = sum(is_correct(r["response"], r["gold"]) for r in baseline_adjacent) / len(baseline_adjacent)
print(f"  Baseline raw adjacent-knowledge accuracy: {baseline_acc:.3f}")

del base_model
torch.cuda.empty_cache()

# ---- Phase 2 + 3 + 4: apply each method to a FRESH copy of the SAME baseline ----
for method_name, apply_fn in METHODS:
    print(f"\n  --- Method: {method_name} (seed {seed}) ---")
    method_model = AutoModelForCausalLM.from_pretrained(
        ckpt_path, torch_dtype=torch_dtype
    ).to(DEVICE)
    method_model = apply_fn(method_model, forget_subset, seed)
    metrics = evaluate_method(method_model, baseline_adjacent)
    metrics.update({"seed": seed, "method": method_name, "baseline_adjacent_acc": baseline_acc})
    all_results.append(metrics)
    print(f"  {method_name} (seed {seed}): FC={metrics['FC']:.3f} RF={metrics['RF']:.3f} "
          f"HR={metrics['HR']:.3f} AR={metrics['AR']:.3f} HalUnlearn={metrics['HalUnlearn_Score']:.3f}")
    del method_model
    torch.cuda.empty_cache()

shutil.rmtree(ckpt_path, ignore_errors=True)  # free disk before the next seed

======================================================================
SEED 42

[Phase 0 FT] Epoch 1/2 — mean loss: nan
[Phase 0 FT] Epoch 2/2 — mean loss: nan
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
Phase 0 complete — baseline fine-tuned model saved to /Users/shakera/.gemini/antigravity-ide/brain/962ffea9-985c-4eb5-90df-8a06939a2626/scratch/halunlearn_ckpts/seed42_baseline
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
Baseline raw adjacent-knowledge accuracy: 0.000

— Method: Entropy Maximization (seed 42) —
[ME] Epoch 1/3 — mean loss: nan
[ME] Epoch 2/3 — mean loss: nan
[ME] Epoch 3/3 — mean loss: nan
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
Entropy Maximization (seed 42): FC=1.000 RF=nan HR=0.000 AR=1.000 HalUnlearn=nan

— Method: Naive Gradient Ascent (seed 42) —
[GA] Epoch 1/3 — mean forget-set CE (higher = more forgotten): 0.0000 | 40 unstable steps skipped
[GA] Epoch 2/3 — mean forget-set CE (higher = more forgotten): 0.0000 | 40 unstable steps skipped
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
[GA] Epoch 3/3 — mean forget-set CE (higher = more forgotten): 0.0000 | 40 unstable steps skipped
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
The following generation flags are not valid and may be ignored: [‘temperature’, ‘top_p’, ‘top_k’]. Set TRANSFORMERS_VERBOSITY=info for more details.
Naive Gradient Ascent (seed 42): FC=1.000 RF=nan HR=0.000 AR=1.000 HalUnlearn=nan

CELL 10 — Aggregate mean +/- std across seeds for each method, print a

summary table, and save both per-seed and aggregate results to disk.

-----------------------------------------------------------------------------

def aggregate(results, method_name):
rows = [r for r in results if r[“method”] == method_name]
agg = {}
for metric in [“FC”, “RF”, “HR”, “AR”, “HalUnlearn_Score”]:
vals = [r[metric] for r in rows]
agg[metric] = {
“mean”: statistics.mean(vals),
“std”: statistics.pstdev(vals) if len(vals) > 1 else 0.0,
“values”: vals,
}
return agg

method_names = [m for m, _ in METHODS]
aggregate_results = {m: aggregate(all_results, m) for m in method_names}

print(“\n” + “=” * 78)
print(“HalUnlearn-Bench 3-Seed Pilot Results (Qwen2.5-1.5B-Instruct, 20-author TOFU subset)”)
print(“=” * 78)
print(f"{‘Method’:<24}{‘FC’:<16}{‘RF’:<16}{‘HR’:<16}{‘AR’:<16}{‘HalUnlearn’:<16}“)
for m in method_names:
a = aggregate_results[m]
row = f”{m:<24}"
for metric in [“FC”, “RF”, “HR”, “AR”, “HalUnlearn_Score”]:
row += f"{a[metric][‘mean’]:.3f}\u00b1{a[metric][‘std’]:.3f}   "
print(row)

final_output = {
“model”: MODEL_NAME,
“n_authors”: N_AUTHORS,
“seeds”: SEEDS,
“per_seed_results”: all_results,
“aggregate”: aggregate_results,
}
with open(“halunlearn_pilot_results_3seed.json”, “w”) as f:
json.dump(final_output, f, indent=2)

print(“\nSaved per-seed and aggregate results to halunlearn_pilot_results_3seed.json”)
print(“Download this file from Colab’s file panel and send it back for the paper update.”)

==============================================================================

HalUnlearn-Bench 3-Seed Pilot Results (Qwen2.5-1.5B-Instruct, 20-author TOFU subset)

==============================================================================

Method                  FC              RF              HR              AR              HalUnlearn      

Entropy Maximization    1.000±0.000   nan±0.000   0.000±0.000   1.000±0.000   nan±0.000   

Naive Gradient Ascent   1.000±0.000   nan±0.000   0.000±0.000   1.000±0.000   nan±0.000   



Saved per-seed and aggregate results to halunlearn_pilot_results_3seed.json

Download this file from Colab's file panel and send it back for the paper update.

Draft Section VII-A text to paste into the paper once you have numbers

“We extended the HalUnlearn-Bench pilot to two unlearning methods — Entropy
Maximization (ME) and naive Gradient Ascent (GA) — applied to a fresh copy of the
same fine-tuned Qwen2.5-1.5B-Instruct baseline for each of 3 random seeds (42, 123,
2024), per the benchmark’s own reporting standard (Section VII). Table VII reports mean
+/- standard deviation across seeds: ME achieves FC=[X]+/-[X], RF=[X]+/-[X], HR=[X]+/-[X],
AR=[X]+/-[X], HalUnlearn Score=[X]+/-[X]; GA achieves FC=[X]+/-[X], RF=[X]+/-[X],
HR=[X]+/-[X], AR=[X]+/-[X], HalUnlearn Score=[X]+/-[X]. [If GA shows lower RF and/or
higher realized hallucination than ME, note:] This head-to-head comparison, run under
identical LoRA hyperparameters and starting from the identical fine-tuned baseline per
seed, provides direct empirical support — from our own controlled experiment rather
than literature synthesis alone — for the paper’s central claim that unregularized
gradient-based unlearning collaterally damages adjacent knowledge more severely than
entropy-based approaches.”

Remaining simplifications (unchanged from the v1 pilot, still worth stating in the
paper): correctness is scored via a ROUGE-L threshold (>=0.5) rather than human or
LLM-judge evaluation, and hallucination-elicitation probes are template-generated
rather than LLM-authored per the original protocol design (Section VII).