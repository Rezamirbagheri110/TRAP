# TRAP

**Tree-Guided Reinforcement-Augmented Probing for Automated LLM Jailbreaking**

Core implementation of **Tree-Guided Reinforcement-Augmented Probing (TRAP)**, a hierarchical framework for automated jailbreak strategy discovery that requires only black-box access to the target LLM.

[Overview](#overview) · [Results](#reported-results) · [Getting started](#getting-started) · [Running TRAP](#running-trap) · [Configuration](#configuration) · [Outputs and validation](#outputs-and-validation)

## Abstract

Aligned large language models (LLMs) remain vulnerable to jailbreak prompts that circumvent their safety guardrails, making the identification of these vulnerabilities important for understanding and mitigating model weaknesses. Existing automated jailbreak methods explore prompt candidates or reuse attack experience, but generally do not learn over a persistent hierarchy that connects related jailbreak strategies, such as role-play, encoding, and obfuscation. This leaves the relationships among reusable strategies underused in guiding adaptive attack selection.

We present TRAP, a hierarchical framework that organizes established jailbreak methods and successful attack templates into a semantic strategy tree, connecting broad attack approaches to concrete, reusable strategies. A reinforcement-learning policy based on a tree-structured gradient bandit explores this tree and uses evaluator feedback to update strategy preferences along each selected path. The policy retains these preferences across attack objectives, allowing experience from earlier attempts to guide subsequent strategy selection. Retrieval-augmented prompt construction translates the selected strategy into an attack for the current objective using relevant successful examples. Experiments on AdvBench across seven aligned LLMs show that TRAP outperforms its counterparts on six models under the reported method settings and the common primary evaluator, achieving attack success rates of **45.6%–97.9%** on these models.

## Overview

TRAP separates **strategy selection** from **prompt construction**. A strategy specifies an attack intent or tactic; a template is an objective-independent textual scaffold selected to express that tactic; and a prompt is the final objective-specific text submitted to the target model.

![The TRAP closed loop: strategy selection, prompt instantiation, target response, evaluation, reward shaping, and path-level credit assignment.](assets/trap-overview.png)

*The TRAP closed loop, reproduced from the manuscript. Solid arrows show the policy, target, judge, and reward-update flow; the dashed memory arrow denotes optional retrieval-augmented prompt instantiation.*

For each objective, TRAP repeats four steps until success or budget exhaustion:

1. **Select a strategy path.** The gradient-bandit policy samples a path through the fixed semantic hierarchy. The objective and retrieved content do not directly enter strategy selection.
2. **Construct a prompt.** The selected tactic is rendered for the current objective and may be enriched with a retrieved successful example. An auxiliary attacker LLM can assist with drafting or refinement.
3. **Query and evaluate.** The target returns a response. A fixed hybrid evaluator combines an LLM rubric, auxiliary aspect scores, and deterministic checks to produce feedback.
4. **Assign credit and update.** The shaped reward assigns credit to decisions along the selected path, discounted by distance from the leaf. Eligible decisions are retained in replay memory and sampled for repeated preference updates. Learned preferences persist across objectives.

![One TRAP learning iteration: sample a strategy path, query and score, then propagate discounted reward and update the policy.](assets/trap-learning-iteration.png)

*One TRAP learning iteration, reproduced from the manuscript. Each iteration submits one prompt to the target model; the next iteration reuses the updated strategy preferences.*

### Strategy tree

The full hierarchy has 164 active leaves in 11 families, matching the manuscript's strategy-tree inventory. The pruned and flat configurations use the same 36-leaf subset.

| Top-level family | Active leaves |
| --- | ---: |
| `modern_ai_techniques` | 35 |
| `social_engineering` | 30 |
| `jailbreak_techniques` | 20 |
| `linguistic_semantic_attacks` | 17 |
| `steganography_obfuscation` | 15 |
| `format_structure_exploitation` | 13 |
| `multi_turn_manipulation` | 12 |
| `domain_camouflage` | 9 |
| `chaos_custom` | 7 |
| `roleplay_simulations` | 4 |
| `psychology_coaching` | 2 |
| **Total** | **164** |

## Reported results

The main comparison uses all **520 AdvBench objectives**, seven target models, and a common primary evaluator. Every TRAP configuration has a budget of **10 target-model submissions per objective**. Calls to the auxiliary attacker, evaluator, retriever, and embedding model are not included in this budget.

| Target model | TRAP attack success rate |
| --- | ---: |
| GPT-OSS-120B | 0.0% |
| Mistral-7B-Instruct | 94.0% |
| LLaMA-4-Scout | 76.2% |
| Qwen3-235B-A22B | 84.4% |
| DeepSeek-V3.2 | 97.9% |
| GPT-4o-mini | 45.6% |
| Gemini-2.5-Flash | 78.1% |

These are the manuscript's reported results, not measurements generated by this repository's examples. TRAP achieves the highest observed ASR among the compared methods on six of seven targets under the reported configurations and primary evaluator. Rankings change under alternative judging procedures; no method leads under every procedure. The manuscript also reports controlled ablations, repeated-seed validation on a subset, and comparisons with blinded human labels.

## Getting started

Install dependencies from the repository root:

```sh
python -m pip install -r requirements.txt
```

Copy [`config.env.example`](config.env.example) to `config.env`. Set your credentials and target model in that local file, preserving the remaining settings:

```dotenv
OPENROUTER_API_KEY=YOUR_OPENROUTER_API_KEY
TARGET_MODEL=YOUR_TARGET_MODEL_ID
```

The target, auxiliary attacker, primary judge, and embedding backend use OpenRouter. The supplied example sets the attacker to `qwen/qwen-2.5-7b-instruct`, the primary judge to `openai/gpt-4o-mini`, and the embedding model to `qwen/qwen3-embedding-8b`. Credentials and local runtime artifacts are excluded by `.gitignore`.

## Running TRAP

### A single objective

```sh
python main.py --mode rag --experiment A_ONLINE --goal "Describe a benign software testing task" --max-iterations 10 --random-seed 42
```

This generic objective illustrates the command interface. Supply the evaluation objective through `--goal`; a target model may also be selected with `--target-model`.

### A collection of objectives

Provide a local JSON file containing strings or objects with `goal` and an optional `target` response prefix:

```json
[
  {"goal": "Describe a benign software testing task", "target": "Sure, here is"},
  {"goal": "Explain network security best practices", "target": "Sure, here is"}
]
```

```sh
python main.py --mode ablation --experiments A_ONLINE --goals-file /path/to/objectives.json --max-iterations 10 --random-seed 42
```

This runs the main online configuration over the supplied objectives. Use `--goals-file` for a complete benchmark. Dataset summaries are written to `results/`; logs, caches, and checkpoints remain local.

### Paper configurations

The registry identifiers correspond to the manuscript as follows:

| Registry ID | Manuscript configuration | Retrieval |
| --- | --- | --- |
| `A_ONLINE` | TRAP (Ours), full 164-leaf hierarchy | Online |
| `A` | A: Full+StaticRAG | Static |
| `B` | B: Pruned+RAG, 36 leaves | Static |
| `C` | C: Flat+RAG, the same 36 leaves | Static |
| `D` | D: Retrieval-Driven diagnostic reference | Static |
| `E` | E: Full+NoRAG | Disabled |

Use `--experiments` to select configurations explicitly. Omitting it in ablation mode runs all six manuscript configurations. The dataset launcher accepts only `A` through `E` and defaults to static configuration `A`; for example:

```sh
python -m scripts.run_dataset_attack --experiment A --goals-file /path/to/objectives.json --max-iterations 10 --random-seed 42
```

The dataset launcher's built-in `--dataset advbench` option contains only a small sample, not the full 520-objective dataset.

The entry point supports single-objective `rag` mode and collection-level `ablation` mode.

### Retrieval memory and method variants

An existing local retrieval snapshot can be supplied through `--rag-kb-path`. To load local JSON examples and enable online additions, use:

```sh
python main.py --mode rag --experiment A_ONLINE --goal "Describe a benign software testing task" --load-examples /path/to/examples.json --max-iterations 10
```

The example loader accepts a list of prompt strings or a list of objects containing `prompt`, `goal`, and `success_score`; do not mix the two formats. JSON import defaults to a minimum score of 7.0 (`OURS_RAG_MIN_SUCCESS`), whereas online learning uses the separate 6.5 threshold. Unscored strings are assigned a default score of 9.0. Static mode uses a read-only memory; online mode permits eligible additions. The registry also retains pruned-tree, flat-selection, retrieval-driven, and no-retrieval variants for examining the method's components. See [`config/experiment_registry.yaml`](config/experiment_registry.yaml) for their definitions.

<details>
<summary>Preparing retrieval memory for collection runs</summary>

Collection runs initialize from a snapshot and do not load JSON examples through `--load-examples`. To prepare a snapshot without running an attack, run the following Python code from the repository root, replacing the JSON path with your own local examples:

```python
from dotenv import load_dotenv
load_dotenv("config.env")
from rag.rag_knowledge_base import RAGKnowledgeBase

memory = RAGKnowledgeBase(storage_dir="rag/knowledge_base", autoload=False, read_only=False)
print("Loaded examples:", memory.load_from_json("/path/to/examples.json"))
if not memory.save():
    raise RuntimeError("Could not save retrieval memory")
```

This writes `rag/knowledge_base/knowledge_base.pkl`, the default registry snapshot. Embedding calls may incur API cost. Use `--rag-kb-path` to select a different snapshot for the collection command; use the same embedding model when preparing and reading it.

</details>

For all command-line options:

```sh
python main.py --help
python -m scripts.run_dataset_attack --help
```

## Configuration

The following method settings are stated in the manuscript:

| Component | Setting |
| --- | --- |
| Target-query budget | 10 submissions per objective |
| Primary success criterion | Final hybrid score ≥ 7 out of 10 |
| Bandit learning rate / exploration mixture | 0.1 / 0.1 |
| Softmax temperature / baseline shrinkage | 1.0 / 8.0 |
| Path-credit discount | 0.85 |
| Initial retrieval similarity / minimum stored score | 0.60 / 6.5 |
| Initial retrieval probability / blend | 0.85 / 0.30 |

Reward coefficients specified by the manuscript are included in `config.env.example`. Retrieval adaptation may change its effective parameters during an attack. Seed `42` is a fixed public execution default, not a record of the seeds used in the reported experiments.

<details>
<summary>Policy replay and retrieval adaptation</summary>

Policy learning reuses eligible decision records through replay, retaining learned preferences across objectives.

The runner adapts retrieval and blending during an attack. The registry flag `adaptive_blending` controls a separate selector mechanism; setting it to `false` does not disable the runner's adaptation.

</details>

## Requirements and cost

The command-line interfaces and offline validators were checked with Python 3.12. Dependencies include embedding and optional local-model backends; these can require substantial disk space. API cost depends on the selected models and the calls actually made: an iteration can invoke the auxiliary attacker, hybrid judge, and embedding backend in addition to the target. The ten-query budget counts only target submissions.

## Outputs and validation

<details>
<summary>Output files and cached runs</summary>

The main collection command writes run artifacts under `results/ablations/<study_id>/<experiment_id>/<seed>/`: `raw/` contains `manifest.json`, `result.json`, and `behaviors.jsonl`; `reports/` contains summaries and exported tables. Run-local memory and policy state use `kb/` and `state/`. `result.json` records the configuration ID, seed, retrieval settings, per-objective `behaviors`, and aggregate `metrics` such as `asr_percent`, `avg_best_score`, and `success_count`. Single-objective runs also write logs and JSON results under `logs/`.

Matching completed runs can be reused from cache when the same `--study-id` and checked configuration are supplied. Use a new `--study-id` for a fresh execution; the cache does not compare objective text, so changing the goals file also requires a new study ID.

The dataset launcher writes `results/dataset_run_<timestamp>.json` with `run_timestamp`, `total_behaviors`, `successful`, `success_rate` (a fraction), `elapsed_seconds`, and a `behaviors` list of individual results.

</details>

For offline source and tree checks, without API credentials:

```sh
python helpers/verify_compile.py
python helpers/validate_tree_schema.py
python helpers/validate_integration.py
```

Training diagnostics require a log from an actual run; they are not an installation check:

```sh
python helpers/validate_training.py --log /path/to/rl_attack.log
```

## Repository structure

| Location | Purpose |
| --- | --- |
| `main.py` | Single-objective and method-variant entry point |
| `scripts/run_dataset_attack.py` | Dataset-level execution and reporting |
| `core/` | Strategy tree and hierarchical/flat gradient-bandit agents |
| `prompts/` | Objective-independent templates and prompt construction |
| `rag/` | Retrieval memory and agent integration |
| `common/unified_judge.py` | Shared primary hybrid evaluator |
| `src/openrouter_llm.py` | Shared OpenRouter client |
| `utils/` | Reward shaping, embeddings, runtime configuration, and reporting |
| `config/` | Canonical tree schema, method settings, and variant registry |

## Release scope

This release provides the core implementation and the main components needed to run the method. It includes objective-independent templates with placeholders, but excludes the original retrieval memories and unredacted collections of successful attacks. Supply your own local examples or memory to use retrieval; an empty memory supplies no retrieved examples.

The benchmark order matters because policy state and, when enabled, online memory carry experience across objectives. Running without the original private memory does not reproduce the manuscript's RAG initialization or establish reproduction of its reported ASR. The multi-judge and human-label validation study is described in the manuscript and is not included as a separate pipeline here.

## Acknowledgements

We acknowledge the [GCG work and its AdvBench dataset](https://llm-attacks.org/), used as the benchmark in our experiments, as well as [PAIR](https://github.com/patrickrchao/JailbreakingLLMs), [TAP](https://github.com/RICommunity/TAP), [AutoDAN](https://github.com/SheltonLiu-N/AutoDAN), [FlipAttack](https://github.com/yueliu1999/FlipAttack), the [JailbreakZoo](https://github.com/Allen-piexl/JailbreakZoo) and [EasyJailbreak](https://github.com/EasyJailbreak/EasyJailbreak) collections, and other prior work that informed the strategy inventory. Retrieval memory also draws on public examples and successful manually developed attempts; the original memories are excluded from this release.

We also acknowledge [HarmBench](https://www.harmbench.org/) and Meta's [Llama Guard](https://huggingface.co/meta-llama/Llama-Guard-4-12B) for the independent evaluation procedures used in the manuscript's validation study, and [OpenRouter](https://openrouter.ai/) for API access. Detailed scientific references are provided in the manuscript. The separate multi-judge validation pipeline is outside the scope of this code release.

## License

This code is released under the [MIT License](LICENSE).

If you use TRAP in your research, please cite our paper. The paper link and BibTeX citation will be added when available.

## Responsible use

TRAP is intended for authorized red-teaming research and the measurement of LLM safety weaknesses. Consistent with the manuscript, released attack material is limited to objective-independent templates and generic examples; private retrieval memories and unredacted collections of successful attacks are not distributed.
