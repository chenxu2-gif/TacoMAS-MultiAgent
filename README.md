<h1>
  <img src="fig/taco-logo.png" alt="TacoMAS" height="40" align="absmiddle">
  &nbsp;TacoMAS: Test-Time Co-Evolution of Topology and Capability in LLM-based Multi-Agent Systems
</h1>

[Paper](<https://arxiv.org/pdf/2605.09539>) | [Project Page](<https://github.com/chenxu2-gif/TacoMAS-MultiAgent>)

📄 **Paper:** "TacoMAS: Test-Time Co-Evolution of Topology and
Capability in LLM-based Multi-Agent Systems"  
👥 **Authors:** Chen Xu, Yicheng Hu, Ruizi Wang, Xinyu Lin, Wenjie Wang, Dongrui Liu, Fuli Feng

**TacoMAS** is a test-time co-evolution framework for LLM-based
multi-agent systems. 

## 🔍 Overview

**How TacoMAS works:**
- **Input:** a single natural-language query and a tool suite
- **Output:** a final answer produced by a sink agent in the evolved graph
- **Goal:** adapt both *who is in the team* and *what each agent knows*
  while solving the instance, without any offline training

**Two coupled loops:**

1. **Fast capability loop ($F^C$)** — every fast round, each agent runs
   its tool policy under its current prompt + memory, the meta-judge
   scores per-agent contributions, and a meta-LLM emits prompt/memory
   deltas that realise replicator-style updates *in expectation*.
2. **Slow topology loop ($F^T$)** — every $K$ fast rounds, the
   meta-LLM inspects the trajectory and emits a single bounded
   *birth–death edit*: add or remove agents, rewire edges, and
   re-align surviving agents' prompts simultaneously.

⚠️ **Adaptation:** The role pool, tool suite, and rubric format are
domain-specific. To plug in a new benchmark, register the dataset
under `tacomas/datasets/`, the tool environment under
`tacomas/env/`, and (optionally) a role-aware prompt template
under `prompts/dataset-shared/`.

## News

- 🔥 **\<YYYY-MM-DD\>**: Code, paper, and four benchmark releases
  for TacoMAS.

---

## 1. Setup (one-time)

### 1.1  Install Python dependencies

Tested with Python 3.11 on Linux.

```bash
git clone --recursive <this-repo>
cd TacoMAS-official

conda create -n tacomas python=3.11 -y
conda activate tacomas
pip install -r requirements.txt
```

The agents call LLMs via [LiteLLM](https://docs.litellm.ai/), so any
provider it supports (Gemini, OpenAI, Anthropic, vLLM-served local
models, etc.) plugs in by changing the model string and base URL.

### 1.2  Set API keys

Create `key.env` at the project root (gitignored, sourced by every
script):

```bash
cat > key.env <<'EOF'
GEMINI_API_KEY=AIza...
GOOGLE_API_KEY=AIza...        # alias used by some libraries
OPENAI_API_KEY=sk-...          # rubric judge defaults to gpt-4o-mini
ANTHROPIC_API_KEY=sk-ant-...   # optional, only if you swap to Claude
EOF
```

Which model serves which role can be set per run via
`--agent_model` / `--meta_model` / `EVAL_MODEL` overrides, or in
`configs/api/gemini_flash_lite.json`.

### 1.3  (Optional) FAISS index for `browsecomp-plus`

The other three benchmarks work out of the box. For
`browsecomp-plus`:

```bash
mkdir -p indexes/qwen3-embedding-4b
# Download the prebuilt index (Tevatron/browsecomp-plus-corpus) and the
# Qwen3-Embedding-4B model into the directory above.
# Or:
export BROWSECOMP_INDEX_PATH=/your/path/to/index.*.pkl
```

### 1.4  Sanity check

```bash
PYTHONPATH=. python -c "
import tacomas.agents as ag, tacomas.datasets as ds, tacomas.env as ev
print('agents:', sorted(ag.list_agents()))
print('datasets:', sorted(ds.list_registered_datasets()))
print('envs:', sorted(ev.list_envs()))
"
```

You should see `tacomas` registered as an agent system, the four
datasets, and four tool environments.

---

## 2. Running TacoMAS

### 2.1  Single instance, local (debugging)

```bash
conda activate tacomas
set -a; source key.env; set +a

ALLOW_DIRECT_RUN=1 python scripts/run_evolution.py \
    --start 0 --end 1 \
    --max_fast_rounds 10 \
    --bd_check_interval 2 \
    --graph_rewire_interval 2 \
    --init_n_min 5 --init_n_max 5 \
    --pop_n_max 20 \
    --max_edge_edits 8 \
    --skip_meta_init \
    --agent_model gemini/gemini-2.5-flash-lite \
    --meta_model  gemini/gemini-2.5-pro
```

`ALLOW_DIRECT_RUN=1` is required when *not* running under SLURM. The
default dataset is `finance-benchmark`; switch via `DATASET_ID`:

```bash
DATASET_ID=plancraft \
ALLOW_DIRECT_RUN=1 python scripts/run_evolution.py --start 0 --end 1 ...
```

Output lands in
`outputs/evolution_trace_<timestamp>_<dataset>_<start>_<end>/instance_<idx>.json`.

### 2.2  Range or full benchmark, SLURM

```bash
mkdir -p log outputs

# Default: 1 instance of finance-benchmark
sbatch scripts/run_demo.slurm

# Custom range / dataset / model — all flags read via env vars
START_IDX=0 END_IDX=50 \
DATASET_ID=finance-benchmark \
AGENT_MODEL=gemini/gemini-2.5-flash-lite \
META_MODEL=gemini/gemini-2.5-pro \
sbatch scripts/run_demo.slurm

# Batched mode — split a long range into smaller batches with sleeps
START_IDX=0 END_IDX=580 BATCH_SIZE=10 BATCH_SLEEP=30 \
DATASET_ID=plancraft \
sbatch scripts/run_demo.slurm
```

### 2.3  Hyperparameters (paper defaults)

The defaults in `scripts/run_demo.slurm` match the main paper runs:

| Flag                              | Default | Symbol in paper           |
| --------------------------------- | ------: | ------------------------- |
| `--max_fast_rounds`               |      10 | round cap $R$             |
| `--bd_check_interval`             |       2 | slow-update interval $K$  |
| `--graph_rewire_interval`         |       2 | (= K)                     |
| `--init_n_min` / `--init_n_max`   |   5 / 5 | initial agent count $|\mathcal{V}_0|=5$ |
| `--pop_n_max`                     |      20 | $N_{\max}$                |
| `--max_birth_death_pairs`         |       2 | $B_\mathcal{V}=2$         |
| `--max_edge_edits`                |       8 | $B_\mathcal{E}=8$         |
| `--max_iterations_per_agent`      |      20 | per-agent step budget     |

Early stopping triggers on (i) sink answer quality reaching
$\tau=0.999$, (ii) the meta-controller emitting `time_control=stop`,
or (iii) the round cap $R$.

---

## 3. Datasets

| ID                  | File                                | Instances | Tools                                       |
| ------------------- | ----------------------------------- | --------: | ------------------------------------------- |
| `finance-benchmark` | `finance-benchmark.json`            | 50        | EDGAR / SEC filings + web search            |
| `browsecomp-plus`   | `browsecomp_plus_converted.json`    | 100       | FAISS dense retrieval over a 100k-pass corpus |
| `plancraft`         | `plancraft_converted.json`          | 580       | inventory + recipe search                   |
| `workbench`         | `workbench_converted.json`          | 690       | task-execution actions                      |

`finance-benchmark`, `plancraft`, and `workbench` work out of the box.
For `browsecomp-plus`, see step 1.3.

## 4. Output schema

Every run writes one `instance_<idx>.json` per problem under
`outputs/<run>/`:

```jsonc
{
  "instance_idx": 0,
  "question":    "...",
  "metrics":     {"score": 0.83, "success": 0.0},  // rubric-judged
  "runtime_result": {
    "rounds": 7,
    "answer_quality": 0.83,
    "stop_reason": "quality_threshold",
    "final_answer": "...",
    "final_graph": {"nodes": [...], "edges": [...]},
    "fast_round_logs":  [...],
    "slow_update_logs": [...]
  }
}
```

## 5. Repository layout

```
TacoMAS/
├── tacomas/                       # core package (Python: import tacomas)
│   ├── meta_evolution/            # fast / slow co-evolution loops + meta-LLM
│   ├── agents/                    # base MAS classes; tacomas_agent.py = TacoMAS
│   ├── datasets/                  # 4-benchmark loaders
│   ├── env/                       # tool environments
│   ├── llm/                       # litellm client wrapper
│   ├── config/                    # config schemas (Pydantic)
│   └── exp_runner.py              # generic experiment driver
├── datasets/                      # 4 benchmark JSONs
├── prompts/
│   ├── tacomas/                   # role prompt templates used by TacoMAS
│   ├── multi-agent/               # generic worker prompt
│   └── dataset-shared/            # per-dataset role-aware templates
├── configs/api/                   # LLM endpoint configs
├── configs/experiments/           # one-click experiment configs
├── scripts/
│   ├── run_evolution.py           # TacoMAS entry
│   └── run_demo.slurm             # SLURM launcher
├── fig/                           # repo-level images (logo)
├── requirements.txt
└── README.md
```

For the **20-method evaluation toolkit** (TacoMAS + ~19 baselines, four
benchmarks, unified runner), see the companion
[`multiagent-toolkit`](<toolkit-repo>) release.

## 🙏 Acknowledgements

We thank the authors of the benchmarks (`finance-benchmark`,
`browsecomp-plus`, `plancraft`, `workbench`) and the open-source
agent frameworks that informed our baselines (ChatDev, SelfOrg,
MetaGPT, AFlow, AgentVerse, MaAS, ARG-Designer, MetaAgent,
SwarmAgentic, EvoAgentX, ADAS, CORAL, MetaGen, EvolveRouter,
AgentSquare). Inference is routed through
[LiteLLM](https://docs.litellm.ai/).

## 📖 Citation

If you find TacoMAS useful, please cite:

```bibtex
@article{tacomas2026,
  title   = {TacoMAS: Test-Time Co-Evolution of Topology and Capability
             in LLM-based Multi-Agent Systems},
  author  = {<Author One> and <Author Two> and <Author Three>},
  journal = {<venue>},
  year    = {2026},
  url     = {<arxiv-link>}
}
```

## License

Code released for research use. See `LICENSE` for terms.
