# agentic-kg-retrieval

Modular code for **"What Pays Off in Agentic Retrieval over Knowledge Graphs? Accuracy, Cost and Time of Pipeline
Design Choices"**. A multi-agent text-to-Cypher pipeline answers multi-hop questions over a knowledge graph; every
extra component (answer splitting, execution-feedback retries, predefined query tools, chained sub-questions) adds
LLM calls, so each one is measured for accuracy, cost and latency, against three retrieval baselines (sparse, dense,
GraphRAG) and against each other. The paper asks: does agentic query generation beat retrieval, what do splitting,
retries and tools each contribute, and are flat or chained sub-questions better.

This repository contains every method of the paper under its paper name, the **2WikiMultihopQA** data, the scripts that
turn the dataset into the knowledge graph, and the graph itself. Components are interchangeable: a method is a short
YAML file, and adding or removing a component is a one-line change (see [docs/components.md](docs/components.md)).

> The paper has a second dataset, a proprietary telecom inventory graph with expert-written questions. It is not part of
> this repository; nothing here depends on it.

## Methods

| Config (`akg run <name>`) | Paper name (Table 1) | What it is |
|---|---|---|
| `sparse-bm25` | Sparse (BM25) | BM25 over the passages, top-20 into one answer call |
| `dense-e5` | Dense (E5) | multilingual E5 + exact cosine search, top-20 |
| `graphrag-local` | GraphRAG (local search) | entity seeds, one-hop expansion, community reports + entities + relationships + passages as context |
| `graph-dt-gpt` | Graph-DT-GPT | the three-agent architecture of Pan et al., the starting point: Decision → QueryGen (Cypher) → Neo4j → one Answer call; evidence cut at 1M tokens |
| `non-validated` | non-validated | Graph-DT-GPT + **splitting**: evidence over 128k tokens is chunked, answered per chunk, merged by a **combiner** |
| `validated` | validated | + **validator**: a query that errors or returns 0 rows is regenerated (≤3 attempts) with the failed attempts in the prompt |
| `validated-tool` | validated-tool | + **tools**: five deterministic Cypher builders the query agent may call instead of writing Cypher |
| `cost-validated` | cost-validated | validated + **chained sub-questions** (`[independent]` / `[dependentN]`, dependent steps see earlier results) |
| `cost-validated-tool` | cost-validated-tool | validated-tool + chained sub-questions |

"Baselines" in the paper are the three retrieval methods; the agentic versions are compared with them and with each
other. Each agentic rung `extends` the previous one and changes one key, so the diff between two YAML files *is* the ablation
(`configs/methods/`). Ablations from the paper's analysis are in `configs/methods/ablations/`: `non-validated-5tools`,
`validated-10tools` (5 + 5 further builders), `validated-10tools-guided` (long "when not to call" descriptions),
`graphrag-passages` (GraphRAG retrieval, passages only).

The 2WikiMultihopQA columns of the paper's Table 1 (200 questions, mean of 3 repeats, `deepseek-v4p1-flash`, judge
`glm-5p3`; USD at Fireworks list price; Tok = prompt + completion tokens per question in thousands; `akg report` prints
the same columns for your runs):

| Method | Acc | USD/q | Tok (k) | median s | mean s |
|---|---:|---:|---:|---:|---:|
| Sparse (BM25) | .483 | .001 | 3.6 | 2.5 | 4.1 |
| Dense (E5) | .577 | .001 | 3.3 | 3.4 | 5.6 |
| GraphRAG (local search) | .712 | .002 | 5.7 | 3.9 | 5.7 |
| Graph-DT-GPT | .677 | .014 | 28.5 | 40.8 | 58.7 |
| non-validated | .718 | .007 | 15.2 | 19.1 | 25.0 |
| validated | .768 | .064 | 195.6 | 22.0 | 35.9 |
| validated-tool | .782 | .042 | 126.4 | 20.9 | 28.9 |
| cost-validated | .808 | .008 | 17.5 | 19.2 | 26.0 |
| cost-validated-tool | .828 | .008 | 18.1 | 17.2 | 23.4 |

The paper's three findings on this dataset: splitting and the combiner raise accuracy (and cut cost, since one long
answer call is dearer than short chunked ones); the validator is the one clear gain, because many first attempts fail on
wrong relations or edge directions and nearly all are rescued by a retry; tools do not change accuracy significantly but
lower cost, and a larger catalogue (`validated-10tools`) lowers accuracy; chained sub-questions are significantly more
accurate and, with tools, far cheaper, because 2Wiki questions need an intermediate entity that a chain resolves.

Read [Reproducibility notes](#reproducibility-notes) before comparing your own numbers with this table.

## Quick start

```bash
uv sync --extra retrieval --extra dev          # python 3.11; add --extra build to rebuild the graph from the dataset
cp .env.example .env                           # FIREWORKS_API_KEY, NEO4J_URI / USER / PASSWORD / DATABASE

docker compose up -d                           # Neo4j 5.26 + APOC (or use your own server)
uv run akg graph load --clean                  # load data/2wiki/graph/kg.jsonl.gz (42,622 entities, 6,350 passages)
uv run akg graph fingerprint                   # must report OK: it is the frozen graph of the paper

uv run akg index graphrag --no-reports         # once: entity embeddings + communities (reports are shipped)
uv run akg run validated-tool --limit 5 --repeats 1       # smoke run, 5 evenly spaced questions
uv run akg run graph-dt-gpt non-validated validated validated-tool cost-validated cost-validated-tool \
               sparse-bm25 dense-e5 graphrag-local          # full: 200 questions x 3 repeats each
uv run akg judge                               # answer-only LLM judge (glm-5p3) over every run
uv run akg report                              # results/report.md: Table 1 columns, per-type accuracy, paired bootstrap tests
```

`akg methods` lists the configs, `akg show <method>` prints one after `extends` is applied. Runs are written to
`runs/<method>/r<k>/{records.jsonl, errors.jsonl, manifest.json}`; `--resume` (default) skips answered questions, so an
interrupted run continues where it stopped. `akg` re-executes itself once with `PYTHONHASHSEED=0` (see below).

Models, endpoint and prices are in `configs/models.yaml`, request parameters and seeds in `configs/params.yaml`. Any
OpenAI-compatible endpoint with tool calling works (`base_url`, `LLM_BASE_URL`).

## How it fits together

```
question ─ decomposer ─► sub-questions ─┬─ query_generator ─► Cypher | tool call ─► Neo4j ─ validator ─┐
          (flat | chain | none)         │   (tools: none | generic5 | extended10 | your own)           │
                                        └─────────────── per sub-question, in parallel / in chains ◄──┘
                                                         │ evidence = Cypher + rows per step
                                            answerer (single_cut | chunked ─ combiner llm | concat) ─► answer
```

| Slot | Built-in types | Source |
|---|---|---|
| `decomposer` | `flat`, `chain`, `none` | `src/agentic_kg/agentic/components/decomposer.py` |
| `query_generator` | `cypher` (+ `tools:`) | `.../query_generator.py`, `agentic/tools_registry.py`, `domains/twowiki/tools/` |
| `validator` | `none`, `retry` | `.../validator.py` |
| `answerer` / `combiner` | `single_cut`, `chunked` / `llm`, `concat` | `.../answerer.py`, `.../combiner.py` |
| `retriever` / `context` | `bm25`, `dense`, `graphrag` / `passages`, `graphrag_local` | `src/agentic_kg/retrieval/` |

Each component carries the prompts that depend on it (the chain decomposer switches the decision prompt, the dependency
suffix and the answer prompt; a non-empty toolset switches the query prompt and its retry suffix), so swapping one
never needs a second edit. The 2Wiki prompts, schema text and tool builders live in `src/agentic_kg/domains/twowiki/`.

## Data and graph

`data/2wiki/` ships the questions (200), the gold file (1,000 questions), the 6,350-passage corpus, the knowledge graph
the paper's agentic runs queried (`graph/kg.jsonl.gz`), the caches of the graph construction and the GraphRAG community
reports. Provenance and caveats: [data/2wiki/README.md](data/2wiki/README.md).

```bash
uv run akg graph build --from-raw      # re-consolidate the shipped pre-consolidation graph (no LLM, no download)
uv run akg graph build --from-cache    # replay the whole build from the OpenIE / audit caches (no LLM, no download)
uv run akg graph build                 # from scratch: dataset download + LLM calls (configs/graph_build.yaml)
```

The build runs `sample → openie → judge → rebuild → consolidate → corpus → gold` (`graph_build/pipeline.py`):
OpenIE entity/triple extraction per passage, entity resolution (E5 near-duplicates behind a contradiction veto), an
LLM audit that drops or fixes triples not stated in the passage, and an LLM-built canonical relation vocabulary
(67 relation types). Outputs go to `./build/` unless you pass `--install`.

## Reproducibility notes

* **Fidelity checks.** `akg verify` (and `pytest`) compare this code with the code and logs of the paper runs, offline:
  all prompts and tool schemas byte for byte, the Cypher of 102 logged tool calls, the decomposition of 120 logged model
  replies, the evidence formatting and token counts of 600 logged questions, the request body of every LLM role, and the
  graph counts. With a Neo4j server they also check the graph fingerprint.
  The retrieval baselines reproduce the paper's logged top-20 lists for all 200 questions (BM25, dense and GraphRAG;
  the GraphRAG entity embeddings are bit-identical, so they are not shipped).
* **Accuracy depends on the endpoint.** Model versions and serving change; a rerun will not give identical answers.
  Compare methods only within one `models.yaml`.
* **`non-validated` and `validated` are not clean draws in the paper.** Their 2Wiki rows were seeded from earlier
  tool-bound runs: 165–193 of 200 answers per repeat were copied from runs where the query agent had tools but called
  none, and only the rest were generated tool-free. A fresh run of these two configs can differ.
* **The `*-tool` rungs were chosen on the evaluated questions** (best of several tool catalogues), so their scores are
  optimistic. The alternatives are the ablation configs.
* **`flat` decomposition keeps a quirk of the paper runs:** when the model replies with a JSON list, the `[`/`]` lines
  and the quoted items become sub-questions (14–16% of question-runs). `decomposer: {type: flat, strip_json_list: true}`
  parses such replies as JSON; the paper numbers use the default.
* **GraphRAG** needs `PYTHONHASHSEED=0` (tie-breaking iterates sets of strings; `akg` sets it). Its loader applies
  `relation_map.json` a second time to the already consolidated graph, which relabels 84 `PRODUCTION_COMPANY` edges as
  `RELATED_TO` in the GraphRAG graph only. The USD of `graphrag-local` excludes the ~9.8k LLM calls that wrote the
  community reports (shipped, keyed by member set and model id).
* **Cost** is `tokens × list price` from the run records (judge, embedding and index-time calls excluded). Latency is the
  per-question wall time of one sequential run on a shared endpoint; its median varies about 2× across repeats, which is
  why the paper compares medians and gives means for the tails.
* **Significance** in the paper is a paired bootstrap over questions (per-question accuracy averaged over repeats, 95% CI,
  p < 0.05); `akg report` also prints an exact McNemar test.
* **The judge** (`glm-5p3`) sees question, gold and final answer. In the paper's blind check of 400 verdicts it agreed on
  2Wiki; the gold has no aliases, so a correct answer in an unusual surface form can still be marked wrong.
* **The graph is frozen data.** A rebuild (`--from-raw`) gives the same entity and edge counts and ~99% identical
  lines; the remaining differences come from set iteration order in the consolidation step. `--from-cache` ends up
  ~97% identical (the rebuild step iterates sets too).
* **Neo4j row order.** Queries without `ORDER BY` return rows in store order, and tools use `LIMIT`. The loader keeps the
  paper's load order and uses `apoc.merge.relationship`; use the same Neo4j version (5.26) for identical row order.
* The 200 questions are a fixed subset of a seeded 1,000-question sample of the validation split; the script that picked
  the 200 was not kept. Gold answers ship with the dataset and have no aliases, so the judge is the headline metric.

## Layout

```
configs/    models.yaml  params.yaml  datasets/  graph_build.yaml  methods/  methods/ablations/
data/2wiki/ questions, gold, corpus, graph/, build_cache/, graphrag/        (see its README)
src/agentic_kg/
  agentic/      pipeline, components/ (decomposer, query_generator, validator, answerer, combiner, evidence), tools_registry
  domains/twowiki/   schema, prompts, tools/ (generic, extended, descriptions)
  retrieval/    bm25, dense, graphrag/, common (retriever -> context -> answer)
  graph_build/  records, openie, judge_openie, resolve, consolidate, build_kg, build_corpus, build_gt, load_neo4j, fingerprint
  eval/         judge, judge_prompts, efficiency, stats, report, verify
  llm.py  neo4j_client.py  config.py  registry.py  runner.py  records.py  cli.py
tests/      unit tests, FakeLLM end-to-end runs of every method config, golden files from the paper runs
```

## License and citation

The code is released under the [Apache License 2.0](LICENSE). The 2WikiMultihopQA questions, passages and gold answers are
derived from the dataset of Ho et al. and keep its licence (see the dataset card).

The "Cite this repository" button uses [CITATION.cff](CITATION.cff); please cite the paper (Duruoğlu, Gürsoy and Toraman,
ECIR 2027 Industry Day) and the sources the work builds on: Ho et al., *Constructing a multi-hop QA dataset for
comprehensive evaluation of reasoning steps*, COLING 2020 (2WikiMultihopQA); Edge et al., *From local to global: a Graph RAG
approach to query-focused summarization*, 2024; and Pan et al., *LLM-enabled multi-agent framework for natural language
interaction with graph-based digital twins*, Automation in Construction 183 (2026), whose three-agent architecture
(Graph-DT-GPT) is the starting point.
