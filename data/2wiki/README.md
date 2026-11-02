# 2WikiMultihopQA data

| File | What it is |
|---|---|
| `questions_sample200.jsonl` | the 200 evaluation questions: `id, question, topic, level, operation_type, level_name` |
| `gold.jsonl` | gold for all 1,000 sampled questions: `answer_value`, `answer_aliases` (always empty), `relevant_sets` (one AND-set of the supporting passages, `Passage:<id>`) |
| `corpus.jsonl.gz` | 6,350 passages (the retrieval baselines index these): `doc_id, id, title, document, keywords` |
| `records.jsonl.gz` | the 1,000 sampled questions with their 10 context passages each and the dataset's gold triples (input of the graph build) |
| `sample.json` | `{"split": "validation", "seed": 1881235750, "sample": 1000}` |
| `graph/kg.jsonl.gz` | **the knowledge graph** queried by the agentic methods: 6,350 passages, 42,622 entities, 58,394 edges, 65,828 mentions, 67 relation types |
| `graph/relation_map.json`, `schema.json` | raw relation phrase → canonical relation (9,718 phrases → 67 types); entity types and relation counts |
| `graph/fingerprint.json` | content fingerprint of the graph as loaded into Neo4j (`akg graph fingerprint`) |
| `build_cache/` | OpenIE output per passage, the triple-audit log, and the graph before consolidation (`kg_raw`): replay the build without LLM calls |
| `graphrag/` | the GraphRAG community side-map and the 9,828 LLM community reports |

## Provenance

* **Source:** 2WikiMultihopQA (Ho et al., COLING 2020), HF `xanhho/2WikiMultihopQA` at revision
  `e37a4050605363be62f1d02e6eb888fe5f56530e`, validation split. See the dataset card for its licence.
  1,000 questions were drawn with `random.Random(1881235750).sample`; `akg graph build --step sample` reproduces
  `records.jsonl` exactly from the dataset.
* **The 200 evaluation questions** are a fixed subset of those 1,000 (compositional 81, comparison 56,
  bridge_comparison 43, inference 20). The script that selected them was not kept, so the subset is documented, not
  re-derivable. It is roughly proportional to the 1,000-question type mix, not stratified.
* **Gold answers** ship with the dataset (`answer`); there are no aliases, so scoring against a single string is strict and
  the LLM judge is the headline metric. `relevant_sets` come from the dataset's `supporting_facts`.
* **The graph** is built from the passages only (the gold triples are not used): OpenIE with `gpt-oss-120b`, entity
  resolution (E5 `multilingual-e5-large`, cosine ≥ 0.92, mutual nearest neighbour, same type, behind a veto on
  contradictory facts), an audit of the triples against their passage by `kimi-k2p6` (1,184 triples dropped, 1,495
  fixed), and a canonical relation vocabulary built by `kimi-k2p6` (67 relation types, including the catch-all
  `RELATED_TO`, 26% of edges). The original sampling parameters were not recorded.

## Known limits

* The graph is extracted, so gold answers can be missing from it (e.g. a person's spouse never stated in a passage). That
  caps every agentic method regardless of its components; the retrieval baselines read the passages directly.
* Edge direction is unreliable (a `DIRECTOR` edge can run film→person or person→organisation): the tools match
  undirected.
* One edge between two entities has the type `MENTIONED_IN` (the relation name collides with the mention edges to
  passages), which is why Neo4j shows 67 relationship types: 66 relation types plus `MENTIONED_IN`.
* Entity types actually used: 11, not the 8 listed in `schema.json` (Award, Position, Title also occur).
