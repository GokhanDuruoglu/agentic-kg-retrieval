# Adding and removing components

A method is a YAML file in `configs/methods/`. Its sections name registered components and their parameters;
`extends: <other method>` starts from another method (mappings merge recursively, `null` deletes a key, and a section
whose `type` changes is replaced as a whole). `akg show <method>` prints the result.

```yaml
# configs/methods/validated-tool.yaml
extends: validated
tools: {preset: generic5, descriptions: concise}
```

## Remove a component

Set it back to the neutral type, or delete the key:

| Remove | Config |
|---|---|
| tools | `tools: null` |
| retry loop | `validator: {type: none}` |
| splitting + combiner | `answerer: {type: single_cut, max_context_tokens: 1000000}` |
| the combiner LLM call only | `answerer: {type: chunked, combiner: {type: concat}}` |
| chained sub-questions | `decomposer: {type: flat}` |
| decomposition altogether | `decomposer: {type: none}` (the question is the only sub-question; no LLM call) |

## Add a component

A component is a class registered under `(kind, name)`; it is built as `cls(rt, **params)`, where `rt` is the
`Runtime` (LLMs, Neo4j client, database, schema texts) and `params` are the other keys of its config section.

```python
# my_plugin.py
from agentic_kg.registry import register
from agentic_kg.agentic.components.evidence import build_attempts_feedback

@register("validator", "retry-once-then-give-up")
class RetryOnce:
    max_attempts = 2
    def __init__(self, rt, only_on_error: bool = False):
        self.only_on_error = only_on_error
    def run(self, generate, execute):          # generate(feedback) -> query dict, execute(query) -> neo4j result
        attempts = []
        qg = generate(None); neo = execute(qg)
        attempts.append({"attempt": 1, "qg": qg, "neo4j": neo})
        failed = neo["error"] is not None if self.only_on_error else (neo["error"] or neo["row_count"] == 0)
        if failed:
            qg = generate(build_attempts_feedback(attempts)); attempts.append({"attempt": 2, "qg": qg, "neo4j": execute(qg)})
        return attempts
```

```yaml
# configs/methods/my-method.yaml
extends: validated
validator: {type: retry-once-then-give-up, only_on_error: true}
```

```bash
uv run akg --plugin my_plugin run my-method --limit 5
```

The interfaces (see also `agentic/components/__init__.py`):

| Kind | Methods / attributes |
|---|---|
| `decomposer` | `decompose(question) -> Decomposition(sub_questions, groups, decomposed, raw, llm, labels, chained)`; `chained`, `answer_system_prompt` |
| `query_generator` | `generate(sub_question, attempts_feedback, dependency_context) -> {"cypher", "params", "tool", "tool_args", "llm"}` |
| `validator` | `run(generate, execute) -> [{"attempt", "qg", "neo4j"}, ...]`; the last attempt is kept |
| `answerer` | `answer(question, steps) -> (record, partials, combiner_record)` |
| `combiner` | `combine(question, partial_answers) -> {"answer", "llm"}` |
| `retriever` | `search(query, top_k) -> [{"rank", "score", "doc_id", "node_type", "title", "document"}]` |
| `context` | `build(retriever, query, hits) -> (text, n_used, meta)` |

## Add a tool

A tool is a function that returns `(cypher, params)`; the `database` argument is injected by the caller and hidden from the
model. Register it, describe it, and select it by name:

```python
from agentic_kg.agentic.tools_registry import register

@register
def films_by_director(name: str, database: str = "") -> tuple[str, dict]:
    return ("MATCH (d:Entity)-[:DIRECTOR]-(f:Entity) WHERE toLower(d.name) CONTAINS toLower($name) "
            "RETURN f.name AS film LIMIT 50"), {"name": name}
```

```yaml
tools: {preset: generic5, include: [films_by_director],
        extra_descriptions: {films_by_director: "Films directed by ONE named person."}}
```

`tools: {names: [...]}` selects an explicit list, `exclude: [...]` removes tools from a preset, and
`descriptions: concise | guided` picks the built-in description style.

## Add a retrieval method

Register a `retriever` (and, if the answer context differs, a `context`) and write a config with `kind: retrieval`:

```yaml
kind: retrieval
top_k: 20
retriever: {type: my-hybrid, alpha: 0.3}
context: {type: passages, max_tokens: 128000}
```

Retrievers are built as `cls(corpus_path, **params)`; the answer step (prompt, one LLM call, cost accounting) is shared.

## Another dataset or domain

The pipeline itself knows nothing about 2Wiki; the dataset-specific parts are `domains/twowiki/` (schema text, prompts,
tool builders) and `configs/datasets/2wiki.yaml`. A new domain needs its own schema text and prompts and a graph in Neo4j;
point `Runtime.schema` / `decision_schema` at them (see `runner.build_runtime`).
