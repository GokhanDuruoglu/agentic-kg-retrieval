"""Pipeline components. Importing this package registers every built-in implementation.

Each component class is constructed as ``cls(rt, **params)`` where ``rt`` is the ``Runtime`` and ``params``
are the keys of its config section (without ``type``).

    kind              built-in types             interface
    decomposer        flat, chain, none          decompose(question) -> Decomposition; .chained, .answer_system_prompt
    query_generator   cypher                     generate(sub_question, feedback, dependency_context) -> dict
    validator         none, retry                run(generate, execute) -> list[attempt]
    answerer          single_cut, chunked        answer(question, steps) -> (record, partials, combiner_record)
    combiner          llm, concat                combine(question, partial_answers) -> dict
"""

from . import answerer, combiner, decomposer, query_generator, validator  # noqa: F401
