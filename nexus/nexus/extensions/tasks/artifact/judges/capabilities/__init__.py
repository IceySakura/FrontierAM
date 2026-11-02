"""Judge capabilities — atomic building blocks composed by the judge strategies.

Grouped by concern (mirrors hywebjudge's ``capabilities/`` layering):
  * :mod:`deploy`  — archive decode, PW-sandbox deploy, screenshots, artifact collect/upload
  * :mod:`agent`   — run Claude Code in a runtime, read reward file, release runtime
  * :mod:`llm`     — vision-LLM scoring, client resolution, retry
  * :mod:`parse`   — extract scores from ``<judge>...</judge>`` XML
  * :mod:`prompts` — load a strategy's default prompt template from ``judges/prompts/``

Strategies import from the specific submodules (explicit), e.g.
``from ..capabilities.deploy import prepare_sandbox_context``.
"""
