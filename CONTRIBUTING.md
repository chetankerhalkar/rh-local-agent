# Contributing

Thanks for helping. This agent can place real orders, so the rules below are strict on purpose.

## The rules every change keeps

These are the design, not style preferences. A PR that breaks one will not be merged.

1. **The model never holds the order tools.** Only `rh.py` talks to Robinhood orders.
2. **The model never sets a size, a price, an order type, an account or a ticker.** Code and `config.yaml` do.
   Laya and the local LLM may only pick from options the code offers.
3. **Fail closed.** A model that is down, slow or answers in an unexpected shape means skip / hold / veto. Never buy.
4. **Exits skip the model.** Stop loss, take profit and the evening sell are plain code. Nothing can veto them,
   and no counter or limit may block an exit.
5. **Strategies are deterministic.** `propose()` in `strategy.py` reads a `Snapshot`, returns `Candidate` objects,
   and never calls the LLM or the network. Same journal in, same trade out.
6. **Every buy goes through `guardrails.check()`**, again on a fresh quote after the model answers.
7. **Log it.** New decisions, blocks and orders get a `journal` event, including the trades that did not happen.

## Before you open a PR

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1   # first time only
.\venv\Scripts\python.exe -m pytest -q
```

All tests must pass. They run against the mock Robinhood MCP server in `tests/`, never a real account.

- Touched `guardrails.py`, `strategy.py`, `daily.py`, `pipeline.py` or `rh.py`? Add a test that shows the new
  behaviour, and one that shows it still says no when it should.
- New config key? Add it to `config.example.yaml` with a comment, and give it a safe default in code.
- New Robinhood tool call? Use the argument names from Robinhood's live tool schema (`agent discover` saves them
  to `discovery\tools.json`), and make the parser fail safe when a key is missing.

## Never commit

`config.yaml`, `.secrets\`, `journal\`, `state\`, `discovery\`. They hold account numbers and login tokens.
`.gitignore` covers them. Run `git status` before every commit anyway.

## Reporting a bug

Open an issue with:

1. What you ran (`morning`, `evening`, `scan`, ...) and your `mode` (`paper` / `approve` / `auto`).
2. The output of `.\venv\Scripts\python.exe -m agent doctor`.
3. The journal lines around the problem (`journal\YYYY-MM-DD.jsonl`).

**Remove account numbers, order IDs and anything from `.secrets\` before you paste.**
If the bug placed or could have placed a wrong real order, say so in the title.

## Good first areas

- Docker Compose for the agent, `laya-serve` and Ollama
- Mac / Linux versions of the scripts in `scripts\`
- A broker adapter interface, so `rh.py` becomes one implementation
- More fact-sheet features in `analyst.py` (RSI, VWAP, volume)
- A backtester that replays the journal against price history
- A holiday calendar and a volatility-spike guardrail

## Not investment advice

Contributions are code, not trading advice. Do not add claims about returns to the README or the code.
By contributing you agree your work is released under the MIT License in `LICENSE`.
