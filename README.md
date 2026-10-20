# rh-local-agent

A daily trading agent that runs on your own GPU machine.
Local LLM (Ollama) + Robinhood's official Trading MCP + hard safety limits in code.
Stocks and crypto. You run it twice a day and it asks you before every buy.

Not investment advice. The strategy and every number in `config.example.yaml` are
placeholders. You set the rules and the sizes, and you approve every buy. You are
responsible for every order this places. Try each new setup in paper mode (a dry run) first.

## The one idea that matters

The LLM never holds the order tools, and it never sets a size or a price.

| Part | Job | Can it place a trade? |
|---|---|---|
| `strategy.py` | Plain rules turn quotes into a proposed order | No, it only proposes |
| `guardrails.py` | Plain code checks every limit | No, it only blocks |
| Local LLM (`llm.py`) | Default mode: reads headlines, may **veto** an entry. Decider mode: picks buy / sell / hold | It never sets size or price, and it cannot veto an exit |
| `rh.py` | The only door to Robinhood orders, locked to one account number | Yes, after all of the above |
| `chat.py` | Research chat, local model with **read-only** tools | No |
| `scan.py` | Morning: fixed rules from `config.yaml` pick which stocks and coins get a look | No |
| `analyst.py` | Morning: the local model says buy or skip for each one | No, it only suggests |
| `daily.py` | Morning and evening sessions: shows you the ideas and asks before every buy | Only after you type `yes` |

If the LLM is down or replies with junk, the answer is veto (fail closed). A poisoned
news headline can, at worst, cancel a trade. Exits (stop loss, take profit, flatten
before close) skip the LLM so nothing can stop the agent from protecting money.

Bonus: the LLM is only called when a rule fires AND the guardrails pass. Most cycles
use zero tokens. `cycle_done` in the journal shows LLM calls and tokens per run.

## Decision engine: Laya (`engine: laya`, the default in config.yaml)

With `engine: laya`, Laya (an open-source decision model, Apache 2.0) makes the three calls the
local LLM used to make: the morning buy/skip, decider buy/sell/hold, and the veto review.
Laya writes no text. It picks one of the options the code offers and gives a probability for each,
so broken JSON, made-up symbols or an action outside the list cannot happen. It only offers
buy/hold when nothing is held and sell/hold when something is. It **can still be wrong**,
and it was not trained on trading. Size, price, limits and exits stay in code exactly as before.
If Laya is down or answers in an unexpected shape: skip / hold / veto. `engine: llm` switches back.
The research `chat` always uses the local LLM.

```powershell
python -m pip install "laya[serve]"                               # once
powershell -ExecutionPolicy Bypass -File .\scripts\run_laya.ps1   # every time, in its own window. Leave it open
.\venv\Scripts\python.exe -m agent doctor     # step 3 checks Laya
```

## Setup (Windows PowerShell)

```powershell
# 1. Local model. A 14B model at Q4 is about 9 GB and fits a 12 GB card.
#    Model tags change. Check ollama.com/library for what is current.
ollama pull qwen3:14b

# 2. Project (run from the folder that contains agent\, scripts\ and requirements.txt)
cd rh-local-agent
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1     # venv, packages, config.yaml, safety tests

# 3. Check every link in the chain. Run this first whenever something fails.
.\venv\Scripts\python.exe -m agent doctor
```

Common first-run problems:

| You see | Cause | Fix |
|---|---|---|
| `.\venv\Scripts\python.exe ... is not recognized` | The venv was never created | Run `setup.ps1` as shown above |
| `Python was not found; run without arguments to install from the Microsoft Store` | No real Python installed | `winget install Python.Python.3.12`, then open a NEW PowerShell window |
| `No module named agent` | Wrong folder. "Extract All" often makes `rh-local-agent\rh-local-agent` | `cd` one level deeper |
| `CONFIG ERROR: ... config.yaml not found` | No config yet | `Copy-Item config.example.yaml config.yaml` |
| `universe still has placeholder names` or `Robinhood does not know this symbol` | The `universe` list has fake or misspelled tickers | Put real symbols in `config.yaml`, then run `python -m agent tickers` |
| `running scripts is disabled on this system` | PowerShell policy | Use the `-ExecutionPolicy Bypass` form above |

## Daily sessions: start here

You run two commands a day. Nothing runs on a schedule and nothing is bought without your `yes`.

```powershell
.\venv\Scripts\python.exe -m agent morning    # stocks: after 8:45 AM Central (15 min after the open). Crypto: any time
.\venv\Scripts\python.exe -m agent evening    # stocks: before 3:00 PM Central (the close). Crypto: any time
```

**Morning**
1. Checks what the agent already holds, and that each holding still has its stop order at Robinhood.
2. Scans, in plain code, with the rules under `scan:` in `config.yaml`:
   stocks with Robinhood's own screener (price $5 to $50, heavily traded, no funds), and
   the coins with the tightest bid/ask spread (stablecoins skipped). Add your own picks with
   `universe:` (stocks) and `scan.crypto.include:` (coins).
3. Your local model reads each one (price moves over 1, 5 and 20 days, today's 5-minute prices and
   headlines for stocks, hourly prices for coins) and answers buy or skip.
4. You see up to 5 ideas and pick them by number. Press Enter to buy nothing.
5. For each pick: a fresh price, every guardrail, Robinhood's own review, then you type `yes`.
6. After a buy fills, a stop order at Robinhood protects it (`daily.stock_stop_loss_pct`,
   `daily.crypto_stop_loss_pct`). The stop stays until it is cancelled, so it also covers the time your PC is off.

**Evening**
- Up or flat since you bought: sell. You confirm once for the whole list.
- Down: you choose. Sell now at a loss, or keep it overnight with its stop order (`daily.when_down`).
- Each sell first cancels the holding's stop order, then checks the fill. A sell that does not fill gets its stop order back.
- The agent only sells what it bought itself (`state/agent_book.json`). Anything else in the account is left alone.

**Crypto hold style** (`daily.crypto_style: hold`)
- A coin is not sold in the evening. Right after it is bought, a target sell order waits at Robinhood,
  `crypto_take_profit_pct` over what you paid, and sells by itself once the price gets there (crypto limit orders last 90 days).
- Robinhood keeps only one sell order per coin, so no stop order can sit next to the target. Every session checks
  the stop level (`crypto_stop_loss_pct` under cost) instead. Under it, the evening offers to sell, cancelling the target first.
  That check only happens when you run the agent. Between runs, nothing guards the downside.
- Why hold: Robinhood's crypto buy/sell gap was about 1.8 to 1.9% in Sept 2026. A same-day trade has to beat that gap by
  the evening; a target 5% up leaves room for it.
- `crypto_style: day` treats coins like stocks: a stop order at Robinhood, sold in the evening if up or flat.

**Limits.** Every buy, stock or crypto, goes through the same guardrails: at most `max_order_usd` per buy,
`max_position_usd` per holding, `max_daily_buy_usd` per day and `max_open_positions` at once.
`mode: paper` turns both sessions into a dry run (nothing sent, nothing cancelled). `mode: approve` sends real
orders, each one only after your `yes`. The two auto-mode unlocks are not needed, because you confirm every order.

**First time**
```powershell
.\venv\Scripts\python.exe -m agent discover   # log in. Copy the agentic account number into config.yaml, run it again
.\venv\Scripts\python.exe -m agent scan       # what the scan finds right now (no model, no orders)
.\venv\Scripts\python.exe -m agent morning    # with mode: paper, a dry run of the whole morning
```
Then set `mode: approve` for real orders.

**Built to survive a bad moment**
- Each order's `ref_id` is written to disk before it is sent. A retry after a dropped connection re-sends the
  same `ref_id`, and Robinhood never places one `ref_id` twice.
- Nothing stays connected while you think. Each step opens a fresh connection.
- Every run first reconciles with Robinhood. It records a stop that already sold, finds an order that was cut
  off mid-send, adopts a buy that filled but was never recorded, and offers a stop for any holding without one.
- Only one session can run at a time.

**Crypto data.** Robinhood's tools have no price history or news for coins. The analysis takes hourly prices from
Coinbase's public API (`scan.crypto.history: none` turns that off). Orders always go to Robinhood only.
Crypto needs your agentic account's crypto side: `discover` shows its number (`rhs_account_number`).

## Unattended style (once / loop / auto-*): do these in order. Do not skip.

This is the older style: it trades only the stocks in your `universe` list, and in auto mode it asks nothing.

**Step 1. Offline dry run (no Robinhood at all).**
In `config.yaml` set `transport: mock`, `account_number: "MOCK-AGENTIC-1"`, `universe: [AAA]`.

```powershell
$env:MOCK_DROP_PCT = "3"
.\venv\Scripts\python.exe -m agent once
.\venv\Scripts\python.exe -m agent report
```

**Step 2. Connect to the real Robinhood MCP.**
Set `transport: http` and `account_number: ""`.

```powershell
.\venv\Scripts\python.exe -m agent discover
```

Your browser opens Robinhood's own login page. The first time, Robinhood walks you
through opening the Agentic account (desktop browser only). Copy the account number
that shows `agentic_allowed=True` into `config.yaml`, then run `discover` again.
Check the last line: `buying_power` and `equity` must show numbers, not `None`.

**Step 3. Paper mode on live quotes. At least two weeks.**
Set `mode: paper`, fill in `universe`, then:

```powershell
.\scripts\run_loop.ps1
```

Real quotes and real headlines, pretend money. Read `journal\*.jsonl` every day.
Look hardest at `blocked_by_guardrails` and `llm_review` lines.

**Step 4. Approve mode.** `mode: approve`. Small money in the Agentic account.
The agent shows Robinhood's review result and waits for you to type `yes`.

**Step 5. Auto mode, only if Step 4 was boring.** Needs two unlocks on purpose:

```powershell
# config.yaml ->  mode: auto   and   i_understand_auto_places_real_orders: true
[Environment]::SetEnvironmentVariable("RH_AGENT_ALLOW_AUTO", "1", "User")
.\scripts\register_task.ps1
```

## Kill switches (fastest first)

1. `New-Item KILL` in the project folder. Next cycle places nothing. `Remove-Item KILL` to resume.
2. `Stop-ScheduledTask -TaskName "RH Local Agent"`
3. Delete the `.secrets` folder (the agent loses its login).
4. Disconnect the agent inside Robinhood. This is the real off switch.

## Commands

```powershell
.\venv\Scripts\python.exe -m agent morning    # daily session: scan, the model suggests, you pick and confirm each buy
.\venv\Scripts\python.exe -m agent evening    # daily session: sell what is up or flat, you decide on what is down
.\venv\Scripts\python.exe -m agent scan       # what the morning scan finds right now (no model, no orders)
.\venv\Scripts\python.exe -m agent discover   # login, list tools, show accounts, save response shapes
.\venv\Scripts\python.exe -m agent once       # unattended: one cycle (universe list only)
.\venv\Scripts\python.exe -m agent loop       # unattended: cycles until the close
.\venv\Scripts\python.exe -m agent auto-morning   # unattended two-session style: decide, buy, place broker stops
.\venv\Scripts\python.exe -m agent auto-evening   # unattended two-session style: cancel stops, sell everything, verify flat
.\venv\Scripts\python.exe -m agent chat       # research chat, read-only tools
.\venv\Scripts\python.exe -m agent report     # today's journal summary
.\venv\Scripts\python.exe -m agent tickers    # do the tickers in your universe fit the rules? (price, spread, state)
.\venv\Scripts\python.exe -m agent doctor     # checks Python, config and the local LLM. No Robinhood involved
.\venv\Scripts\python.exe -m agent bench --models gemma4:12b qwen3:14b qwen3.5:9b   # compare models on this agent's job
.\venv\Scripts\python.exe -m pytest -q        # run after ANY change to guardrails or strategy
```

## Decider mode: the agent makes the call

Set `strategy.name: llm_decider` and the local LLM decides buy, sell or hold for each ticker in
your universe, from live quotes, recent 5-minute closes and headlines. With `mode: auto` on top,
the agent decides and trades with no human in the loop.

What the LLM controls: which symbol, which action, how confident it is.
What stays in code, always:

| Thing | Who owns it |
|---|---|
| Which tickers exist (`universe`) | You |
| Order size, price, order type, account | Code, from `config.yaml`. Size fields in the LLM reply are ignored |
| Stop loss, take profit, flatten before close | Code. Runs before the LLM is asked. The LLM cannot stop it |
| Every limit in `guardrails` | Code. Applied to every LLM decision, then again on a fresh quote |
| Broken JSON, unknown ticker, low confidence, LLM offline | Becomes hold |

Be clear-eyed about three things:

1. **A 12B local model has no proven trading edge.** Nobody has shown that small LLMs beat the
   market intraday, and most human day traders lose money. Treat this as an experiment with money
   you can lose, not as income.
2. **Headlines can now cause a buy.** In veto mode a poisoned headline could only cancel a trade.
   In decider mode it can trigger one, inside your universe and caps. Your guardrails are the
   blast radius. Set `decider.use_news: false` to decide on prices only.
3. **Paper mode is the only honest benchmark.** Run decider in `mode: paper` for a few weeks and
   read `python -m agent report` daily. It shows every decision, the reason, and paper P&L.
   Paper fills assume you get the limit price, so real results will be a bit worse.

Full autonomy checklist:

```powershell
# config.yaml
#   mode: auto
#   i_understand_auto_places_real_orders: true
#   strategy.name: llm_decider
[Environment]::SetEnvironmentVariable("RH_AGENT_ALLOW_AUTO", "1", "User")   # then open a NEW PowerShell window
.\scripts\register_task.ps1
```

## Two ways to run the day unattended

This is for the unattended style only. The daily sessions above never run on a schedule.

| | Continuous | Two sessions |
|---|---|---|
| What happens | A check every 5 minutes, all day | Morning: decide and buy. Evening: sell everything. Nothing in between |
| Register with | `.\scripts\register_task.ps1` | `.\scripts\register_task.ps1 -Style two_sessions` |
| LLM calls per day | Up to ~26 in decider mode | 1 |
| Take profit during the day | Yes (code checks it) | No. You hold until the evening sell |
| Stop loss during the day | Code check every 5 min + broker stop | Broker stop only |
| PC must stay on all day | Yes | Only at the two task times |

**The broker stop is what makes two sessions safe.** With `protect.broker_stop: true`, right after a
buy fills the agent places a real stop order at Robinhood, `stop_loss_pct` under your cost. That
order lives on Robinhood's servers. If your PC sleeps, crashes, or loses internet at 11 AM, the stop
still works. In the evening the agent cancels the stop first (shares under an open order cannot be
sold), sells, waits, and checks that nothing is left.

Read the evening log. Two loud events mean you must open the Robinhood app yourself:
- `UNPROTECTED_POSITION`: a buy filled but the stop could not be placed.
- `STILL_HOLDING`: the evening sell did not fully fill. That position carries overnight with no stop.

Limits of a stop order, honestly: a stop becomes a market order when touched. In a fast drop or a
gap it can fill below your stop price. It limits the damage, it does not guarantee the price.

Times in `register_task.ps1` are your PC's local time and the defaults assume US Central
(8:50 AM and 2:40 PM). If the evening task does not run (PC off, not logged in), the broker stop
expires at the close and the position carries overnight. `StartWhenAvailable` runs a missed task
as soon as the PC wakes, but after 3 PM Central the market is closed and the agent can only log it.

Half days: the market shuts at 1 PM Eastern on 2026-11-27 and 2026-12-24, before the evening task.
Both dates are pre-filled in `schedule.no_trade_dates`, so the agent makes no new buys on them.

## Picking the local model

The model does one small job here: read a few headlines, answer approve or veto in clean JSON.
A smarter model does not make better trades, because the rules decide the trades. What matters is
zero JSON errors, no missed vetoes, and that it fits in VRAM. So measure, do not guess:

```powershell
ollama pull gemma4:12b
ollama pull qwen3.5:9b
.\venv\Scripts\python.exe -m agent bench --models gemma4:12b qwen3:14b qwen3.5:9b
```

Then put the winner in `config.yaml` under `llm.model`. Add your own cases to `agent/bench.py`.
Check ollama.com/library for current tags before pulling.

## Using llama.cpp instead of Ollama

The agent speaks the OpenAI chat format, and `llama-server` serves exactly that, so it is a
drop-in swap. `llama-server` also gives you a chat page in the browser for poking at the model by hand.

```powershell
winget install llama.cpp
# open a NEW PowerShell window, then (first run downloads the model, about 7 to 8 GB):
llama-server -hf unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL --jinja -ngl 99 -c 8192 --port 8080
```

- `--jinja` applies Gemma's chat template and is needed for tool calling in `agent chat`.
- `-ngl 99` puts all layers on the GPU. `-c 8192` keeps the context small so it fits in 12 GB.
- Wait for `listening on http://127.0.0.1:8080`. The browser UI is at that address. The agent uses `/v1` on the same port.
- Check that `winget` gave you a GPU build: the startup log should mention CUDA (or Vulkan) and your GPU name.
  If it shows CPU only, download the CUDA zip for Windows from the llama.cpp GitHub releases page instead.

Then in `config.yaml`:

```yaml
llm:
  base_url: http://localhost:8080/v1
  model: gemma4-12b            # just a label. llama-server uses the model it loaded
  extra_body:
    chat_template_kwargs: {enable_thinking: false}
```

Why `enable_thinking: false`: on llama.cpp, Gemma 4 can spend its whole reply "thinking" and leave the
real answer empty. For a one-line approve/veto JSON, thinking only adds delay. The agent also rescues a
reply that lands in the reasoning field, but switching thinking off is faster and cleaner.

Do not run Ollama and llama-server with the same model at the same time on a 12 GB card. Two copies
will not fit. Stop one first (`ollama stop gemma4:12b`, or quit Ollama from the tray).

Two-session scheduling note: Ollama starts with Windows. `llama-server` does not. If you schedule
the agent with llama.cpp, the server must already be running when the morning task fires, or the
agent fails closed and makes no trades.

## What was verified, and what was not

Verified when this was built (Sept 2026):
- MCP Python SDK 2.2 API. Note that 2.x renamed things; most online tutorials show 1.x and will not run.
- Order argument names, taken from Robinhood's live tool schemas (`review_equity_order`, `place_equity_order`, including `ref_id`).
- The quote response shape, from one real `get_equity_quotes` call.
- The whole pipeline over the real MCP protocol against the mock server. 51 tests pass.

Not verified. `discover` exists to check these on your machine:
- Live OAuth from this custom client. Robinhood's MCP uses dynamic client registration, so it should work with no app approval. If registration is rejected, delete the `token_endpoint_auth_method="none"` line in `agent/oauth.py` and try again.
- The response shapes of `get_accounts`, `get_portfolio`, `get_equity_positions`, `get_equity_news`, `get_equity_orders`, `get_equity_historicals`. If open orders cannot be read, the evening cancel step finds nothing, the sell is rejected by Robinhood, and you get `STILL_HOLDING`. Test the full morning and evening flow in approve mode with one share before trusting it.
  The parsers search for several likely key names. If one is wrong the agent fails safe: unknown buying power or equity blocks all buys, and unknown average cost disables exits for that position with a journal warning.
- The daily sessions (added Sept 2026) were built from Robinhood's live tool schemas (argument names for
  `preview_scan`, the crypto tools and `get_equity_orders`) and tested against the mock server. The RESPONSE shapes
  of the screener and the crypto tools are not verified yet. `discover` saves them to `discovery\` and prints a parser
  check: the screener's picks, the number of crypto pairs, and a BTC-USD bid, ask and time. If the screener shows
  nothing, the morning carries on with your `universe` list. If a crypto quote has no readable time, crypto buys stay
  blocked as stale. Do your first real morning with one small order and watch it in the Robinhood app.

## Things that will bite a daily trader

- **Settled cash.** A cash account must wait for sales to settle before that money can buy again. Robinhood's review step reports problems like this as alerts, and `block_on_any_review_alert: true` stops auto mode when one appears.
- **Day-trading rules.** FINRA removed the old $25,000 pattern-day-trader rule effective June 4, 2026, but brokers may phase the change in until October 2027. Treat Robinhood's review alerts as the truth for your account.
- **Holidays and halts.** There is no holiday calendar. The stale-quote check covers it: no fresh trade print, no order.
- **Local model latency.** A 14B model can take a minute. The pipeline pulls a fresh quote and re-runs every guardrail after the LLM answers.
- **Taxes.** Frequent trading means short-term gains and wash sales. The journal is JSONL so you can export it.
- **Secrets on camera.** `.secrets\`, `config.yaml`, `discovery\` and `journal\` hold account numbers or tokens. Blur them if you record this.

## Writing your own strategy

Replace `propose()` in `agent/strategy.py`. Keep the contract: read a `Snapshot`, return
`Candidate` objects, and do not call the LLM or the network. Keep strategies
deterministic so the same journal always explains the same trade.
