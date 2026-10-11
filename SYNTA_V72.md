# Synta v72: web operations and execution cost evidence

The master has real named tools, not simulated descriptions of operations:

- `inspect_web_page`: public HTTPS text, links and form field names. Returned content is untrusted. No hidden field values or credentials are exposed.
- `open_web_page`: inspect the page and launch Firefox in the isolated Debian guest. Existing `sandbox_gui` and `sandbox_run` operate inside that guest; launch success is not task completion.
- `prepare_web_submission`: write an exact bounded form POST draft with URL, fields, purpose and request hash. It does not send anything.
- `/approve-web <id>`: an explicit authenticated operator command submits that exact draft once. HTTPS is certificate-verified and pinned to public DNS addresses. No redirects or retries. An interrupted/failed transport has an unknown external outcome and will not be resent automatically. Drafts expire after one hour. Passwords/payment secrets require operator handoff. This does not add real trading, paid services or automatic account/mail permission.
- `inspect_trade_costs`: read archived execution-price, cost and valuation evidence. Ordinary questions in interactive lab mode can select read-only tools; they cannot change plans or prepare/send forms.

A complaint mentioning a chart and buying no longer receives the canned visualization instructions. Internal tool text stays English; the reply follows the operator's language.

## Chart correction

Historical buy/sell markers previously used the nearest closing equity point. A later intra-bar price decline could therefore look like an immediate purchase charge. v72 emits separate before/after execution points at the opening reference price and keeps subsequent closing valuations separate. Markers use the exact execution-time equity. Receipts include spot equity, liquidation equity and estimated future exit costs.

With 50 bps assumed fee and 10 bps assumed slippage, the spot entry loss is `1 - 1 / (1.001 * 1.005)`, approximately 0.60%. Instant liquidation value is lower by approximately 1.19% including estimated exit friction. Those assumptions are not certified exchange fees or observed bid/ask spreads. A cumulative negative return at another buy is not the cost of that buy alone.

Original report files, simulator returns, drawdown, fills and training eligibility are unchanged. A chart presentation error is not grounds to erase valid losses or label them profitable. Replaying archived data does not independently establish vendor accuracy or explain every market loss.

## Update

Use `tools/update-synta-web-trades.sh <pinned revision>` from v67-v72. It preserves the current accepted/live mission profile, stops the owned mission, installs the pinned package, restarts mission serving and the dashboard/remote gateway if active. Goals, weights, paper ledger, old reports and Windows policy are retained. Reload the dashboard once to load the new JavaScript. New hardware throughput and live external task success are not verified by unit tests.
