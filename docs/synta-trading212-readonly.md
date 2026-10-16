# Trading 212 Invest: prepared read-only connector

`rlm.v100.trading212_readonly.Trading212ReadOnly` is an opt-in stdlib transport.
Importing or constructing it makes no request. It is not connected to the dashboard,
does not place orders, and is not proof of account integration or revenue.

The [official API documentation](https://docs.trading212.com/api) specifies Invest
and Stocks ISA support, separate demo/live hosts, and Basic authentication using
an API key and API secret. CFD is outside this connector. API responses use the
primary account currency; 100 PLN must not be inferred from an unknown currency.

Credentials are read only from `SYNTA_TRADING212_API_KEY` and
`SYNTA_TRADING212_API_SECRET` in the calling process environment. Supply them via
private local credential management; never paste them into chat, command arguments,
source files or logs. Prefer keys restricted to account/portfolio read permissions.
Do not print or persist the returned private financial data without a separate design.

The default environment is demo. Explicit `environment="live"` selects real-account
**reads only**. Supported calls are `account_summary()` and `positions()` using GET
on exactly `/api/v0/equity/account/summary` and `/api/v0/equity/positions`.
There are no order methods, retries, pagination, redirects, proxy usage or background
requests. TLS uses the standard verified HTTPS transport. Responses are limited to
1 MiB, socket operations to at most 30 seconds, with a deadline during body reads.
Errors omit response bodies and original network exception messages. Account identifier
fields (including generic `id`) are removed recursively. Other fields retain the API's
schema and must be validated before dashboard accounting.

No credential-bearing requests were performed during development. Tests use a fake
HTTPS transport. Before any future integration: verify a demo response privately,
validate currency and financial fields, show account freshness separately from paper
results, and keep actual order execution disabled. This connector cannot enforce
trading loss limits because it never trades.

## Development verification — 2026-10-16

25 isolated tests passed: default demo and explicit live selection, GET-only paths,
recursive identifier removal, proxy independence, four redirect statuses refused
without a second request, response-size enforcement, invalid JSON/UTF-8/schema,
endpoint allowlist before credentials/network, absent credentials, timeout bounds,
body deadline, HTTP 429 without retries, sanitized transport failures, CLI environment
credentials, hidden prompts with restoration, and CLI sanitized error/exit status.
Targeted Ruff lint and format checks passed; `git diff --check` reported no errors.
Only mocked transport was exercised during development. Deployment and account
connection status are tracked separately in the project progress file; passing these
tests does not establish an authenticated account connection.

## Private one-shot CLI

From a private terminal in the installed project environment, run:

```sh
python -m rlm.v100.trading212_readonly --environment demo
```

If the environment credentials are absent, the CLI asks for each using a hidden
terminal prompt. No secret belongs in command arguments or shell history. It refuses
an echoing prompt fallback, restores the previous process environment afterwards,
and reports sanitized failures with exit code 2. Success prints the private account
summary and positions to that terminal only, with the environment and
`read-only; orders disabled` scope. Do not share the terminal output publicly or run
it under logging/recording. It performs two GETs once, with no polling or files.

Demo API credentials and live API credentials are separate; use a demo key/secret
for the command above. Only explicitly choosing `--environment live` reads the
real Invest account; this still cannot submit orders.
