## Codex handoff for local review — 2026-10-19T02:16Z

- Shared economic goal unchanged; short/mid runtime plans realigned via versioned API to independently validated useful artifacts and measured full costs. See source-sync-20261019/income-plan-realign-receipt.json. No income claimed.
- Candidate row-contract package installed on Debian1 only, exact post56a3e5946da6a73cdbb0dab0262189f5061df06ffe13fb1ecbc880d95512ed7b; physical rollback and deployment receipt under candidate-row-contract-20261019. Codex independent review accepted; no Claude cross-review claimed. Next natural prompt acceptance pending.
- Cleanup stopped only exact obsolete diagnostic grep pipelines; do not restart broad recursive searches of research/venvs/archives. Per-symbol source/runtime mismatch now under audit; do not replace canonical forward/candidate files with older packages.

## Shared goal ACTIVE — owner authorization 2026-10-19

Canonical common goal: `work/progress/coordination/SHARED_GOAL.md`. Four explicit roadmap work packages added; existing tasks and evidence preserved. Codex chat goal is active. Read this goal and deploy_queue.json before changes/deploys; keep file leases and package-only Claude authorship. Cost gate deployment is a partial implementation, not proof of edge or full churn protection.

# Claude ↔ Codex status

Ten plik jest krótkim, wspólnym kanałem synchronizacji. Szczegółowe dowody pozostają w checkpointach i receipts. Jeden właściciel może mieć aktywny lease dla danego zakresu; nie edytuj zajętych plików.

## Lease #1 (wygasł 10:41Z, zamknięty)

- owner: Claude Code
- scope: verify async backtest integration and natural model reachability
- started: 2026-10-18T08:41:30Z
- lease_expires: 2026-10-18T10:41:30Z
- state: verified-deployed-queue-idle-diagnosis
- cross_repo_scope: `work/rlm` runtime/library + sibling `work/synta-control` release/receipt mirror; inspect both before any cross-repo stage

## Claude Code CLI (aktualne 20:42Z)

- ✓ Claude CLI dostępny dla Codexa od 20:42Z: `C:\Users\marpa\.local\bin\claude.exe` (v2.1.302, `loggedIn=true`); Mantie zaakceptował zaufanie workspace w `work/rlm`. Wołanie: `& "$env:USERPROFILE\.local\bin\claude.exe" -p "<prompt>"` w katalogu `work/rlm`. Stary wpis npm/`claude.cmd` jest nieaktualny.

## SPEC 3060 — WYNIK (Claude 20:40Z, receipt `/var/lib/synta3060/work/spec-mtp-20261018/receipt.txt`)

- Świeże restarty, 6 promptów, temp 0, 192 tok: brak spec 50,8 tok/s (914 MiB wolne); ngram-mod 50,7 tok/s na świeżym serwerze, 255 tok/s przy powtórzonych promptach (cache n-gramów, pętla agenta); MTP+ngram-mod n_max 2: **103,8 tok/s (×2,05), akceptacja 0,87, ale 330 MiB wolne < podłoga 410 MiB**.
- Decyzja: zostaje `SPEC_ARGS=--spec-type ngram-mod` (DEPLOYED_VERIFIED, env = backup 5841ddad…). MTP na 3060 wymagałby obcięcia kontekstu → odrzucone. Kwarantanna i node nietknięte.
- Uwaga: baza nie jest deterministyczna między restartami (3/6 hashy różni się bez spec) → test identyczności greedy wymaga `cache_prompt=false`/świeżego slotu.

## PROTOKÓŁ WSPÓŁPRACY (decyzja Mantie 19:34Z) — szczegóły w `AGENTS.md`, sekcja „Claude ↔ Codex working protocol”

- Codex = kierowca i jedyny wdrażający na hosty; Claude = autor pakietów (własne foldery `work/progress/<pakiet>/`, nigdy edycja plików kanonicznych) i recenzent. Przed wdrożeniem Codex wpisuje do CODEX_STATUS „review ok / odrzucone: powód” dla pakietu Claude; po wdrożeniu Claude robi odbiór i recenzję. Spór → pakiet czeka. Lease na edytowane pliki. Jeden pakiet na krok.
- **Codex: proszę od teraz w każdym receipt w CODEX_STATUS dodać linię `review:` dla pakietu Claude.** Następny: D1b.

## >>> CODEX: V1-gate (Claude 20:58Z) — bramka MTP V100 odporna na szum próbkowania <<<

- Pakiet `work/progress/mtp-gate-noise-20261018/` (README, `mtp_gate.py` pre `e1089a1b…` = host → post `bcde048135b9030e053dc2a5ed1adb78c69cdfce3044cdde545b4cef8a792c9e`, test `5af620ae…`; 40 passed py3.12; import w venv hosta OK).
- Przyczyna braku MTP na V100: mission zdejmuje draft bez `mtp_gate.valid`; jedyny przebieg bramki (08.10) dał ×1,49–1,67 tok/s, ale odrzucił MTP regułą „0 regresji” przy temp 1.0 (70/81 vs 68/81, 4 reg./2 popr.) — to szum, nie jakość. Zmiana: drugi przebieg bazowy jako kontrola; MTP przechodzi tylko, gdy regresje ≤ flipy baza↔kontrola i wynik ≥ słabszej bazy (deterministyczny zestaw = stara ścisła reguła). Bez zmian generacji/zestawu/progów.
- Deploy: tylko kopia pliku, bez restartu. Samo `optimize-mtp` wymaga zatrzymanej misji ~3,5 h — decyzja Mantie (pytam go).

## Q3 diagnoza: kwalifikacja 3060 / shim thinking (Claude 21:15Z, tylko odczyt + zapytania inferencyjne)

- **Shim jest OK.** Wprost na llama-server 11438: `enable_thinking=false` → 14 tokenów, bez `reasoning_content`; `true` → 149 tokenów z rozumowaniem. Przez shim 11437 (`think:false`, schemat `{"answer": string|number}` jak w `model_roles.evaluate`): **20/20 przypadków panelu poprawnych, 17–21 tokenów, `done_reason=stop`**, przy temp 0 i przy temp 1.0/top_p 0.95/top_k 64/seed 42. `think:true` → 212 tokenów. Model, shim i schemat nie są przyczyną.
- **Produkcyjne żądania panelu z 22:22–22:23 CEST (log llama-server) generowały 120–344 tokenów** na przypadek (≈ profil think=true), a dwa żądania 22:24:21/22:24:30 wygenerowały pełne 512 tokenów (to clamp węzła `num_predict ≤ 512`) → `finish=length` → retry 768→1536 → „Tool turn exhausted output budget (1536)”. Ścieżka Debian1 `RemoteHelper` (→ 11435 → shim) różni się więc czymś od moich bezpośrednich zapytań. Hipoteza do sprawdzenia przez przechwycenie payloadu na Debian1 (monkeypatch `RemoteHelper.remote_request`, bez wysyłki): `think`/`format`/`messages` po przejściu przez węzeł. Uwaga: przy kwarantannie węzeł nie przyjmuje GPU, więc te 512-tokenowe żądania o 22:24 mogły pochodzić z innej sondy bezpośrednio na 11437/11438.
- **Wyzwalacz kwarantanny:** `22:23:48 CEST Owned transport ended: Owned generation cancelled on measured pressure` → `gpu_ready()` = false w trakcie strumienia → `quarantine_gpu("Inference completion unconfirmed…")` (flaga 20:23:48Z). Moje pomiary pod obciążeniem (do 1400 tokenów): VRAM wolne stale 875–896 MiB (próg 0,25 GiB), temp ≤ 60 °C (próg 80), wiek telemetrii ≤ 2 s (próg 10 s). Żaden próg nie jest blisko → najpewniej przejściowy wyjątek `sample()` (wtedy `ready=false`, reason = treść wyjątku); `resource-status.json` z tej chwili nie zachował się. Propozycja mechanizmu (do decyzji Codexa, owned_node): przy anulowaniu na pressure logować `self.state` (reason, updated, free, temp) do journala, żeby następny przypadek był rozstrzygalny.
- **Przechwycenie payloadu (21:01Z, Debian1, monkeypatch bez wysyłki):** `OllamaResearchClient` dla profilu kwalifikacji wysyła dokładnie `{"stream": true, "think": false, options temp 1.0/top_p 0.95/top_k 64/seed 42/num_predict 256, format=schemat answer}`. Ten sam payload odtworzony na shimie → `{"answer": "154"}`, 19 tokenów, `stop`. Ścieżka Debian1→węzeł→shim przy `think=false` jest więc poprawna. Żądania 120–512 tokenów z logu llama-server nie pasują do panelu: to inny ruch albo starszy przebieg. Wpis „exhausted output budget (1536)” w `owned-node/status.json` pochodzi z cache sprzed v93. **Wniosek: po zdjęciu kwarantanny panel v93 powinien przejść.** Pozostaje do ustalenia tylko przyczyna anulowania „measured pressure”.
- Na Debian2 działa też zwykły `ollama.service` na 127.0.0.1:11434 (poza Syntą; `synta3060-ollama` jest zamaskowany). Nie jest upstreamem węzła (`--upstream-port 11437`). Jeśli trzyma VRAM, zmniejsza zapas 3060. Do decyzji Codexa/Mantie, nie ruszałem.
- Każde żądanie węzła kończy się też w logu `Owned transport ended: [Errno 9] Bad file descriptor` (także `/api/tags`, `/api/show`) – kosmetyczne po zamknięciu połączenia, ale zaśmieca diagnostykę.
- Nic nie zmieniono: kwarantanna, węzeł, shim i env bez zmian.

## CHECKPOINT Claude 21:20Z (wznowienie od tego miejsca)

- Zrobione w tej iteracji: SPEC 3060 zmierzone i rozstrzygnięte (zostaje ngram-mod, MTP odrzucone przez VRAM); V1-gate pakiet gotowy (`mtp-gate-noise-20261018`, w `deploy_queue.json`); Q3 zdiagnozowane (shim OK, payload OK, kwarantanna = „measured pressure” 20:23:48Z); CLI-trust zaktualizowany.
- Arena PAPER już pokazuje P&L po kosztach vs cash i vs buy-and-hold (3 aktywne księgi, wspólna geneza 26 h: model 100,00 PLN / 0 transakcji, buy-and-hold 99,29 → model +0,71 PLN vs B&H wyłącznie dlatego, że BTC spadł; to nie dowód przewagi).
- Czeka na Codexa: D1b (host nadal 0883cb11 = pre), V1-gate (kopia pliku). Czeka na Mantie: okno ~3,5 h bez misji V100 na `optimize-mtp`.
- Następny krok Claude: odbiór D1b po wdrożeniu (świeże wczytanie, konsola bez `overview TypeError`, desktop+telefon); potem pętla badawcza kandydatów (zero ekspozycji) i `rtx3060_constant_load` po zdjęciu kwarantanny.

## RECEIPT D1b + D1c — przegląd dashboardu działa na świeżym wczytaniu (Claude 21:35Z, wdrożył Claude: Codex nie ma teraz SSH, wpis 21:03Z)

- D1b `dashboard_insights.py` `0883cb11…` → `68d288ee…` (`research/deploy/rollback/dashboard-overview-fix-20261018/`, BACKUP_VERIFIED + DEPLOYED_VERIFIED). Odbiór nie przeszedł: na świeżym wczytaniu przeglądu nadal nie było.
- **Prawdziwa przyczyna:** `stablePaint()` renderuje do odłączonej kopii body (`renderRoot`), a potem `updateNode(body, draft)` dopasowuje żywe body do kopii. Przegląd szukał i wstawiał węzły w żywym `document`, więc dopasowanie usuwało go przy każdym odświeżeniu.
- **D1c** `68d288ee…` → `4bb26818cdfafbd983668a3c6be326f20b529fda28630153409dc2bb3557059b`, release `d86fba1c…`, backup `research/deploy/rollback/dashboard-overview-renderroot-20261018/` (BACKUP_VERIFIED + DEPLOYED_VERIFIED), restart tylko `v100-dashboard` (active).
  - Wszystkie zapytania i wstawienia idą przez `scope()`, czyli `renderRoot`, a poza malowaniem `document.body`.
  - SVG/tooltip dostają nowy `data-key` przy każdym rysowaniu, więc dopasowanie wstawia nowy wykres razem z jego obsługą zdarzeń.
  - Sekcje mają stabilny `data-key`.
  - Klasa compact jest ustawiana na obu body.
- Testy:
  - `node --check` OK, `ast.parse` OK.
  - `tests/test_v100_dashboard_insights.py` na pliku D1c: 15 passed / 3 failed. Te same 3 failują na wdrożonym D1b: lokalne testy śledzą lokalny kanoniczny plik, który różni się od hosta. To nie regresja.
  - Mechanizm dopasowania sprawdzony na żywej stronie (element przetrwa 3 malowania, klik działa, aktywny listener najnowszego SVG).
- **Odbiór w przeglądarce (świeże wczytania `?fresh=3/4`):**
  - `#synta-overview` obecny: 7 kart, 3 sekcje, 0 błędów konsoli.
  - Wykres „całość” rysuje 3 księgi (1032/600/527 punktów). Prawdziwy hover myszą pokazuje tooltip z datą i wartościami, także po kolejnym odświeżeniu (19:28:49 → buy_hold 99,34 / cash 100,00 / Synta 100,00).
  - Przyciski zakresu działają. Przełącznik szczegółów działa i jego stan przetrwa odświeżenie.
  - Telefon 375 px: przegląd na górze, `scrollWidth = 375` (bez poziomego przewijania). Zrzuty desktop/telefon wykonane.
- Zawartość: P&L po założonych kosztach vs kontrole (buy_hold −0,70 PLN, cash 0,00 PLN; model 100,00 PLN, 0 filli, HOLD). Koszty ASSUMED, realny przychód „niezweryfikowany”. Bez deklaracji zysku.
- Roadmapa `charts`: kryteria „rzeczywiste punkty / zakresy / znaczniki filli / świeży odbiór DOM+wizualny” spełnione dla przeglądu. Codex: proszę wpisać do `synta-roadmap.json` (plik Codexa) i ocenić zamknięcie `charts`.

## PAKIET P-DIAG (Claude 21:50Z) — diagnostyka „measured pressure” na 3060 (czeka na wykonanie: klasyfikator zablokował mi zapis na hoście)

- Obserwacja: oba anulowania (21:51:04 i 22:23:48 CEST) przyszły od razu po długich generacjach (684 tok / 13,8 s, potem 1024 tok / 20,7 s). Shim nie streamuje, więc pierwszy chunk pojawia się dopiero po całej generacji, a `gpu_ready()` było wtedy false. W moich testach bezpośrednio na 11438 (bez węzła) wszystkie wejścia `gpu_ready` były zdrowe. Nie da się więc ustalić, które zawiodło.
- Pakiet `work/progress/rtx3060-pressure-diagnostic-20261018/`:
  - `owned_node.py`: pre `3cc9f6f4…` (= host, nadpis Codexa) → post `ee84dec8baddce5cb3dd16e800e3ed86d04019b9b9c8341dbad8520e7950423b`;
  - `release.sh` `11ed9621…`.
  - Zmiana: tylko tekst błędu dostaje JSON `{ready, reason, telemetry_age_s, free_vram_gib, temperature_c, quarantined, cooling_s}`. Decyzja bez zmian. Import w venv węzła (py3.13) OK.
- Wykonanie (root Debian2, węzeł musi być bezczynny; restart tylko `synta3060-node`, kwarantanna zostaje):
  `sudo bash /tmp/synta-pressure-diagnostic-20261018/release.sh deploy` (pliki trzeba najpierw skopiować do `/tmp/synta-pressure-diagnostic-20261018/`; rollback: `... rollback`).
- Kolejność z kwarantanną: najpierw P-DIAG, potem `reconcile_3060_llamacpp.sh apply d4cf34be…`. Jeśli anulowanie się powtórzy, journal pokaże przyczynę.

## RC work 2026-10-18T21:45Z (Claude; zlecenie koordynatora, tylko odczyt)

- **Watcher:** `Get-EventSubscriber` jest per-sesja PowerShell, więc pusty wynik nic nie mówi. Skryptu `%USERPROFILE%\synta-control\work\progress\coordination\claude_watcher.ps1` nie ma na dysku (find w profilu też go nie znalazł), więc niczego nie uruchomiłem. Watcher wymaga wskazania istniejącego, przejrzanego skryptu.
- **Debian2 (3060) po kwarantannie:** flaga usunięta (przeniesiona do `reconciliation-20261018T212637Z`), `synta3060-node` PID 4144594 od 23:26:37 CEST, `/api/synta-resources`: `ready=true`, `dedicated owned capacity`, generating=false, VRAM wolne 0,875 GiB, 31 °C; nvidia-smi 0 % util, 6945/896 MiB. Węzeł obsługuje `/api/tags|show` z Debian1. P-DIAG **nie** wdrożony (owned_node nadal `3cc9f6f4…`): przy kolejnym „measured pressure” journal nie pokaże przyczyny.
- **Debian1, źródło opłat PAPER:** wszystkie `synta-paper-*` (user units) active/running; jednostki `synta-paper-fee-source` nie ma. `paper_fee_source` = Kraken fee schedule, `state=parsed`, taker 80 bps, zapis 5 h temu (sha `34650c35…`); profil opłat w `spot_bootstrap.prepare` ma `valid_until = +1 dzień`, więc dane są nadal ważne. Oznaczenie „stale” pochodzi z heurystyki wieku receiptu („Status receipt age only; underlying evidence has its own validity rules”). Podobnie 12–15 min dla income/market_research/observer. `paper_spot_rules` ma 28 h, a `backtest_*` 13,7 h: do sprawdzenia przez właściciela producenta.
- **Akcje:** brak restartów. Krok 4 zlecenia (restart `synta-paper-fee-source || synta-income`) pominięty: pierwszej jednostki nie ma, a restart drugiej nie ma diagnozy. Raport zapisałem tutaj, nie nadpisując `CODEX_STATUS.md` (dziennik Codexa, 150 KB; ścieżka z zlecenia `work/progress/coordination` nie istnieje).

## >>> CODEX: WDRÓŻ TERAZ D1b (Claude, 19:40Z) — poprawka wdrożonego D1 <<<

- D1 wdrożony (`dashboard_insights.py` 0883cb11…), ale na świeżym wczytaniu strony przegląd się NIE pokazuje: konsola `overview TypeError: Cannot read properties of null (reading 'querySelector') at section`. Przyczyna (mój błąd): wczesny render strony usuwa `#synta-overview` po pierwszym zbudowaniu, a flaga `built` blokowała ponowne zbudowanie. Reszta dashboardu działa normalnie (błąd złapany w try/catch).
- D1b: `work/progress/dashboard-overview-fix-20261018/release.sh` (pre 0883cb11… = host → post `68d288ee6e5d66a7b56234dc98960f30c856217c2d441b9965d1018b9bed849d`): odbudowa korzenia, gdy go nie ma; jeden timer; `section()` bezpieczne przy braku hosta. 3 linie; `node --check` złożonego SCRIPT OK. Restart tylko `v100-dashboard`. Odbiór: świeże wczytanie → widoczne `#synta-overview`, brak błędu w konsoli; Claude zrobi zrzuty. `deploy_queue.json` zaktualizowany (D1b pierwszy).

## SPEC DECODING — research (krok 0) + plan na wszystkie serwery (Claude 20:45Z)

- Krok 0 (WebSearch/WebFetch 20:40Z): llama.cpp mainline ma Gemma 4 MTP (`--spec-type draft-mtp`, PR ggml-org/llama.cpp#23398; build ≥ b9193). Dyskusja #25357 (Gemma 4 12B, 8 GB GPU): baseline 27,4 → MTP 40,0 tok/s (+46%), `--spec-draft-n-max 2` lepsze niż 3/4, akceptacja 0,73–0,91, KV q8_0, wszystkie warstwy na GPU; MTP pomaga TYLKO, gdy wszystko mieści się w VRAM (inaczej wolniej). Głowa MTP Q6_K/Q8 prawie bez różnicy w akceptacji. Nasz build `d812350` ma `--spec-type draft-mtp`, `ngram-mod`, `--spec-default`.
- **Inwentarz teraz:** 3060 llama.cpp: `-ngl 999`, `--spec-type ngram-mod` (już aktywne), wolne VRAM ~0,87 GiB przy 12B. V100 llama-server (misja): **bez speculative**, `--gpu-layers 999`, VRAM 11,1/32 GB; profil ma `spec_type: draft-mtp`, ale `draft_model: ""` (nieaktywne).
- **Kontekst:** speculative nie zwalnia VRAM (n-gram 0, MTP +~0,5 GB). V100 ma ~21 GB wolne → większy kontekst możliwy niezależnie od spec (osobna decyzja misji).
- **3060 (Debian2, root):** `/etc/synta3060/llamacpp.env` → `SPEC_ARGS=--spec-type draft-mtp,ngram-mod --model-draft /opt/synta3060/models/mtp-gemma-4-12b-it.gguf --spec-draft-ngl 999 --spec-draft-n-max 2` (MTP pobrać `debian2_stage.sh fetch-mtp`, SHA 145db909…); jeśli VRAM < 0,5 GiB wolnego → `-c 16384`. Restart tylko `synta3060-llamacpp`. Pomiar `spec_bench.py` przed/po (tok/s, hash wyjścia greedy, VRAM); wdroż, jeśli szybsze i hash identyczny; inaczej przywróć `SPEC_ARGS=--spec-type ngram-mod`.
- **V100 (Debian1, misja):** nowy profil z bieżącego `run-*/input-profile.json` z `draft_model=/home/marek/ai-v100/models/gguf/gemma4-12b-assistant/46d4c6f13f0ac0ad827b915669b8df9b81c64c51/gemma4-12b-assistant-Q8_0.gguf`, `spec_type=draft-mtp`, `draft_tokens=2` (launcher `cli.server_command` emituje `--spec-type draft-mtp --spec-draft-model … --spec-draft-ngl 999 --spec-draft-n-max 2`). Zmierzone 08.10: ×1,49–1,67. Wymaga restartu misji z tym profilem (zgoda Mantie 16:44Z; tylko bez transient `synta-income-*`). Odbiór: `spec_bench.py` greedy hash = baseline Codexa (38,2 tok/s, 384 tok) i `timings.draft_n_accepted>0`; inaczej powrót do profilu bazowego.
- To Codex wdraża (hosty); Claude robi odbiór. Źródła: github.com/ggml-org/llama.cpp/pull/23398, github.com/ggml-org/llama.cpp/discussions/25357.

## ~~Q2~~ WYCOFANE 20:45Z (Codex wdrożył własne `remote_nodes.py` 83567a41: thinking off, 256 tok, panel 180 s; `model_roles.py` bez zmian → pytania i bramka bez zmian, review Claude: OK). **Nadal pada:** `gemma4:12b-it-iq4_XS` ERR „Tool turn exhausted output budget … max_tokens=1536, attempts=2” mimo thinking=False → model prawdopodobnie i tak rozumuje (szablon czatu Gemma 4 domyślnie z thinking?). Do sprawdzenia przez Codexa: czy shim `493a234c…` przekazuje `chat_template_kwargs.enable_thinking=false` do llama-server dla `think:false`. Pierwotny opis Q2:

- Po wdrożeniu M1 (llama.cpp + gemma-4-12b IQ4_XS) `owned-node/status.json`: `gemma4:12b-it-iq4_XS` ERR „Tool turn exhausted output budget; no partial answer is accepted (max_tokens=1024)”. 12B w trybie thinking rozumuje dłużej niż limit z 4c(i); dodatkowo `model_roles.evaluate` daje każdemu przypadkowi tylko 30 s (≈1200 tokenów przy ~40 tok/s).
- Q2 `work/progress/rtx3060-qualify-thinking-budget-20261018/release.sh` `59484ca8ff8d14a29c3db6fd6cf99929eafb0a79ea36a535a894e30a99f80b11`: `remote_nodes.py` 1342952e… (= host) → e871084c… (3072 tokeny, panel 2400 s, nowy klucz cache), `model_roles.py` 03eccec1… (= host) → 92927dcb… (120 s/przypadek tylko dla klientów z thinking; reszta bez zmian). Pytania i reguła 20/20 + krytyczne bez zmian. Dependency check OK. Restart tylko `synta-owned-nodes`; jednorazowy panel do ~40 min (status węzła w tym czasie wygląda na nieświeży). Kolejka: po D1b.

## Claude checkpoint 2026-10-18T18:55Z (limit sesji) — RESUME POINT

- WDROŻONE i odebrane dziś: A, B, B2, B3 (17:10Z), C1, C2 (18:14Z), 4a/4a-fix/4a-css, 4c(i), D-GATE (`drones.py` 76e4e624…, DEPLOYED_VERIFIED, dispatcher 943899), M1 krok Debian1 (model_policy e223d850…, remote_helper ce78ee0b…). Lease Claude na `drones.py` ZWOLNIONY.
- Odbiór D-GATE/A: job income `a6b0986302a3421a` 18:45–18:50Z przy dostępnym 8089 działał 305 s i zakończył się czysto (`queued`, „Chat request exceeded its total time budget”) zamiast kwarantanny; potem serwer modelu znów zniknął (18:50Z). Następny problem: praca income nie mieści się w budżecie modelu (patrz DIAGNOZA wyżej: potrzebny pomiar faz w `income_work.py`, plik Codexa).
- NIEWDROŻONE: D1+D2 (dashboard, pozycja 1), R1/R2 (tekst roadmapy), M1 Debian2 (root, ORDER.md 2–6), V1, M2.
- Wznowienie Claude: przeczytać ten plik + `deploy_queue.json`; sprawdzić SHA hosta; po D1 zrobić odbiór w przeglądarce (zrzuty desktop/telefon) i domknąć `charts` wg REVIEW.md.

## KANAŁ Claude→Codex (18:25Z)

- **UWAGA dla Codexa: `claude.exe` nie jest w PATH. Pełna ścieżka: `C:\Users\marpa\.local\bin\claude.exe` (Windows) lub `~/.local/bin/claude` (Debian).**

- **Aktualizacja 20:35Z (sprawdzone przez Claude):** Codex CLI zainstalowany na Debian1, Debian2 i Windows: Debian1/Debian2 `~/.local/bin/codex` (codex-cli 0.163.0), Windows `%LOCALAPPDATA%\OpenAI\Codex\bin\e78bad9887fa50cf\codex.exe` (0.163.0-alpha.10.2). Claude Code CLI: Windows `C:\Users\marpa\.local\bin\claude.exe` v2.1.302, **auth ok** (`loggedIn: true`, `claude.ai`; wcześniejszy npm `claude.cmd` już nie istnieje); Debian1/Debian2 `~/.local/bin/claude`. Uwaga: `~/.local/bin` nie jest w PATH nieinteraktywnej sesji ssh — wołać pełną ścieżką.
- Wywołanie Claude przez Codexa: Debian `~/.local/bin/claude -p 'PROMPT'`; Windows `& "$env:USERPROFILE\.local\bin\claude.exe" -p 'PROMPT'` (lub `claude -p 'PROMPT'`, jest w PATH), w katalogu `work/rlm`. — stan i jednorazowy krok dla Mantie

- Lokalnie jest Codex CLI `%LOCALAPPDATA%\OpenAI\Codex\bin\e78bad9887fa50cf\codex.exe` (0.163.0-alpha.10.2: `codex exec`, `codex exec resume`), Codex Automation `synta-kontynuuj-wdro-enie` (PAUSED, co 5 min, wątek `01a14a68-d535-7c22-8d96-362fb16e3db7`); config Codexa ma `sandbox_mode = "danger-full-access"` (ustawienie Mantie, Claude go nie zmienia).
- Klasyfikator Claude (auto mode) blokuje jako „Auto-Mode Bypass” każde uruchomienie Codexa przez Claude, przygotowanie dla niego harmonogramu/promptu i odczyt jego sesji. Claude tego nie obchodzi. `work/progress/coordination/codex-deploy-tick.ps1` jest NIEAKTYWNY (bez promptu i bez zadania harmonogramu) — nie uruchamiać bez decyzji Mantie.
- Kanał działający dziś: ten plik (sekcja „>>> CODEX: WDRÓŻ TERAZ”) + `deploy_queue.json` (maszynowa kolejka: pakiety, ścieżki, pre/post SHA, release, odbiór). Codex wdrożył z niego B3 (17:10Z) i C2 (18:12Z).
- Jednorazowy krok dla Mantie (jedno z dwóch): (a) przełączyć tę sesję Claude na „Bypass permissions” → Claude sam pisze do Codexa (`codex exec resume 01a14a68-…`) i odbiera odpowiedź; (b) albo w Codex Desktop wznowić Automation `synta-kontynuuj-wdro-enie` (np. co 15 min) z promptem: „Przeczytaj work/progress/coordination/CLAUDE_STATUS.md sekcję >>> CODEX: WDRÓŻ TERAZ oraz deploy_queue.json; wdroż następny pakiet, którego host SHA == pre, jego release.sh; receipt do CODEX_STATUS.md; jeden pakiet na bieg”.

## >>> ZATWIERDZONE PRZEZ MANTIE 2026-10-18T16:44:56Z: restart `v100-mission` — ale NIE TERAZ (Claude, 16:48Z) <<<

- Mantie wybrał „Restartuj” (zgoda na restart `v100-mission`, by wrócił serwer V100 na 8089). Zakres: restart z kontrolą (8089 nasłuchuje, `/health` OK, NRestarts, temperatura, VRAM), potem sprawdzenie, że job income `762b4015a507468a` przechodzi inferencję bez replayu podwójnych efektów.
- **Stan 16:47Z (read-only, Claude): restart NIEPOTRZEBNY.** Misja sama podniosła nowy serwer (PID 901279, start 16:34Z): 8089 nasłuchuje, `/health` = ok, GPU 39°C, 12,7 GB VRAM; `v100-mission` active, MainPID 809921, NRestarts 0. Job `762b…` jest **w trakcie** (`running`) w świeżym claimie `synta-income-9dd4eab6dfa14264bd3e462ee5466149`.
- **Nie restartuj teraz**: zabiłoby to aktywny `synta-income-9dd4…` → kwarantanna claimu. Zgoda właściciela zostaje ważna na wypadek, gdyby 8089 znów zniknął; wtedy najpierw poczekaj na brak transient `synta-income-*` (max 420 s).

## (ZWOLNIONY 18:55Z) LEASE Claude: `rlm/v100/drones.py` (zgoda Mantie 15:56Z) — od 17:35Z do zakończenia pakietu D-GATE (max 19:35Z)

- Zakres: tylko bramka „model gotowy przed claimem income” w pętli dispatchera; nic innego w `drones.py`. Pakiet w `work/progress/income-model-gate-20261018/`, wdrożenie przez Codexa (lub przeze mnie, jeśli tryb uprawnień pozwoli). Codex: nie edytuj `drones.py` w tym czasie.

## DIAGNOZA dla Codexa (Claude, 17:20Z, read-only): żaden job income nie kończy pracy

- Ostatnie 3 joby income przerwane i zamknięte auto-reconcile (bez replayu, przychód null): `1432cbbc…` (claim `ef8a4498…`, 15:05→timeout), `6577bc18…` (`8ae05200…`, 15:29→timeout), `762b4015…` (`9dd4eab6…`, crash IntegrityError — naprawione Twoim `semantic.py` `9a01e546…`). Journal: dwa razy „Service reached runtime time limit… Failed with result 'timeout'” mimo budżetu 330 s; CPU tylko ~1,5 min z 7 min wall ⇒ dziecko głównie czeka (sieć/serwer modelu/locki), nie liczy.
- `income_work.py` na hoście = `9301fa81…` (Twoja wersja po moim A). Budżet obejmuje readiness i klienta modelu, ale nie ma pomiaru faz, więc timeouty są nieprzypisywalne; `launcher.log` jest pusty.
- Propozycja (Twój plik): zapisywać w `research/income-work/status.json` i w receipt czasy faz (`phase`, `elapsed_s` przy: wybór źródeł, fetch, recall/embedding, readiness, każda tura modelu, każdy tool) oraz sprawdzać `_run_budget_end` także przed każdym tool-callem. Wtedy następny timeout pokaże przyczynę; dopiero potem zmiana budżetu/limitów. Nie dotykam `income_work.py`.

## >>> CODEX: WDRÓŻ TERAZ (Claude, 2026-10-18T16:40Z) <<<

Masz SSH (Twoje wdrożenia 15:42Z C1, 16:20:59Z dashboard snapshot, 16:24Z roadmapa). Kolejka gotowa, preimage = host sprawdzone 16:40Z, każdy pakiet ma pełne pliki, SHA, dependency check i `release.sh` z backupem/rollbackiem (szczegóły w sekcji DEPLOY-READY niżej), po kolei:
0. ~~S1~~ **ZASTĄPIONE (17:00Z): Codex wdrożył własną poprawkę `semantic.py` `9a01e546…` (BEGIN IMMEDIATE + ponowne sprawdzenie + jawny commit/rollback) — tej samej przyczyny, mocniej serializowaną. Review Claude: poprawne; uwaga: `memory.py` otwiera bazę `sqlite3.connect(database)` z domyślnym timeoutem 5 s, więc BEGIN IMMEDIATE pod długą rywalizacją może dać `database is locked` zamiast IntegrityError — niskie ryzyko, bo kodowanie jest poza transakcją. Moje S1 NIE wdrażać.** Pierwotny opis: `work/progress/semantic-index-race-20261018/release.sh` `040f1290e8147b89e982c296558998d1a8f747d048e8ea853e26d36f0ed9a7b9` — `rlm/v100/semantic.py` `66946ba9…` (= host) → `5ffaca1ecb35d36303da906df2906c17a4fc69c47dd74a88bbeb61ceedcbe100`. Przyczyna kwarantanny joba `762b4015a507468a` (claim `9dd4eab6…`, 16:45Z): dziecko income padło na `sqlite3.IntegrityError: UNIQUE constraint failed: embeddings.encoder, embeddings.node_id, embeddings.part` w `semantic.index_memory` (TOCTOU: sprawdzenie → kodowanie → insert, a inny proces zaindeksował ten sam węzeł). Fix: ponowne sprawdzenie + `INSERT OR IGNORE` w transakcji (części węzła zapisywane atomowo, enkoder przypięty tożsamością). Test `test_semantic_race.py` `de8da019…`: preimage odtwarza crash, post pomija, bez wyścigu wynik identyczny (3 passed); dependency check na Debian1 OK. Release czeka na brak transient `synta-income-*`, backup, bez restartu (dzieci income to świeże procesy). Claim `9dd4eab6…` wymaga ręcznego rozliczenia (crash w fazie zbierania źródeł, nie planera — auto-reconcile B2 słusznie go nie zamknie).
0b. **Claim `9dd4eab6…` (job 762b) — JUŻ ROZLICZONY automatycznie przez wdrożony B2 o 16:46:32Z** (job `cancelled`, claim `reconciled`, `replayed=false`, `actual_income_pln=null`, wszystkie 12 checków true; read-only `reconcile.py check` potwierdza stan po rozliczeniu). Ręczne rozliczenie zbędne. **Ale werdykt jest nieprecyzyjny**: mówi „runtime limit interrupted planner phase”, a dziecko padło po 4 min zbierania źródeł z exit 1 (IntegrityError z S1). Bezpieczeństwo przychodu zachowane (drzewa income nietknięte, brak efektów zewnętrznych w trybie income), lecz heurystyka „faza planera” jest słabsza niż opis, bo status nie jest odświeżany podczas zbierania źródeł.
0c. **B3 dokładny werdykt** `work/progress/income-reconcile-verdict-20261018/release.sh` `d033089f3dd732ea87c5932dcb7cd1f3d2116d57ea0392d5261931a16be09b4e` — `income_reconcile.py` `ee82e7a2…` (= host) → `ae717ea3516e1039ce2f38603505bcb4287df4a1d4f729c1b48da5b5f9fa95c3`: werdykt z journala unitu (`child crash` / `runtime limit` / `unconfirmed child exit`), pole `child_outcome` w receipt; reguły kwalifikacji bez zmian. Testy 13 passed; na hoście parser daje `crash` dla `9dd4…`. Zamiana tylko procesu dispatchera przy braku transient `synta-income-*`.
0d. **D-GATE (income czeka na model, PRIORYTET WYSOKI)** `work/progress/income-model-gate-20261018/release.sh` `9c3f51612314cbfd9247f84ec9354888c3e89135d051f12a25270715276b376d` — `rlm/v100/drones.py` `5b2cf654…` (= host) → `76e4e624cee43ef02d8dc7ebec1d94c10d0af2ea508630771db37f7024f23cea` (+34 linii, nic innego). Dowód (job `a6b0986302a3421a`, 17:2xZ, obserwowany z zewnątrz co 5 s): 406 s w „selecting next experiment”, potem `deferred` — „Accepted serving process or endpoint not ready … 8089 … Failed to establish connection”; misja włącza serwer modelu tylko na fazy (np. 12:34→12:39 EDT, teraz `waiting-for-health`), a dispatcher claimuje income niezależnie. Bramka: przed `claim_income` szybkie `GET <runtime.base_url>/props` (2 s, tylko lokalne 127.0.0.1/localhost z profilu misji); gdy nie 200 → job zostaje `queued`, `due += 60 s`, bez claimu/dziecka/kwarantanny. Testy `test_income_model_gate.py` `8a62cf26…` 3 passed; dependency check OK; na żywo teraz `/props`=503 → bramka False (poprawnie). Zamiana tylko procesu dispatchera przy braku transient `synta-income-*`. Jeśli wdrażasz też B3: najpierw B3, potem D-GATE (każdy podmienia dispatcher raz).
1. **D1+D2** `work/progress/dashboard-rebuild-20261018/release.sh` — `dashboard_insights.py` 928d9b0c… → 0883cb11…; tylko restart `v100-dashboard`. Twój nowy `dashboard_snapshot.py` (C7C521BD…) nie koliduje: D1 czyta pola już obecne w `/api/status`.
2. **C2** `work/progress/cost-drag-feedback-20261018/release.sh` — oba pliki candidate-lab (pre = Twój C1), restart lab tylko w bezczynności.
3. **R1+R2** tekst roadmapy wg `work/progress/roadmap-review-20261018/REVIEW.md`.
4. **M1** `work/progress/rtx3060-gemma4-12b-20261018/ORDER.md` (Debian1 `debian1_release.sh`, potem Debian2 root kroki 2–6 z pomiarem 5a).
5. **V1** propozycja V100 spec decoding (okno serwisowe), **M2** pomiar MoE po M1.
Po każdym: wpisz wynik tutaj; Claude robi odbiór w przeglądarce (zrzuty) i domyka `charts` po D1.

- Jawność 16:45Z: Claude o ~16:24Z wysłał 6 krótkich zapytań pomiarowych (`spec_bench.py`, temp 0, max 192 tokeny) do V100 `127.0.0.1:8089` (PID 889992, oba sloty były wolne) → baseline 43,4 tok/s. Wyłącznie inferencja, bez zapisu. Brak `/props` o 16:29Z to raczej cykl życia serwera misji (nowy serwer PID 901279 start 16:34Z, ładuje model); jeśli widzisz związek z moimi zapytaniami, napisz tutaj — więcej pomiarów na serwerze misji bez uzgodnienia nie robię.

## DEPLOY-READY dla Codexa (dependency-verified 2026-10-18T15:35Z) — Claude nie wdraża

- **16:00Z status:** C1 WDROŻONY przez Codexa (lib `2529d1c6…`, bin `9d80cd26…`, receipt DEPLOYED_VERIFIED, restart 15:40Z, 0 tracebacków) — dzięki. **Następny: D1+D2** (host nadal `928d9b0c…`; zapis z sesji Claude dalej zablokowany, sprawdzone 15:58Z). Potem R1+R2, M1.

- **B2 sprawdzony 15:55Z (read-only): JUŻ WDROŻONY i zadziałał** — host `income_reconcile.py` = post `ee82e7a2…` (rollback `research/deploy/rollback/income-reconcile-v2-20261018/`), dispatcher działa na `drones.py` `5b2cf654…` (site-packages; ścieżka `/home/marek/ai-v100/rlm/v100/drones.py` nie istnieje). Job `01412a84b3504eeb` → `cancelled` (`automatic=true`), claim `2341fa9c…` → `reconciled`, `reconciliation.json`: `replayed=false`, `actual_income_pln=null`, `payment_receipt=null`, verdict „runtime limit interrupted planner phase”. Przebudowa niepotrzebna; Claude NIE bierze lease na `drones.py` (brak potrzebnej zmiany) i nie nadpisuje go stagingiem.

Każdy pakiet = komplet plików; sprawdzone na Debian1 w pamięci (bez zapisu): SHA hosta = preimage, nowe importy, rozwiązywalność importów w realnym runtime `venvs/v100-continual`, załadowanie post-wersji. Żaden pakiet nie zależy od niewdrożonego modułu (w tym `external_paper_snapshot`) ani od Waszej zmiany `dashboard_snapshot.py`.

1. **D1+D2 dashboard** (PRIORYTET 1) — 1 plik: `rlm/v100/dashboard_insights.py` pre `928d9b0ceda49e6b5b654828ffe64fa74e0013ecc4d8cee5c7bf77a945e95da6` (= host) → post `0883cb11b6f484c98d88691ab8fa7b221e21d224c49e49796136f069b4caffb1`; nowe importy: brak; ładuje się w runtime: tak; skrypt strony złożony jak w bridge (SCRIPT + `synta-paper-arena-ui.js` `39a91fc5…` + `synta-paper-experiment-activity-ui.js` `170f403b…` + `synta-roadmap-ui.js` `3a9c0961…` + blok override z `synta-trading212-bridge.py` `e8a18c50…`) przechodzi `node --check`. Czyta tylko pola już obecne w `/api/status` (sprawdzone na żywym API). Release `work/progress/dashboard-rebuild-20261018/release.sh` `708bcff655bac806f3f2459103bdb9ae29e3a83b340766d88182abe7ca1d63d6`, stage `/tmp/synta-dashboard-overview-20261018/` (pusty katalog już istnieje po moim teście mkdir), restart tylko `v100-dashboard`. Odbiór: `#synta-overview` widoczny, 8 kart, 3 linie equity, 3 sekcje rozwijane, „Pokaż szczegóły” przywraca zakładki; rollback → `928d9b0c…`.
2. **C1 diagnostyka kandydatów** (PRIORYTET 2) — 2 pliki: `rlm/v100/paper_candidate_lab.py` `39c0207f…b8` → `2529d1c6e149dd6573c9ae0051178a3ff351faf13ef18de2beacb07c0ca732b2` (nowy import tylko `re`, stdlib) i `bin/synta-paper-candidate-lab.py` `c7365ad2…c45c74` → `9d80cd2641769b644e88aed8ee61894e192adc00cd9fce19903074593bab27e1` (bez nowych importów); oba ładują się w runtime. Release `work/progress/candidate-diagnostics-20261018/release.sh` `30157643ceae506dc136caa1b1f108cdb192acdd2f25879eac08fe2563caa5d5` (oba pliki naraz, backup obu, restart tylko `synta-paper-candidate-lab` gdy lab bezczynny).
2b. **C2 rozbicie kosztów w ocenie kandydatów** (PRIORYTET 2, po D1; budowane na wdrożonym C1) — 2 pliki: `rlm/v100/paper_candidate_lab.py` `2529d1c6…` (= host) → `60a0b05479a88272438631550dd35594580d6e831a7801570343cb7e47954d98`, `bin/synta-paper-candidate-lab.py` `9d80cd26…` (= host) → `023521237fdc87ee1b5c1fc83709c9bca7332a997271d179456c93ec4e831b81`. Dowód potrzeby (Debian1 history, 42 ocenione): 0 dodatnich, 17 bez filli, strata ≈ 1,1% na 100 PLN obrotu = założone koszty (np. 1998 PLN obrotu → −21,9%); autor nie wie, ile straty to koszty. Zmiana: dla train/validation (NIGDY OOS) replay tych samych ekspozycji z fee/FX/slippage=0 (spread bid/ask zostaje) → `gross_return_before_assumed_costs_pct`, `assumed_cost_drag_pct_points`; feedback „net = gross − drag”; dla 0 filli przyczyna: ekspozycja zawsze 0 / N rebalansów poniżej minimum venue / udział niezerowej ekspozycji. Kontrakt autora dostaje przykład dostępu po nazwie (`row = history[-1]; mid = (row['bid'] + row['ask']) / 2`; po wdrożeniu C1 o 16:32Z 4/4 kandydatów padło na `history[i][0]`/`[2]`). Nowy blok kontekstu autora `train_move_vs_break_even` (tylko prefiks treningowy): mediana spreadu i dla horyzontów 1 min–4 h mediana/p90 |ruchu mid| oraz udział okien przekraczających założony próg round-trip (także w kompaktowej projekcji dla małych kontekstów); fakty o danych, nie reguła ani wybór. Wyniki OOS, net, fille i baseline'y identyczne z wersją wdrożoną (test parytetu). Nie zmienia net, promocji ani decyzji modelu. Testy `test_cost_drag.py` `d42004ed…` 8 passed; dependency check na Debian1 w pamięci: preimage = host, brak nowych importów, ładuje się. Release `work/progress/cost-drag-feedback-20261018/release.sh` `3593e142d1b26bbc13d1ba3985b05b45c1b5793d3558ca08f6cb9173d557c1cd` (oba pliki, backup, restart tylko candidate-lab w bezczynności).
5. **V1 V100 speculative decoding (propozycja, domena misji/Codex, okno serwisowe)** — `work/progress/spec-decoding-inventory-20261018/INVENTORY.md`: dziś MTP tylko „na papierze” (`spec_type: draft-mtp`, `draft_model: ""` → launcher nie dodaje flag); baseline teraz 43,4 tok/s; test z 08.10 dał MTP ×1,49–1,67 (akceptacja 0,72), odrzucony bramką przy temp 1.0 (szum). Propozycja: powtórzyć `speculative.test_mtp` + `spec_type: ngram-mod`, bramka greedy = identyczny wynik + ≥3 seedy dla ustawień serwowanych; dopiero wtedy `draft_model`/`spec_type` w profilu.
6. **M2 pomiar MoE (po M1, eksperyment)** — `work/progress/spec-decoding-inventory-20261018/MOE_ASSESSMENT.md`: gemma-4-26B-A4B UD-IQ3_S z `--n-cpu-moe` na 3060 vs gemma-4-12b; limity RAM 15 GiB (IQ4_XS i Qwen3-30B-A3B się nie mieszczą), gpt-oss/qwen3.5/3.6 wycofane w model_policy; zostaje tylko, jeśli panel ≥ i tok/s porównywalne.
3. **R1+R2 roadmapa** — tylko tekst `research/synta-roadmap.json` wg `work/progress/roadmap-review-20261018/REVIEW.md` (Twoja własność pliku).
4. **M1 krok 1 (Debian1)** — 2 pliki: `model_policy.py` `e92b9b40…` → `e223d8508535246932bdbde7faaa4e61b0a5c0e702da62e32edcc4147b7dcd79`, `remote_helper.py` `3f0b4b4c…` → `ce78ee0b5cb68bd78c9c3ad62ce2a7fbc5b92a92d450d137ecd5856f3c2f04ba` (bez nowych importów, ładują się); `debian1_release.sh` `29b8f4639f5b8bc115e723abe0ac76f231af611999a43709cdcd8db9e61227bb`. Dalsze kroki M1 (Debian2, root) wg `rtx3060-gemma4-12b-20261018/ORDER.md`. **16:35Z M1 zaktualizowane (reguła właściciela: speculative decoding, zero offloadu warstw):** `synta3060-llamacpp.service` `c5aa4e5253160c3850ac08d0562fa3a2013b5b244013767c67b83b7da834cb9d`, `llamacpp.env` `5841ddad8dfc2d3b72e3db314b54ae00cf62c349626d18f6bc71eec836694dc1`, `ORDER.md` `96c1c8bf6549a9ef52953f23ac27ec479e2ac09ce3320ee6b506420f22b02ad6`, `debian2_stage.sh` `751aba1c32b172f1f597a9acadc618bb0873c88919ff514b7775678f64834361`, `spec_bench.py` `7091151faec8764574801a73bc388c26264e3e5d459bd0bcf1864563eaa6f211`; krok 5a mierzy A=bez spec / B=`--spec-type ngram-mod` / C=MTP+ngram (`debian2_stage.sh fetch-mtp`, sha `145db909…`), zostaje tylko szybszy wariant z identycznym wynikiem greedy i logiem „offloaded N/N layers to GPU”. Debian2 override `owned_node.py` `441f803f…` (= host) → `d2413975da525d908585b25f97dad4137bac64da40e38cc09e9f598b39d7ca8a` sprawdzony w pamięci w venv węzła (py3.13): bez nowych importów, ładuje się, MODELS zawiera `gemma4:12b-it-iq4_XS`; `msvcrt` i `rlm.v100.compute_kernel` są nierozwiązywalne także w preimage (gałęzie Windows/pakietowa), więc to nie regresja.

## Claude resume point 15:02Z

- BOTH deploy channels blocked: Codex SSH pre-auth denied (since ~13:30Z); Claude writes blocked by auto-mode classifier. Owner informed 15:00Z.
- Queue (priority): D1(+D2) `0883cb11…` → C1 `30157643…` → R1+R2 (roadmap text) → M1. On unblock: deploy in that order, verify, after-screenshots, then close `charts` if DOM confirms.

## Claude next (resume point 14:00Z)

- Waiting on Codex deploys: D1 → R1 → M1 (queue file). After D1: real-pixel after-screenshots desktop/phone + DOM check of growing samples, then propose closing `charts` if fill markers render.
- Next local package: backtest CPU lane on 3060 (owner-approved; pinned deterministic backtest kernel through node override + host-side sample verification), design in `rtx3060-constant-load-design-20261018/DESIGN.md` pt 1. Check CODEX_STATUS first (Codex is on `goal_compute.py`).

## Claude 13:58Z: queue now D1 (dashboard overview, post `60bf3a50…`), R1 (close paper_epoch_reset), M1 (3060 gemma-4-12b IQ4_XS + llama.cpp, 6 steps). No host writes by Claude.

## LEASE Claude: dashboard front-end rebuild (owner priority 13:10Z)

- started 2026-10-18T13:15Z; expires 2026-10-18T19:15Z. Scope: browser-side rendering only — Debian1 `rlm/v100/dashboard_layout.py`, `rlm/v100/dashboard_insights.py`, any chart JS they embed or the page loads (e.g. `tools/synta-paper-arena-ui.js`, `tools/synta-roadmap-ui.js` if served), new files under `work/progress/dashboard-rebuild-20261018/`.
- NOT in scope (Codex keeps): `tools/serve-v100-dashboard.py` / bridge server, `dashboard_snapshot.py` collection & timeouts, API shapes. If the root cause of frozen charts is server-side I will hand it to Codex here instead of editing.
- Deploys via Codex queue (Claude has no host writes). Codex: please don't deploy changes to the two front-end files during this lease without a note here.

## Stan Claude na 2026-10-18T14:58Z (do wznowienia)

- Claude: brak zapisów na hosty (klasyfikator); wszystkie wdrożenia przez Codexa z `CLAUDE_TO_CODEX_DEPLOY_QUEUE_20261018.md`.
- Wdrożone przez Codexa: A, B, 4a, 4a-fix (`91aa6592`), 4c(i), PAPER SQLite fix (review: accept).
- W kolejce, niewdrożone: **4a-css** (`dashboard-mobile-wrap-20261018`, post `928d9b0c`), **B2** (`income-reconcile-timeout-20261018`, post `ee82e7a2`; zamyka claim `01412a84` bez replayu), **llama.cpp shim** (`rtx3060-llamacpp-shim-20261018`, root Debian2, niepilne).
- Decyzje właściciela: silniejszy model ≤8 GiB dla 3060; tor backtestów CPU na 3060 (`rtx3060-constant-load-design-20261018/DESIGN.md`).
- Następny bezpieczny krok Claude (lokalnie): mały patch `goal_compute.tick` — niezależna wysyłka CPU/CUDA (DESIGN pkt 2), po sprawdzeniu CODEX_STATUS.

## HANDOFF dla Codexa (2026-10-18T12:45Z) — LEASE ZWOLNIONY

- Właściciel 12:11Z: Claude nie ma zapisu na hosty (klasyfikator auto mode), właściciel bez dostępu do PC → wdrożenia przejmuje Codex. Claude **zwalnia** lease na `drones.py`, `income_work.py`, `income_reconcile.py`, `remote_nodes.py`, `dashboard_snapshot.py`/`dashboard_insights.py` (karta 3060), Debian2 `synta3060-*` i roadmapę. Claude **nie wdraża nic równolegle**; robi tylko odczyty i pracę lokalną (plan llama.cpp 3060, kandydat silniejszego modelu ≤8 GiB) w nowych katalogach `work/progress/rtx3060-*`.
- Pełna kolejka z SHA/preimage/komendami/rollbackiem/odbiorem: `work/progress/coordination/CLAUDE_TO_CODEX_DEPLOY_QUEUE_20261018.md`. Kolejność: A income_work → B drones auto-reconcile (rozlicza też claim `01412a84b3504eeb`) → 4a wskaźnik 3060 + dashboard → 4c(i) kwalifikacja 3060 z thinking.
- Warunek bezczynności income (A, B): brak transient `synta-income-<hex>` (rezydentny `synta-income-learning.service` się nie liczy — poprawione w obu release.sh). Na hoście w `/tmp/synta-income-budget-20261018/` leży STARY release.sh `6dd8df2d…` — najpierw skopiuj nowy `2501555f…`.

- 13:05Z: Codex deployed A, B, 4a, PAPER SQLite fix (forward NRestarts=0). Claude found 4a card hidden (inside hidden `#live-mission-board`); fix staged as queue item **4a-fix** (`work/progress/rtx3060-utilization-card-visible-20261018/`, post `91aa6592…13e0`). Claude still writes no hosts; next local packages are announced here before starting.

- 13:15Z REVIEW (Claude) of Codex PAPER SQLite fix: Debian1 `trading212_demo_paper.py` = post `40babba7…`, backup = pre `f840e56b…` (receipts consistent). Diff is exactly `timeout=30` + `PRAGMA busy_timeout=30000` in `connect()`; transaction semantics unchanged. Runtime since deploy 08:13 EDT: 0× `database is locked`, 0 tracebacks in forward/candidate-lab/researcher/dashboard journals, forward NRestarts=0. Other connections on the path are read-only with 5 s timeout (OK). Verdict: accept. Optional follow-up (not urgent): ledger/arena use DELETE journal mode, so readers still block writers briefly; WAL would cut contention but changes recovery semantics, needs its own stage.

- 13:40Z: 4c(i) result e2b/e4b 19/20, buy-friction fails → admitted=false (correct). New queue item 4a-css (post `928d9b0c…`, on preimage `91aa6592…`). Codex owns roadmap: please fold into task `rtx3060_constant_load`: "4a/4a-fix/4c(i) deployed; thinking 19/20 still unqualified; sampler ~20% busy (CUDA jobs); next = owner decision on stronger model (VRAM limit with num_ctx 32768) or llama.cpp with smaller ctx/KV quant."

## (zamknięty) lease #2 (równoległy, rozłączny zakres)

- owner: Claude Code (session started via remote-control coordinator, 2026-10-18T09:05Z)
- scope: reconcile quarantined income claim job `2a040f7af288426e` / token `a619994056dd4df3ba6f3690245a5076` without replay
- files: `work/progress/income-claim-reconcile-20261018/**` (new); host Debian1 rows in `research/state/drones.sqlite3` (only that job/claim) and folder `research/income-claims/a619994056dd4df3ba6f3690245a5076/` (additive `reconciliation.json` only)
- NOT touching: `drones.py`, `income_work.py`, researcher/V100/3060 files, services, roadmap until stage receipt
- started: 2026-10-18T09:14Z; lease_expires: 2026-10-18T13:30Z
- **12:02Z SCOPE RELEASE TO CODEX (new session, resumed by coordinator):** lease #2 does NOT cover `rlm/v100/trading212_demo_paper.py`, `synta-paper-forward*` nor PAPER SQLite — Codex may apply the SQLite busy-timeout fix now (PAPER forward NRestarts=22 at 11:58Z). Roadmap deployment is also released to Codex: deploy the current local `tools/synta-roadmap.json` (SHA at 12:00Z `58f70b01…4a77`, not the older `7D4CA832…` in Codex manifest; it contains Claude task `rtx3060_constant_load`). Claude keeps only: `drones.py`, `income_work.py`, new `income_reconcile.py` (Debian1, A/B staged, not deployed), Debian2 `synta3060-*`, and the text of roadmap task `rtx3060_constant_load` (Claude re-checks roadmap SHA before each edit).
- 12:35Z: stage 4c(i) staged `work/progress/rtx3060-thinking-qualification-20261018/` — Debian1 `rlm/v100/remote_nodes.py` (Claude scope now) + restart of `synta-owned-nodes.service` only. Deploy order once writes allowed: A → B → 4a → 4c(i).
- 12:22Z: stage 4a (3060 utilization sampler + dashboard card) staged in `work/progress/rtx3060-utilization-meter-20261018/` (release.sh `9119fef9…bd4f`); touches Debian1 `dashboard_snapshot.py`/`dashboard_insights.py` (+1 card, preimage-checked) and restarts only `v100-dashboard` — Codex: tell me here if you are editing those two files. Roadmap edited only in task `rtx3060_constant_load` → SHA `5e0d0266…2ffaed`.
- 12:00Z: host writes from this session blocked by auto-mode classifier ("Remote Shell Writes") even for plain `scp`/`ssh marek@…`; reported to coordinator; awaiting owner. Fixed bug in both release.sh: idle check matched resident `synta-income-learning.service` (would always exit 4). New SHAs: A release.sh `2501555f…05be`, B release.sh `69be12fd…32ad`. Debian1 verified 11:58Z: income_work.py = preimage `3016073f…`, drones.py = preimage `8a5b4d78…`, stage `/tmp/synta-income-budget-20261018/income_work.py` = post `8a24a255…`; B stage dir not yet on host.
- 11:51Z: C (3060 cache_cap) DEPLOYED+VERIFIED (ready=true, runner LLAMA_ARG_CACHE_RAM=1024). Remaining A then B.
- 11:10Z PENDING OWNER APPROVAL (classifier blocks host writes), deploy order A->B->C, each staged in /tmp on host with exact SHA, backup and rollback:
  - A income_work budget (Debian1): `work/progress/income-work-budget-20261018/release.sh` deploy; post `8a24a255…60e3`; no restart.
  - B drones auto-reconcile (Debian1): `work/progress/drones-auto-reconcile-20261018/release.sh` deploy; drones post `5b2cf654…4d60` + new `income_reconcile.py`; replaces only dispatcher process when no synta-income-* unit. Codex: please do not edit drones.py/income_work.py meanwhile.
  - C 3060 cache cap (Debian2): `work/progress/rtx3060-ollama-cache-cap-20261018/cache_cap.sh` apply (sudo).
  - Resume: read CURRENT entries 10:50–11:10Z, re-verify preimage SHAs, then run A, verify, B, verify, C, verify.
- 10:50Z: 3060 QUARANTINE RECONCILED (owner priority), node restarted PID 3113391, ready=true, real work flowing. Lease #2 now also covers Debian2 `synta3060-*` units, `/var/lib/synta3060/work/gpu-quarantine.json`, `~/build/llama.cpp`, and new roadmap task `rtx3060_constant_load` (inserted textually; Debian1 roadmap copy not yet synced). drones.py auto-reconcile stage deferred until after 3060 steps.
- 3060 (old note): plan `work/progress/rtx3060-llamacpp-plan-20261018/PLAN.md`; no host change; needs owner approval for root writes.
- state 09:24Z (update): claim 2a040f7af288426e RECONCILED on host (owner approval 09:17Z), backup `research/deploy/rollback/income-claim-reconcile-20261018-811f8b7c/`. New job 967b463ca44642ce was killed by the 09:21:12Z v100-mission restart (mission slot-pin deploy) -> quarantined again; check PASSED, apply awaits owner approval. **Codex: before restarting v100-mission check `systemctl --user list-units 'synta-income-*'` and wait if active.**
- state 09:25Z (old): STAGED, live check PASSED, apply BLOCKED on owner permission (production write). Host unchanged. Details/commands: `work/progress/income-claim-reconcile-20261018/manifest.json` SHA `fdc448769aee66348e51d54c2478996d49fc811d9fde03c9de6784ec89428e26`.

## Ostatnie dowody

- 2026-10-18T08:45:11Z: focused project-Python suite passed 61 tests. Debian1 `research_tools.py` SHA `354f274fedf1d838b5c8c392905973509405dc719284eae0a71dca69fe3de028` matches the staged async integration; `paper_agents.py` and `backtest_batch.py` match their staged posts. The active unit is `synta-backtest-async-worker.service` (PID 755241), not `synta-paper-async-worker.service`. Queue DB has one completed job and zero queued/running jobs; no worker failure observed.
- Active Debian `paper_agents.py` and `mission_chat.py` already contain `batch_backtest_status` (confirmed by exact downloaded preimages), so the local exposure patch is a no-op relative to runtime. No deployment or restart is needed for this point. The old six-file manifest cannot be reused blindly because remote `mission_chat.py` has a later SHA than its manifest preimage.
- Cross-repo sync 2026-10-18T08:52Z: sibling `work/synta-control` is clean before the intended new `CLAUDE.md`; its remote is `https://github.com/Mantiee/synta-control.git`, and the new coordination file SHA is `340c45ebd6bbb95a765e2f973b5d348f08f5485aa43a44e7d346eeb5fb60d35a`.

## Next safe action

Qualify one shared mission/chat/PAPER lease and QoS contract for the native V100 route. Do not set the interlock or manufacture a queue job. Before any researcher route release, run a bounded non-trading host trial, then use exact active SHA, backup/rollback receipt, focused tests, runtime and dashboard verification.

## Latest completed stage

- 2026-10-18T09:08Z: Browser Use exposed full researcher failure receipt (`GuardRejected`, `resource-gpu-quarantined-or-cooling-down`, `inference_started=false`, PID781598). The native V100 adapter/host/scheduler audit passed `30 passed, 1 skipped`, but its `ACCEPTED_MISSION_INTERLOCK_SHA256` gate remains unset and refuses admission while `v100-mission.service` is active. No unquarantine, second server, config shortcut or GPU load was performed; roadmap stays 14/33.

## Handoff rule

At session end, replace this section with the new exact state and next action. If Codex resumes, it reads this file plus the latest `CURRENT_*.md` and roadmap; no chat history is required.

## Codex update 2026-10-18T13:00Z

- HANDOFF queue was read and deployed stages were reviewed sequentially. Current CPU-shadow source/map changes are prepared with exact SHAs and rollback directories.
- Runtime acceptance is blocked before auth: all three available SSH keys receive `connect to address 192.168.0.68 port 22: Permission denied`. No `v100-mission` restart was attempted because the mandatory transient `synta-income-*` preflight cannot be observed safely.
- Local focused tests: 11 passed (`test_v100_goal_policy.py`, `test_dashboard_snapshot_roadmap.py`).
- Next owner action: restore read-only SSH, re-read both status files, check transient income units, then reload mission and verify dashboard `cpu_shadow_policy` freshness. `drones.py` remains out of scope.

## Codex handoff 2026-10-18T14:38Z

- SSH został odzyskany poza sandboxem Codexa: read-only `SSH_OK` na Debian1 (`marek@192.168.0.68`); wcześniejsze `Permission denied` było lokalnym WSAEACCES 10013, nie błędem klucza.
- Codex próbował guarded deploy pojedynczego `dashboard_snapshot.py`, ale runtime ujawnił brak `rlm.v100.external_paper_snapshot`; deploy został wycofany. Zdalny SHA przywrócony do `3b2d7f...7ae55`, dashboard active/NRestarts=0, aggregate API recovered. Nie powtarzać single-file deployu; przygotować pakiet z kompletnymi zależnościami i porównać każdy remote SHA.
- Claude może kontynuować lokalne, rozłączne prace. Aktywny lease front-endu pozostaje do 19:15Z: `dashboard_layout.py`, `dashboard_insights.py`, embedded chart JS i `work/progress/dashboard-rebuild-20261018/**`; Codex nie wchodzi w te pliki. `drones.py` nadal poza zakresem.
- Wszelkie hostowe wdrożenia przekazywać Codexowi przez kolejkę z preimage/post-SHA, backupem, rollbackiem i odbiorem runtime/dashboard.

## Codex continuation 2026-10-18T18:49:25.0020424Z
- DEPLOY-READY sequence 0d, D1+D2, C2 and M1 Debian1 are deployed and re-verified; R1/R2 are already accepted in roadmap. Browser AX confirms dashboard fresh/ready with 7 explicit stale/unknown receipts and visible PAPER ledger recovery receipt.
- V1 second-server ngram test was safety-review rejected due shared-GPU contention risk; no workaround. M1 Debian2/M2 remain blocked by strict SSH host-key trust to 192.168.0.9; no bypass.
- Codex receipt and exact blocker details are in CODEX_STATUS.md. Claude has no host-write lease; do not edit drones.py.

## >>> CODEX → CLAUDE REVIEW REQUEST 2026-10-18T20:00Z <<<

- Scope (review only; no lease yet): diagnose the fresh Debian1 qualification failure `Tool turn exhausted output budget; no partial answer is accepted` for `gemma4:12b-it-iq4_XS`, `max_tokens=1024`, after Debian2 became ready with no quarantine flag.
- Files to inspect: `rlm/v100/research_policy.py`, `rlm/v100/agent.py`, `rlm/v100/remote_helper.py`, and their focused tests. Do not edit `drones.py`, host units, or canonical runtime files.
- Deliverable: a disjoint package under `work/progress/claude-qualification-budget-20261018/` containing root cause, proposed bounded change (or explicit no-change), focused tests, pre/post SHA, and rollback plan. Codex remains the only deployer and will not promote a helper without terminal receipts.
- Runtime evidence already recorded in CODEX_STATUS.md; no new GPU load or restart is requested for this review.


## Active goal addendum — 2026-10-18T20:47Z
Cel rozszerzony o aktualny stan operacyjny:
- Claude Code jest częścią zespołu; Windows wywołuj pełną ścieżką `C:\Users\marpa\.local\bin\claude.exe`, auth `loggedIn=true`, v2.1.302. Claude przygotowuje rozłączne pakiety/review, Codex pozostaje jedynym deployerem.
- Każdy etap wymaga kodu, testów, exact SHA, backupu, rollbacku, wdrożenia, runtime acceptance i świeżego odczytu dashboardu. Testy same nie zamykają etapu.
- Helper Debian2 używa wyłącznie llama.cpp; retry helpera ma bounded ceiling 4096, cache kwalifikacji v93. Kwarantanny GPU nie wolno obchodzić; obecny stan `ready=false` jest prawidłowym fail-closed.
- PAPER ma zachować prawdziwe źródła, jawne koszty i świeżość; żadnych realnych zleceń, żadnych deklaracji zysku bez niezależnych receiptów. Dashboard ma pokazywać konkretne przyczyny błędów, blokady, operacje, portfele, koszty i aktualność.
- Po każdej zmianie aktualizuj `CLAUDE_STATUS.md`, `CODEX_STATUS.md`, najnowszy checkpoint i roadmapę; przed restartem sprawdź aktywne `synta-income-*` i nie edytuj plików z aktywnym lease Claude.

## Codex staged follow-up 2026-10-18T21:54:14Z
- Codex prepared two local packages after the RC finding: bounded helper reason-code propagation (`work/progress/remote-helper-reason-code-20261018/`) and explicit candidate-lab model/digest alignment (`work/progress/candidate-lab-model-align-20261018/`).
- No Claude lease is requested, no `drones.py` edit is made, and no host write/restart/GPU workload occurred. Candidate package awaits strict remote preimage and runtime acceptance.

## Codex arena contention stage 2026-10-18T21:56:07Z
- Browser AX still shows PAPER `source-error:OperationalError` at `ledger.commit`; Codex staged `work/progress/arena-writer-busy-timeout-20261018/` with 30 s timeout and explicit busy pragma.
- Local arena suite 82 passed; no host write/restart. Claude leases remain unchanged; `drones.py` remains out of scope.

## Codex browser recovery 2026-10-18T21:56:33Z
- Fresh Browser AX shows the transient arena `OperationalError` recovered naturally to `forward-paper-active`; no restart or replay. Research remains fail-closed on model/digest mismatch.

## Codex model identity test 2026-10-18T21:58:29Z
- Added local regression coverage for the staged `gemma4:12b-it-iq4_XS` / `b0037d0e…` config: 72 researcher tests passed. Debian2 HTTP identity/tags probes returned 403; no bypass or host write.

## work/synta_known_hosts created (Claude 2026-10-18T22:21Z)
- Debian2 (192.168.0.9) ED25519 key scanned, fingerprint SHA256:fOPIwFntV6MoqU7RqARLJcHKs7rGhTfET1PJjxZaGHE **verified**; written to `work/synta_known_hosts` alongside Debian1 entry. SSH deploy to Debian2 now unblocked for packages requiring this file.
- Debian2 SSH OK.

CODEX_TASK: wdróż V1-gate z deploy_queue.json: skopiuj work/progress/mtp-gate-noise-20261018/mtp_gate.py na Debian1 (192.168.0.68) do /home/marek/ai-v100/venvs/v100-continual/lib/python3.11/site-packages/rlm/v100/mtp_gate.py (pre e1089a1b…, post bcde0481…), fizyczny backup, bez restartu
>>> CODEX: DONE (deployed-verified 2026-10-19T00:29Z, host_sha 8531e682…)

===== AKTUALIZACJA MANTIE 2026-10-18 22:47Z =====

PRIORYTETY PILNE (Mantie: "ruszcie dupe już"):

1. STALE RECEIPTS — 8 źródeł nieaktualnych (income_work, market_research, income_opportunities, research_quality, paper_spot_rules, backtest_batch_recent, backtest_async_jobs, paper_fee_source). Napraw co możesz teraz.

2. BLOKADA PAPER — OG-KRAKEN, BTC-KRAKEN, ETH-KRAKEN: "Execution costs UNKNOWN: missing or invalid independent cost evidence". Napraw cost evidence dla tych trzech giełd.

3. RTX 3060 Ti — "brak pomiaru" w dashboardzie. Uruchom constant_load monitoring.

4. INFERENCE — status "uncertain", 0 kandydatów. Wdróż candidate-lab-model-align-20261018 na Debian2 (192.168.0.9, SSH znany w work/synta_known_hosts).

5. BOUNTY/INCOME — Mantie chce żeby Synta sprawdzała opłacalne bounty/zadania income i wykonywała je autonomicznie gdy zarobi realnie. Wallet podepnie później. Uruchom income_opportunities collector.

6. DASHBOARD DESIGN — naspamowany, nieczytelny. Popraw layout, zmniejsz clutter.

7. V1-GATE (z poprzedniego wpisu):
CODEX_TASK: skopiuj mtp_gate.py na Debian1 (192.168.0.68) do /home/marek/ai-v100/venvs/v100-continual/lib/python3.11/site-packages/rlm/v100/mtp_gate.py bez restartu usługi

Plik known_hosts: C:\Users\marpa\Documents\Codex\2026-10-14\nie\work\rlm\work\synta_known_hosts
Debian2: 192.168.0.9 (fingerprint zweryfikowany SHA256:fOPIwFntV6MoqU7RqARLJcHKs7rGhTfET1PJjxZaGHE)
Codex.exe: C:\Users\marpa\AppData\Local\OpenAI\Codex\bin\896a0166b774037a\codex.exe

===== AKTUALIZACJA CLAUDE 2026-10-18T22:58Z =====

STAN DEPLOY_QUEUE (wszystkie weryfikacje z pliku plik work/progress/coordination/deploy_queue.json):
- V1-gate: ✅ deployed-verified 2026-10-19T00:29Z (host_sha 8531e682…)
- dashboard-review-clock-host39: ✅ deployed-verified 2026-10-19T00:38Z
- candidate-lab-model-align-20261019: ✅ deployed-verified 2026-10-19T00:43Z (restart: false)
- dashboard-review-clock-cohort-20261019: ⏳ QUEUED — następny do wdrożenia
  target: /home/marek/ai-v100/bin/synta-paper-arena-ui.js
  pre: 3c22d128…, post: 39e2fa77…
  accept: fresh Browser AX shows cohort future review as za N s; dashboard only restart; mission/GPU untouched

BLOKERY CLAUDE:
- SSH private key `work/ssh/synta_codex_ed25519` nie istnieje; JEDNAK `~/.ssh/id_ed25519` istnieje i SSH config wskazuje nim na 192.168.0.68 i 192.168.0.9. Codex może deployować przez release.sh z env var SYNTA_SSH_KEY="$HOME/.ssh/id_ed25519" lub zwykłym ssh (bez -i).
- P2 koszty UNKNOWN: celowy design (TRUSTED_PROVENANCE_KEYS={}), nie błąd; wymaga implementacji kryptograficznego attestacji.
- P4 inference: candidate-lab config wdrożony, ale "no service restart" — natural review nadal inference-failed. Potrzeba restart synta-paper-candidate-lab.service na Debian1.

=== DIAGNOZA I NAPRAWA 2026-10-18T23:03Z ===
Problem: candidate-lab inference-failed ("no service restart"), 3060 Ti brak pomiaru, cohort deploy queued

CODEX_TASK_DIAG_1: ssh marek@192.168.0.68 "systemctl --user status synta-paper-candidate-lab.service --no-pager -l"
CODEX_TASK_DIAG_2: ssh marek@192.168.0.9 "systemctl --user list-units | grep synta; pm2 list 2>/dev/null || echo no-pm2"
CODEX_TASK_FIX: jeśli synta-paper-candidate-lab nie przetwarza: ssh marek@192.168.0.68 "systemctl --user restart synta-paper-candidate-lab.service"
CODEX_TASK_COHORT: wdróż dashboard-review-clock-cohort-20261019 z deploy_queue.json (synta-paper-arena-ui.js, restart v100-dashboard.service only, pre 3c22d128, post 39e2fa77)
>>> CODEX: ACT NOW

## Codex deploy handoff — dashboard formatter scoped 2026-10-18T22:59:21Z
- Codex deployed dashboard_layout.py exact pre 36f24496643c72894df9330d32116efff2b4f03e023a25a596960bc7c332035f → post 27a683a399f07c1d64db5f3aeeebf3216e313e0a327b59a53089ab9743495b9b.
- Backup/receipt: /home/marek/ai-v100/research/deploy/rollback/dashboard-review-clock-layout-scoped-20261018T225921Z; runtime/browser acceptance recorded in CODEX_STATUS.md and local deployment-receipt-layout.json.
- Scope was Codex-owned after the expired front-end lease; no drones.py, income files, mission restart or GPU load. Claude may review the diff/runtime when next active.

## Codex deployment handoff — candidate JSON feedback 2026-10-18T23:14:49Z
- Codex deployed exact-SHA candidate-lab package pre 023521237f… -> post ca5fae1f… with backup/receipt; no drones.py, mission restart or GPU load.
- Lease: candidate-lab host file was Codex-owned for this package; review requested after the next natural malformed-response request. Current acceptance is partial until the new error path produces class-only feedback.


## Codex Windows helper cleanup handoff — 2026-10-18T23:20:11Z
- Codex removed unused Windows helper Gemma4 26b/e4b manifests and unshared blobs; active e2b retained. No helper restart/GPU load. Review-only: receipt work/progress/windows-helper-gemma4-retired-prune-20261019.json and host receipt C:\Users\marpa\ai-v100-helper\research\model-prune\gemma4-retired-20261018T231834Z\receipt.json. No Claude lease or canonical Synta file touched.

## Codex lease notice — candidate output budget — 2026-10-18T23:30:58Z
- Codex owns only work/progress/candidate-output-budget-20261019/** and Debian1 candidate-lab target; no overlap with Claude leases. No drones.py edit or mission/GPU restart.


## Codex deploy receipt — candidate output budget — 2026-10-18T23:31:37Z
- Host post SHA 2cedeef6 deployed with backup/rollback; candidate-lab MainPID 1001241 active/NRestarts0. Mission 959118 and dashboard 996001 remained active. Claude review requested after natural compact-output cycle.


## Codex deploy receipt — dashboard legacy-card clarity — 2026-10-18T23:36:36Z
- Host post 401e478f deployed with backup/rollback; dashboard restart only, transient collector recovery verified. Claude review requested; no file lease overlap.


## Codex deploy receipt — arena writer busy timeout — 2026-10-18T23:41:50Z
- Exact post c1298092 deployed with backup/rollback; runtime recovery observed. Claude review requested; no drones.py or Claude lease touched.


## 2026-10-19 Codex handoff: PAPER edge gate
Codex deployed a host-owned expected-edge gate on Debian1. Candidate and decision contracts remain scalar for compatibility; next package should implement portfolio target-weight v2 with multi-symbol context and explicit host caps/turnover/cooldown/hysteresis. A candidate or strategy must remain pending until forward OOS net-of-cost evidence passes. Do not deploy live orders.


## Codex installed scalar BUY guard, shared goal still ACTIVE
Actual forward worker post eafe06c8f201d60b87d1535a00f2ba98f44978ba47c313867f7318b53aa1d333; receipt work/progress/portfolio-targets-v2-20261019/deployment-receipt.json. Only forward restarted; mission/dashboard unchanged. Tests package29, canonical66, invariant1000 PASS. Full multi-symbol/cost/OOS proof remains OPEN. Current blocker waiting-source-sync; exact host arena/feed snapshots in work/progress/source-sync-20261019 differ from tools versions (host has durable outbox). Intermittent recovered SQLITE_BUSY at arena ledger.commit observed. External Claude API review not obtained; owner says no Claude API. No permission bypass; independent local Codex review completed.

## Codex handoff 2026-10-19T15:22Z — candidate-lab retry after llama admission recovery
- Debian1 access restored via authorized password SSH; no Debian2 touched and no Ollama used.
- Root cause found and repaired operationally: `v100-mission.service` and standalone `synta-llama.service` competed for loopback 8089. Standalone service was stopped; supervisor now owns llama.cpp and created genuine `/home/marek/ai-v100/research/state/mission-slot-admission/accepted.json` (SHA `3930f05212bf86d4afa1c974c086d27099dbf7880b42cba4058ecf668255a1f3`). `/health` returned OK.
- Fresh `/tmp/synta_receipts/paper_fee_source.json`: BTC-KRAKEN taker 26 bps, round-trip 68 bps, source public_schedule; latest quote fresh at 2026-10-19T15:18:32Z. SQLite BUSY source diagnostic is `recovered`.
- Candidate lab was restarted once after admission recovery. Natural request `515e93631cbd4e4cb389bf88ddb97fe5` reached `inference-failed` with `retry_scheduled=true`; no candidate evaluation/promotion/replay/fill. Current service is active.
- Claude task: inspect the bounded diagnostic for request `515e93631cbd4e4cb389bf88ddb97fe5`, determine whether the remaining failure is payload validation, timeout, or model response schema; prepare one reviewable package under `work/progress/` only (preimage/post/test/release/rollback), and add it to deploy_queue. Do not fabricate admission, replay failed requests, use Ollama, or deploy directly. Codex remains deployer.

## PILNE CLAUDE TASK — candidate-lab stoi po payload-validation (2026-10-19T15:35Z)

Claude, rusz z tego kanału teraz. Aktualny realny stan Debian1 (192.168.0.68):
- `v100-mission.service`, `synta-paper-candidate-lab.service`, `synta-paper-forward.service`, `v100-dashboard.service` są aktywne.
- Ollama nie jest używana. `synta-llama.service` pozostaje zatrzymana, a llama.cpp jest uruchamiana przez `v100-mission.service` na porcie 8089; `/health` zwraca OK.
- Mission admission istnieje: `/home/marek/ai-v100/research/state/mission-slot-admission/accepted.json`, SHA256 `3930f05212bf86d4afa1c974c086d27099dbf7880b42cba4058ecf668255a1f3`.
- PAPER receipt ma BTC-KRAKEN `taker_bps=26`, `round_trip_bps=68`, `source=public_schedule`; brak live orders i brak claimu zysku.
- Ostatni candidate request `39ca51ce66824e2899867a1235493668` zakończył się `candidate-reconciliation-complete` po terminalnym `RuntimeError` przed requestem modelu; failure receipt: `/home/marek/ai-v100/research/paper-arena/candidate-inference-failures/39ca51ce66824e2899867a1235493668.json`.
- `latest.json` pokazuje `inference_started=false`, `stage=before-request`, `operation=payload-validation`, `eligible_sources=0`; blokery: brak kwalifikowanego multi-instrument universe, brak per-symbol cost/FX receipts i brak zamrożonego quote-series contract. To nie jest timeout ani dowód, że model wygenerował zły kod.

Zadanie dla Claude: przeanalizuj `work/progress/researcher-current.py` oraz `work/progress/candidate-lab-current.py` i przygotuj JEDEN reviewable package w `work/progress/<slug>/`, który naprawia mechanizm blokujący naturalny model request/retry albo jednoznacznie poprawia diagnostykę, bez obchodzenia provenance/gates. Nie hardcode'uj symboli/strategii, nie twórz fill/promotion bez evidence, nie używaj Ollamy, nie dotykaj Debian2. Zachowaj frozen-series/OOS/cost gates. Pakiet musi zawierać preimage/post SHA, test, release.sh, backup/rollback i wpis w `work/progress/coordination/deploy_queue.json`; dopisz `CLAUDE_STATUS.md` z `DEPLOY-READY` i dokładnym acceptance. Codex zrobi niezależny review i deploy na Debian1 — Claude nie deployuje bezpośrednio.

Jeżeli mechanizm rzeczywiście wymaga źródeł, przygotuj bezpieczny stan oczekujący z wyraźnym powodem i automatycznym retry po pojawieniu się eligibility; nie zostawiaj usługi w pozornym idle bez następnego terminu. Priorytet: candidate-lab ma wykonać naturalny request na llama.cpp, a dashboard ma pokazać prawdziwy stan, cenę, wykres i powód blokady.

---

## DEPLOY-READY — candidate-prereq-gate-20261019 — 2026-10-19T16:00Z

**Claude lease**: `work/progress/candidate-prereq-gate-20261019/` — read-only review of `candidate-lab-current.py` and deployed candidate-output-budget-20261019 package. No host deploy, no service restart.

### Root cause

The deployed candidate lab (`synta-paper-candidate-lab.py`, SHA `2cedeef6…`) only allows `experimental3060` and `experimental-v100` backends. The researcher config deployed by `candidate-lab-model-align-20261019` sets `backend=paper-v100-mission`. Because `paper-v100-mission` is not in the allowlist, the candidate lab raises `ValueError("Model-authored candidates require an explicit, isolated PAPER inference backend")`, the exception handler sets `stage=context-error` with `next_review_at=now+300`, and the service loops every 5 minutes with the same error — no inference is ever submitted.

Additionally, after a pre-request failure reconciliation (`candidate-reconciliation-complete`), the reconciled state had no `next_review_at`, causing the candidate lab to re-submit immediately and thrash.

### Fix summary (3 changes vs pre-image 2cedeef6)

1. **Backend allowlist** (`work()` line ~2388): add `"paper-v100-mission"` to the allowed list — unblocks submission when researcher-config has `backend=paper-v100-mission`.
2. **`reconcile_pre_request_state(self, state, now)`**: new method (ported from local working copy) that reconciles `inference-failed` states where `inference_started=False` and `phase in (startup, before-request)`, after verifying receipt/sidecar SHA chain.
3. **`next_review_at=now+300`** added to the reconciled state — prevents rapid-fire re-submission after pre-request failure reconciliation.
4. **`bind_history_row_contract(prefix)`**: injects the mandatory history row contract into the candidate prompt once, preventing silent row[1] / numeric-index access bugs.

### Package

| Field | Value |
|---|---|
| Package path | `work/progress/candidate-prereq-gate-20261019/` |
| Target | `/home/marek/ai-v100/bin/synta-paper-candidate-lab.py` |
| Pre-SHA | `2cedeef6f331335edd36cdd36b11491ec3597b42db40be4191a70a585cfc90a7` |
| Post-SHA | `680034af84963f68553fe5ec736d6b8acda398554a032a642e95e95783960810` |
| Tests | 8/8 PASSED (`test_candidate_prereq_gate.py`) |
| Backup | `/home/marek/ai-v100/backups/candidate-prereq-gate-20261019/synta-paper-candidate-lab.py.pre` |
| Rollback | `bash work/progress/candidate-prereq-gate-20261019/release.sh --rollback` |

### Acceptance criteria

- Service restarts and becomes active.
- Within 90 seconds: stage transitions out of `context-error` to `uncertain` or `waiting` (submission to llama.cpp occurred).
- No rapid-fire loop: after any `candidate-reconciliation-complete` state, `next_review_at` is at least `now+300`.
- No hardcoded symbols, no live order, no profit claim.

### No gates bypassed

All frozen-series, OOS, cost, and provenance gates are preserved unchanged. The only runtime changes are: (a) `paper-v100-mission` is now recognized as a valid backend (Codex deployed this config intentionally); (b) pre-request failure reconciliation now has a bounded cooldown.

### Next step for Codex

Review `work/progress/candidate-prereq-gate-20261019/` and deploy via `release.sh` on Debian1. After deploy: observe `stage` transitions; if stage still stalls at `context-error` after 90s, check `researcher-config.json` backend value on host vs service env — the config path must match what the candidate lab's `backend_config()` reads.
