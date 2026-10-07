# Forward paper research, v100.14

v100.14 skraca katalog narzędzi pomocnika CPU, poprawia jego limit czasu oraz
wykonuje jego zadania kolejno przy pojedynczym slocie. Nie zmienia opłat,
instrumentów, portfeli ani zasad składania zleceń paper.

v100.13 dodaje `mission-start`, `mission-status` i `mission-stop`: background
research celu, trwałe źródła i streszczenia oraz automatyczną kalibrację przed
ciągłym uczeniem A/B. Szczegóły są w [CONTINUAL_LEARNING.md](CONTINUAL_LEARNING.md).
Z pustą konfiguracją opłat/feedów prowadzi research i raportuje hold; nie wykonuje
transakcji. Nie uznaje zysku paper ani modelowej opinii za etykietę do treningu.

v100.12 dodaje offline `prepare-challenge` oraz osobny test użycia narzędzi.
Ćwiczenia kosztowe mają fikcyjne wartości opłat, funding i slippage w groszach;
nie konfigurują realnych taryf ani instrumentów. Odczyt źródeł respektuje cutoff
publikacji. Testy nie składają zleceń i nie generują wyników strategii finansowej.

v100.11 dodaje osobny profil `prepare-thinking` opisany w
[CONTINUAL_LEARNING.md](CONTINUAL_LEARNING.md). Rodzice A/B dziedziczą jego sampling
i budżet rozumowania; CPU helper zachowuje osobny szybki profil. Przed uczeniem
potrzebny jest pełny baseline w nowych warunkach. Poprawna arytmetyka w czterech
przypadkach nie jest dowodem zyskownej strategii. Bramki opłat, świeżości danych,
weryfikacji ćwiczeń i ochrony poprzednich wersji nadal obowiązują.

Cel użytkownika: A/B mają szukać możliwie wysokiego, powtarzalnego zysku netto,
porównując research sportowy, crypto oraz akcje, także izolowaną dźwignię.
Wyłącznie paper, bez wpłat, płatnych źródeł i prawdziwych zleceń. Brak strat
nie jest gwarantowalny. Cash/hold jest pełnoprawnym wynikiem decyzji.

## Co działa w tym wydaniu

- `paper-init`: osobne, równe portfele A/B, domyślnie 10 000 PLN każdy.
  Nie zmienia modeli, środowisk, starej pamięci ani tekstowego celu treningu.
- `paper-configure FILE`: rejestruje wersjonowane reguły instrumentów i opłaty
  zweryfikowane przez operatora. Modele nie mają narzędzia do ich zatwierdzania.
- `paper-ingest FILE`: import JSONL niezależnych, timestampowanych kwotowań,
  materiałów źródłowych, wyników zakładów i corporate actions.
- `paper-source URL`: archiwizuje publiczną dokumentację przez istniejący reader.
- `paper-poll --crypto`: darmowe Coinbase Exchange L1 dla zarejestrowanego spotu.
  NBP służy do referencyjnego przeliczenia USD/EUR/GBP na PLN, nie jako wykonywalny
  kurs brokera. Koszt rzeczywistego przewalutowania musi być w tabeli opłat.
- `paper-poll --cik NUMBER --sec-contact EMAIL`: darmowe metadane najnowszych
  10-Q/10-K/8-K z SEC. EMAIL jest prawdziwym kontaktem operatora zgodnie z polityką
  dostępu SEC. Zawartość raportu wymaga osobnego odczytu, nie jest wymyślana z metadanych.
- `paper-round`: A/B prowadzą research i proponują po jednej decyzji. Serwer
  lokalnego modelu musi już działać. W tym poleceniu CPU helper też musi działać.
- `paper-loop`: powtarza research, opcjonalnie polling crypto/SEC, i lokalne raporty.
  `--cycles 0` oznacza ciągłą pracę, `--interval` domyślnie 300 sekund po zakończeniu
  rundy. To polling/research, nie HFT ani gwarancja reakcji co dokładnie pięć minut.
- `paper-report`: HTML z SVG, Markdown, JSON i CSV transakcji w `research/paper/reports/`.

Instalacja nie uruchamia pętli, researchu ani żadnego serwera. Najpierw wybieramy
platformy, dokumentujemy wszystkie koszty i źródła wykonania. Brak danych lub
znanych opłat blokuje handel; nie tworzymy domyślnych zerowych prowizji.

## R&D i inteligentny wybór

Domyślnie są dwie rundy researchu na gałąź, do sześciu po `--research-rounds`.
Researcher szuka dowodów, krytyk próbuje obalić hipotezę i sprawdzić koszty.
Mogą czytać publiczne social media, strony spółek, kwartalne raporty i newsy,
korzystając z istniejących narzędzi do publicznych źródeł i wyłącznie bezpłatnych
konsultacji. Reader obsługuje HTML/text/JSON z limitem 256 KiB; nie obsługuje
PDF, treści wymagających logowania ani nie omija blokad stron.
To ograniczony budżet R&D, nie obietnica kompletnego internetu lub pełnego raportu.

Testerzy mają dodatkowo `paper_status`, `paper_observed_results` oraz
`paper_test_position`: niezależną kalkulację kosztów i syntetyczny szok ceny.
Ten test nie tworzy zlecenia, nie przewiduje wyniku i nie jest zatwierdzonym labelem.
Model główny może odrzucić wszystkie ustalenia. Publiczne decyzje i research
trafiają do wspólnej pamięci A/B, z oddzielnymi dziennikami activity.

Z `--researcher-profile` używany jest istniejący mały model CPU; dwie role mogą
pracować współbieżnie. `paper-loop` uruchamia i kończy tylko własny CPU serwer.
Profil musi mieć `gpu_layers=0`. Bez helpera role korzystają kolejno z tego samego
serwera GPU, co oszczędza VRAM, lecz jest wolniejsze. Główna Gemma nie jest
duplikowana przez ten moduł. A/B w tym wydaniu to różne portfele i role tego samego
modelu wskazanego profilem, nie dwie nowe niezależnie wytrenowane Gemmy.
Loop tworzy własny snapshot profilu helpera z kontekstem 8192, pozostawiając
oryginalny profil bez zmian. Jednocześnie może działać tylko jedna pętla paper
w danym root; lease pliku zapobiega dublowaniu zleceń i raportów.

Speculative decoding jest dziedziczone z faktycznie uruchomionego serwera.
Moduł paper nie włącza go sam i nie dowodzi przyspieszenia. Najpierw sweep
MTP2/4/8/16 i osobna bramka jakości z v100.8. Modelowy research, prompt length
i czas oczekiwania na dane też wpływają na szybkość całego cyklu.

## Bez patrzenia do przodu

Kontroler nadaje `observed_at`, model nie może go dostarczyć ani zmienić.
`available_at` z feedu nie może być późniejsze od zegara kontrolera. Kwotowania
muszą mieć rosnący czas dostępności i być świeże, domyślnie maksymalnie 300 sekund.
Decyzja jest związana z konkretną sekwencją wiedzy; po jej zmianie wymaga nowej analizy.
Wykonanie możliwe jest dopiero na późniejszym kwotowaniu niezależnego feedu,
a nie na cenie wybranej przez model. Modele nie mają narzędzia do importu cen
ani rozliczania zakładów. Zlecenia wygasają po 300 sekundach; można je anulować.
Wynik meczu może rozliczyć tylko niezależny observation po rozpoczęciu wydarzenia.

SQLite zachowuje hash chain obserwacji, decyzji i zmian konfiguracji. UPDATE/DELETE
historii są blokowane, a stan portfela jest sprawdzany względem dziennika.
Pliki źródłowe i importy są snapshotowane. Hostowy ledger, opłaty, risk/evaluator,
prompty i feedy są wyłączone z samodzielnych edycji kodu przez modele.
Operator z pełnym dostępem do hosta nie jest kryptograficznie odcięty od własnych
plików; zewnętrznych, ręcznie importowanych timestampów program nie potrafi
certyfikować. To granica zaufania do operatora/feedu, nie gwarancja prawdy źródeł.

Social post odczytany dziś nie staje się danymi dostępnymi wczoraj. Koniec kwartału
nie jest datą publikacji wyników, a dzisiejsze restatementy nie służą do oceny
starych decyzji. Wiedza historyczna w pretrained LLM może zanieczyścić backtest.
Ten moduł odrzuca stare kwotowania do forward paper i nie przedstawia takiego
backtestu jako dowodu zdolności zarabiania. Hipotezy trzeba badać na nowych danych.

## Koszty i reguły wykonania

Konfiguracja ma listy `fee_profiles`, `instruments`, opcjonalnie `fee_updates`.
Nowa tabela opłat dostaje nowe ID; przypięcie do instrumentu zmienia się tylko
przez operatora i pozostawia zdarzenie w dzienniku. Stare transakcje się nie zmieniają.
Przy otwartych pozycjach nie zmieniamy w tym wydaniu stałego modelu finansowania.
Wzrost minimalnej opłaty zamknięcia jest blokowany przy otwartej pozycji,
bo istniejąca rezerwa nie pokrywałaby nowego minimum. Najpierw zamknij pozycję.

Każdy fee profile wymaga:

- `id`, `currency`, `operator_verified=true`, `source_url`, `source_sha256`,
  `available_at`, `valid_until`.
- `entry_commission_bps`, `entry_minimum_commission`, `entry_venue_fee_bps`,
`entry_fx_bps` i osobnych odpowiedników z prefiksem `exit_`.
- `slippage_bps`, `financing_annual_bps`, `liquidation_bps`, `stake_tax_bps`,
  `winnings_tax_bps`, `winnings_tax_threshold`, `winnings_commission_bps`.
- `winnings_tax_basis`: `whole-payout` albo `excess-only`;
- `winnings_tax_base`: `gross-payout` albo `after-commission`;
  `void_refunds_stake_tax` i `void_refunds_entry_fees`: jawne wartości bool.

Wartości liczbowe są nieujemne i jawne, nigdy null. Jednostką stóp jest bps
(100 bps = 1%); minima i progi są w walucie portfela. Opłaty wejścia/wyjścia mogą
być asymetryczne, np. podatki transakcyjne lub opłaty giełdowe tylko po jednej stronie.
Spread jest w bid/ask, dodatkowy slippage pogarsza wykonanie. Każdy partial fill
ma własne minimum prowizji, co może konserwatywnie zawyżać rzeczywisty koszt.
To model market/taker orders z L1, bez fikcyjnych maker rebates lub limit fills.
Rezerwa minimalnej opłaty zamknięcia pozostaje w alokacji pozycji.

Finansowanie jest uproszczonym stałym annual rate liczonym na entry notional.
Zmienne stopy/basis, tiered margin i niestandardowe instrumenty wymagają nowego
zweryfikowanego adaptera. Perpy wymagają jawnych, zrealizowanych `funding_events`
z czasem, `rate_bps` i faktycznym `mark`, a nie prognozy następnego fundingu.
Quote musi obejmować cały interval od poprzedniej obserwacji, wraz z `mark_low`,
`mark_high` i `mark`; brak tych danych blokuje ocenę likwidacji.

Każdy instrument wymaga: `symbol`, `market` (`crypto`, `equities`, `sports`),
`product` (`spot`, `isolated-linear`, `bet`), `cluster`, `feed_id`, `quantity_step`,
`fee_profile`, `price_basis=raw-unadjusted`, `operator_verified` i źródła jak wyżej.
Bet wymaga `starts_at`. Isolated-linear wymaga udokumentowanego `loss_cap=position`,
`auto_add_margin=false` i stałego `maintenance_fraction`. Nie zakładamy, że
zwykły margin akcyjny lub ochrona ujemnego salda całego konta daje taki cap pozycji.

Przy likwidacji symulator konserwatywnie traci całą alokację tej pozycji,
bez dobierania z pozostałego portfela. Loguje koszty i adjustment związany z tym capem.
Nie jest to pełna replika procesu konkretnego brokera, ADL ani liquidation tiers.
Nie wolno przenosić tej własności na realny produkt bez sprawdzenia jego umowy.

Akcje używają surowych cen i jawnych splitów. Dywidendy są naliczane według
historycznego stanu udziałów na `entitlement_at`, z `withholding_bps`, dopiero przy
obserwowanym payment (`effective_at`). Należności pomiędzy ex-date a payment
nie są jeszcze markowane jako receivable, więc equity w tym okresie może być
konserwatywnie zaniżone. Split i payment należy podać między odpowiednimi surowymi
kwotowaniami, bez importu wstecz. Sports obsługuje pojedyncze fixed-odds bets;
netting prowizji całego rynku zakładów exchange, cashout i zakłady lay są poza zakresem.
Podatek dochodowy zależny od kraju/osoby/rachunku nie jest wyliczany bez tych reguł.
Raport jawnie mówi `net_pnl_before_personal_tax`, nie obiecuje zysku po wszystkich podatkach.

## Dywersyfikacja, raporty i uczenie

Domyślny budżet eksperymentalny: do 5% equity na pojedynczą alokację, 15% na
ustaloną grupę skorelowanych instrumentów, 20% na rynek, 30% wszystkich alokacji;
dźwignia maksymalnie 3x. Pending orders także zużywają budżet. Model nie może
ominąć korelacji przez przemianowanie grupy. Po obserwowanym drawdown 10% nowe
otwarcia są pauzowane. Już otwarte pozycje mogą dalej stracić, szczególnie w luce.
To ustawienia testowe, nie zalecane parametry realnego portfela ani limit gwarantowanej straty.

Gdy lokalny loop działa, raport dzienny powstaje po 20:00 Europe/Warsaw,
tygodniowy w niedzielę po 20:00. Oceniają ostatnie 24 godziny / siedem dni,
z pełną historią jako kontekstem. Harmonogram zachowuje już wygenerowane okresy
po restarcie. Bez działającego loopu nie ma automatycznych raportów ani dostawy wiadomości.
Raporty pozostają plikami na Debianie. Wykresy obejmują equity A/B oraz ceny
z zaznaczonymi wykonaniami kupna/sprzedaży; CSV zawiera również zakłady i settlementy.

Raport pokazuje marked P&L z otwartymi stratami, realized P&L, koszty, próby/fills,
zaobserwowany drawdown, stan cash/pozycji i SHA audytu. Nie promuje strategii do real
money. Jednorazowy wynik, mała próba, stare fee lub stare kwotowania nie dowodzą
powtarzalności. Regularne nowe obserwacje, niezależne okresy i stress tests są
potrzebne przed rozważeniem osobnego wdrożenia z realnym kapitałem.

Research i wyniki portfela są pamięcią/obserwacjami, nie automatycznymi "dobrymi"
labelami treningu. Istniejące formalne ćwiczenia kosztowe od testerów mogą przejść
referencyjny verifier, akceptację rodzica i dotychczasowy pipeline LoRA z replay/KL.
Sam `paper-loop` nie wykonuje optimizer step i nie zmienia wag. Automatyczne
uczenie polityki finansowej z reward, czasowy holdout i promocja adapterów według
wyników rynku wymagają osobnego walidowanego modułu; nie dodajemy fałszywych etykiet
"wygrana = poprawny wniosek" do stale uczącej się Gemmy.

## Połączenie z uczeniem wag, v100.10

`learn-loop --paper-config FILE` łączy powyższy research z istniejącą pętlą
zweryfikowanych danych i treningu A/B. Nadal wymaga poolu, stałej suite,
raportów baseline i profilu CPU researchera. `paper-learning-status` sprawdza
ścieżki; `--datasets DIR` pokazuje ograniczoną listę plików bez czytania danych.
Nie uruchamia modeli ani treningu. Nie powtarzaj `paper-init` po pierwszej instalacji.

Konfiguracja JSON musi zawierać wszystkie pola:
`schema="v100-paper-learning-v1"`, `objective` (cel użytkownika), `crypto` (bool),
`ciks` (lista do dziesięciu numerów SEC), `sec_contact` (kontakt operatora lub
pusty string bez SEC), `observer_interval` (30-300 sekund), `research_rounds`
(1-6) i `other_income_rnd` (bool). Zmiana konfiguracji podczas pracy blokuje
kontynuowanie; nowy cel wymaga osobnej próby. Snapshot jest zachowywany według SHA.

Z `other_income_rnd=true` researcher sam wybiera badanie rynku, innego legalnego
dochodu bez wpłat lub usprawnienia uczenia/narzędzi. Krytyk bada koszty, czas,
legalność i powtarzalność. Dwie role pozostają w dotychczasowym budżecie rund,
bez dodawania drugiej kopii Gemmy. Małe podmodele mają istniejący budżet pilota
CPU i bramki bubblewrap; to nie automatyczna przebudowa/wymiana głównej Gemmy.
Rzeczywiste działania sprzedażowe i dochody spoza rynku nie mają jeszcze modułu
wykonania ani niezależnego rozliczenia. Są hipotezami R&D, nie nowym kontem z gotówką.

Podczas serwowania bieżącej wersji research może dostarczać formalne ćwiczenia,
których etykiety oblicza referencyjny verifier. Dopiero po akceptacji rodzica
trafiają do poolu, replay i treningu A/B. Brak nowych zaakceptowanych danych
oznacza brak kolejnego treningu. Aktywacja kolejnej wersji wymaga dotychczasowych
skończonych testów nowych i dawnych umiejętności. Poprzednie wagi/checkpointy
pozostają, a w dzienniku faz jest wersja modelu i SHA profilu.

Jedna niezależna nić obserwacji zbiera zatwierdzone dane spot/SEC i raporty także
w czasie treningu; nie ładuje modelu ani kontekstu CUDA. Jej błąd źródła/audytu
blokuje dalsze fazy na najbliższym sprawdzeniu. Przerwa sieciowa jest logowana,
nie uzupełniana fikcyjną ceną. Sam polling nie daje danych tickowych ani HFT.
Inferencja GPU i trening głównej Gemmy są kolejnymi fazami; własny serwer
inferencji jest zatrzymywany przed treningiem. CPU researcherzy mogą pracować
podczas treningu przez istniejący runner A/B, z kontrolą RAM. V100 nie ma MIG:
moduł nie obiecuje twardej izolacji dowolnych części VRAM.

Nie uruchamiaj jednocześnie `paper-loop` i tego samego zintegrowanego loopu:
wspólny lease blokuje drugiego właściciela. Zysk paper nadal nie jest labelem,
miarą treningowego loss ani kryterium automatycznej promocji finansowej strategii.
Ten bridge uczy tylko istniejących weryfikowalnych ćwiczeń formalnych, nie całego
researchu ani prognoz. Nie dowodzi wzrostu inteligencji, zyskowności lub absolutnego
braku forgetting; potwierdzenie wymaga niezależnej oceny na nowych zadaniach/danych.

## Źródła protokołu

- Coinbase: https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-book
- NBP: https://api.nbp.pl/
- SEC: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- Isolated/cross oraz funding: https://www.bybit.com/en/contract-rules/
  (opis mechanizmu, nie preset aktualnych fees ani rekomendacja platformy).
- Ochrona CFD odnosi się do środków konta, nie automatycznie pojedynczej pozycji:
  https://www.esma.europa.eu/publications-data/questions-answers/1981
- Prowizja exchange od net winnings rynku:
  https://support.betfair.com/app/answers/detail/a_id/413/

Testy są deterministyczne, ze sztucznymi opłatami i zastępczym transportem.
Nie wykonano paper tradingu na Twoim Debianie ani badań rzeczywistej przewagi
A/B, nie otwarto kont i nie wysłano żadnego prawdziwego zlecenia.
Walidacja v100.9: 443 testy zaliczone, 63 pominięte; ruff, formatter i hooki
pre-commit zaliczone. Nie jest to pomiar wydajności lub zyskowności na V100.
Walidacja v100.10: 460 testów zaliczonych, 63 pominięte; lint/formatter/hooki
zaliczone, zmienione moduły bridge i paper przeszły ty bez ignorowania błędów.
Integracja treningu jest testowana na zastępczym runnerze; istniejący test małej
LoRA na CPU potwierdza zmianę wag od zweryfikowanych formalnych przykładów.
Nie uruchomiono tej integracji z Gemmą, feedami i podmodelami na Twoim Debianie.
