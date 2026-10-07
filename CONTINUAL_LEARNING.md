# V100 continual learning, v100.16

## Budżet thinking i większy kontekst, v100.16

Logi użytkownika potwierdziły ucięte odpowiedzi Gemmy w researchu i planowaniu:
thinking było włączone, ale odpowiedź miała limit 2048 tokenów. Tylko wywołania
R&D, recenzji, planowania eksperymentów i decyzji paper mogą ponowić tę samą
prośbę po `finish_reason=length`, podwajając budżet do 8192 tokenów. Maksymalnie
trzy próby, dodatkowo ograniczone wolnym miejscem w kontekście. Nie dopisujemy
częściowych odpowiedzi i nie wykonujemy uciętych akcji. Sampling, schema i
thinking pozostają takie same. Zwykła ocena jakości nadal nie ponawia próśb.
Pomocnicy bez thinking zachowują dotychczasowe limity. Zdarzenia
`research-output-retry` pokazują model, próby i koszt tokenów w timeline.

`mission-start --max-context 131072 --flash-attention on` próbuje 128k,
następnie 64k/32k/16k/8k, przy zachowaniu minimum 4 GiB wolnego VRAM po
załadowaniu. Ustawienia są snapshotem nowej misji, nie nadpisują profilu
wejściowego. Jawne `on` wymaga obsługi w lokalnym buildzie i nie jest po cichu
wyłączane przy błędzie. KV cache zachowuje typ z profilu, domyślnie F16.
Cache zostaje zwolniony wraz z serwerem przed treningiem; długość sekwencji
treningowych nie wzrasta automatycznie. Nowy kontekst lub tryb Flash Attention
wymaga nowego baseline, wcześniejszy raport pozostaje zachowany.

Samo uruchomienie i wolny VRAM po załadowaniu nie dowodzą szybkości, szczytowego
zużycia pamięci ani jakości przy pełnym 128k promptu. Wymaga to osobnego testu
na sprzęcie i modelu użytkownika. Dłuższe wejście może wydłużyć prefill; nie
zapełniamy go automatycznie wszystkimi źródłami. Pamięć oryginałów i hierarchiczne
podsumowania nadal pozwalają dobierać potrzebne fragmenty.

Nowy workflow ma osobne środowisko `venvs/v100-continual`, komendę
`bin/v100-continual`, profil `research/v100-continual.toml`, kopię pamięci SQLite
i port 8089. Instalator `install-continual-v100.sh` kopiuje zależności działającego
środowiska treningowego, zachowując Torch 2.6.0 CUDA 12.4. Nie aktualizuje `train`,
`memory-lab`, `v100-lab` ani istniejącego profilu i nie uruchamia treningu/serwera.
To pełne osobne środowisko, nie optymalizacja globalnych sterowników lub CUDA.
Kod v100.14 jest przygotowany do instalacji; sam commit nie aktualizuje serwera
Debian ani nie uruchamia procesu. Nowe zależności opcjonalne `selflab` obejmują
pytest i psutil; trening wymaga istniejącego Torch/Transformers/PEFT.

## Poprawka CPU i widoczny postęp, v100.14

Log użytkownika pokazał, że CPU helper przetworzył około 45% promptu po 84 s,
po czym klient anulował żądanie przy limicie 90 s. Snapshot nowej misji ma limit
900 s na żądanie oraz krótszy katalog narzędzi CPU: pamięć, publiczne źródła,
stan/wyniki paper, koszty pozycji i sprawdzanie kandydata kodu. Rodzic GPU zachowuje
pełny katalog. Jest to poprawka budżetu i objętości promptu, nie zmierzone
przyspieszenie CPU. Oryginalny profil researchera pozostaje bez zmian.

Przy jednym slocie CPU zadania researchera i krytyka są wykonywane kolejno;
drugi klient nie czeka już w kolejce podczas obsługi pierwszego. Profile CPU
z co najmniej dwoma slotami nadal pozwalają na dwa równoległe zadania. Pomocnik
nadal działa na CPU podczas treningu GPU i nie zajmuje pamięci karty.

Ocena drukuje rozpoczęcie przypadku, numer i liczebność oraz czas/wynik po
zakończeniu. Osobny `baseline-32768.progress.json` (lub dla wybranego kontekstu)
zawiera postęp i jest pokazywany przez `mission-status`. Plik progress nie jest
ukończonym raportem jakości i nie może służyć jako baseline.

Nowa misja może wykorzystać ukończony baseline z wcześniejszego katalogu misji.
Wymaga identycznego modelu, binarnego wykonania i ustawień inferencji, tego samego
zestawu i wszystkich jego przypadków. Nie akceptuje częściowej oceny. Pochodzenie
i hash raportu są zapisane w `baseline-reused.json`. Nie wznawia przez to
optymalizatora ani wcześniejszego zwycięzcy A/B. Aktualizacja pakietu nie zmienia
już uruchomionego procesu; nowy snapshot pomocnika wymaga nowej misji. Aby nie
powtarzać oceny, zachowaj ukończony raport przed przełączeniem wersji.

Walidacja v100.14: 524 testy zaliczone, 63 pominięte; lint, format, hooki i
ścisłe sprawdzenie typów zmienionych modułów kontrolera przeszły. Obejmuje
kolejkowanie według liczby slotów, krótszy katalog CPU, licznik oceny i odmowę
ponownego użycia niekompletnego lub niezgodnego baseline'u. Nowego budżetu
researchu na Xeonie nie zmierzono jeszcze na maszynie użytkownika.

## Research celu i uczenie w tle, v100.13

Po przygotowaniu `income-challenge-v1` i CPU researchera uruchom
`--profile research/v100-thinking.json mission-start`. Startuje osobny proces
w tle, który działa po zamknięciu terminala. `mission-status` podaje PID, fazę,
wybrany kontekst, katalog logów i ukończone cykle. `mission-stop` kończy tylko
zweryfikowaną sesję tej misji i jej procesy potomne; zachowuje wyniki i checkpointy.
Nie jest to usługa systemowa: po restarcie maszyny trzeba uruchomić proces ponownie.
Nowe `mission-start` tworzy nowy eksperyment, a nie automatycznie wznawia optymalizator
lub wybiera zwycięzcę z poprzedniego katalogu. Istniejący trening można wznawiać
osobnym mechanizmem kompletnych checkpointów opisanym poniżej.

Misja zapisuje wskazany cel albo domyślny cel legalnego, powtarzalnego dochodu
netto bez wpłat i płatnych API. Archiwizuje początkowe publiczne źródła opłat i
crypto, uruchamia research Gemmy oraz małych pomocników CPU, następnie samodzielnie
wykonuje baseline 81 przypadków. Nie wymaga kolejnego ręcznie uruchamianego testu.
Później powtarza research, propozycje ćwiczeń, niezależną weryfikację, trening A/B,
ocenę oraz przyjęcie lub odrzucenie kandydata. Pierwsza aktualizacja może użyć
gotowego zweryfikowanego curriculum. Kolejne wymagają nowych zweryfikowanych i
zaakceptowanych przykładów; nie trenuje stale na tych samych rekordach dla samego
zapełnienia GPU. Modele wybierają ograniczone hiperparametry i dane, natomiast
kontroler zachowuje źródła referencyjne, kryteria oceny i poprzednie wersje.

Błąd natywnych wywołań narzędzi z v100.12 nie ustalał, czy niepoprawny JSON był
w odpowiedzi końcowej, czy w argumentach narzędzia. Nowa misja używa akcji JSON
z wymuszonym schematem i walidacją nazw/argumentów po stronie kontrolera. Historia
narzędzi trafia jako zwykłe dane, bez zależności od natywnych znaczników Gemmy.
Stare profile i raporty nie są zmieniane. Kontroler jest testowany z atrapą modelu;
rzeczywiste wykonanie Gemmy na V100 potwierdza raport misji, nie testy jednostkowe.

Kontekst roboczy dobierany jest kolejno jako 32768, 16384 albo 8192 tokenów:
profil musi się załadować i pozostawić co najmniej 4 GiB wolnego VRAM. To zapas
po załadowaniu, nie gwarancja dowolnego obciążenia przy pełnym oknie. Faktyczny
tokenizer sprawdza budżet każdego żądania. Nie ma dwóch pełnych modeli GPU
jednocześnie; A/B to kolejne kandydaty i osobne role/portfele. CPU helper ma
8192 kontekstu i osobny budżet wyjścia 768 tokenów.

`research/state/mission-memory.sqlite3` przechowuje pełne odczytane źródła oraz
publiczne wyniki researchu. Hierarchiczne streszczenia grupują po cztery fragmenty,
z budżetem czterech nowych streszczeń na cykl i kontynuacją częściowego drzewa.
R&D otrzymuje wybrane trafienia, a narzędzia mogą odczytać oryginalny fragment.
To selektywny dostęp do dużego archiwum poza VRAM; nie pełna uwaga nad całym
archiwum. Streszczenia mogą pominąć istotny szczegół, dlatego oryginały pozostają.
Ta pamięć nie aktualizuje wag. Wagi adaptera zmieniają wyłącznie kroki treningu
na dopuszczonych danych, obecnie formalnych ćwiczeniach albo zweryfikowanym feedbacku.

Log kontrolera jest w `research/mission/run-*/controller.log`; logi modeli,
baseline, `learning/state.json`, próby A/B i metryki loss/eval_loss mają osobne
pliki. Błąd pojedynczego researchu, nieudany upgrade, limit danych lub brak miejsca
na eksport odkłada aktualizację i zachowuje poprzednią wersję. Awaria kontrolera
lub obserwatora jest jawnie zapisana; nie jest raportowana jako sukces.

Misja wykonuje research dochodu, lecz transakcje paper nadal wymagają
zarejestrowanych, zweryfikowanych opłat, instrumentów i świeżych feedów. Nie
dopowiada zerowych opłat i nie zakłada kont, nie realizuje sprzedaży ani zleceń
za prawdziwe pieniądze. Obecny reader czyta publiczne URL-e, nie ma pełnej
wyszukiwarki ani automatyzacji logowania. Nie jest to nieograniczona przebudowa
architektury, dowód wzrostu inteligencji ani gwarancja braku zapominania/zysku.
Speculative decoding pozostaje wyłączone w profilu misji do osobnego sprawdzenia
kompatybilności i przyspieszenia po zmianach wag.

Walidacja kontrolera v100.13: 514 testów zaliczonych, 63 pominięte; lint,
formatter, hooki i dodatkowe sprawdzenie typów nowych modułów przeszły. Testy
obejmują oba przypadki narzędzi przez akcje JSON, zachowanie oryginałów podczas
kompresji, automatyczny wybór kontekstu i odrzucenie nieudanego upgrade'u.

## Trudniejsze ćwiczenia i rzeczywiste wywołania narzędzi, v100.12

Po pełnym wyniku 41/41 w `income-bootstrap-v1/baseline-thinking.json` użyj nowego
profilu `research/v100-thinking.json` przez `--profile` i `prepare-challenge`.
Komenda potwierdza hash modelu, zestawu oraz warunki generacji z ukończonego
baseline. Tworzy osobny katalog `research/income-challenge-v1`:

- `pool.jsonl`: 256 nowych ćwiczeń z dzieleniem całkowitym, modulo oraz równaniami,
  plus poprzednie 128 przykładów jako replay. Wszystkie odpowiedzi oblicza stały
  referencyjny kontroler. Zachowuje dotychczasowe przypisania train/validation.
- `development.jsonl`: 41 dotychczasowych przypadków plus 24 nowe zadania
  matematyczne, 8 zadań pracy na źródłach/kontekście i 8 przypadków narzędziowych.
  Nowe źródła oceny są zarezerwowane przed tworzeniem podziału treningowego.
- `smoke.jsonl`: dwa przypadki sprawdzające kalkulator i odczyt źródła z pamięci.
- `manifest.json`: hashe plików, ustawienia generacji oraz liczebności podziałów.

Istniejące pliki o odmiennej treści nie są zastępowane. Przygotowanie nie zmienia
wag, oryginalnego zestawu, live memory ani portfeli. To jawny zestaw deweloperski,
nie tajny audit i nie dowód skuteczności na realnym rynku. Replay i testy regresji
nie dają uniwersalnej gwarancji braku zapominania.

Najpierw uruchom `challenge-smoke` w tym samym profilu. Zarządza jednym serwerem
GPU, odmawia zajętej karty i wyłącza własny serwer po ocenie. Po sprawdzeniu dwóch
przypadków uruchom `challenge-baseline`, aby ocenić wszystkie 81. Raporty
`tool-smoke.json` oraz `baseline.json` nie są nadpisywane. Dane źródłowe trafiają
do osobnego SQLite, który jest zamykany i otwierany ponownie przed odczytem.
Rekordy z czasem publikacji późniejszym niż cutoff nie trafiają do tej pamięci.
Testuje to zapis/odczyt i dobór narzędzi w stałych warunkach, nie całą długotrwałą
pamięć użytkownika ani odporność na wszystkie możliwe ataki.

`calculate` używa istniejącego ograniczonego parsera wyrażeń całkowitych;
nie wykonuje kodu modelu. Każda stała ma wartość bezwzględną do 10000, a wynik
pośredni do 10^12. Ćwiczenia kosztowe podają grosze oraz jawne fikcyjne opłaty,
funding i dodatkowy slippage. To dane zadania, nie taryfa prawdziwego brokera.
Wywołania narzędzi, argumenty i wyniki są w `tool_trace` raportu i dzienniku
aktywności. Samo zgadnięcie poprawnej liczby bez wymaganego narzędzia nie zalicza
przypadku; wynik kosztowy musi odpowiadać wynikowi kalkulatora. Raport podaje
czas całego przypadku, finish_reason i liczbę znaków rozumowania bez jego tekstu.

Ten etap nie uruchamia treningu A/B, speculative decoding ani transakcji.
Walidacja kontrolera v100.12: 492 testy zaliczone, 63 pominięte; lint, formatter
i hooki przeszły. Obsługa narzędzi przez model na V100 wymaga lokalnego smoke testu.

## Osobny profil rozumowania, v100.11

`prepare-thinking` tworzy `research/v100-thinking.json` z aktualnego profilu:
`enable_thinking=true`, budżet wyjścia 2048 tokenów, temperature 1.0, top_p 0.95,
top_k 64 i seed 42. Nie zmienia starego profilu, modelu ani portfeli. Przy ponownym
wywołaniu akceptuje identyczny plik; odmienny istniejący profil nie jest nadpisywany.
Przekazuj nowy plik przez `--profile` do oceny i późniejszego `learn-loop`.
Klienci rodziców/R&D oraz ocena A/B dziedziczą te ustawienia. Osobny CPU researcher
pozostaje w szybkim trybie bez rozumowania. Benchmark przepustowości domyślnie
pozostaje bez rozumowania; jego `--thinking` jest osobnym pomiarem.

Wstępna diagnoza na czterech zadaniach zmieniła wynik 0/4 na 4/4 przy niezmienionych
wagach, z czasem odpowiedzi 9–38 sekund. Zmieniono jednocześnie tryb rozumowania,
sampling i budżet; nie jest to izolowany test jednej przyczyny. Następny krok to
pełne 41 przypadków zapisane jako `baseline-thinking.json`. Zachowaj stare
`baseline.json` (9/41); raporty z różnymi warunkami generacji nie mogą służyć jako
porównanie przed/po treningu. Nowe kandydaty oceniaj wobec nowego baseline w tych
samych warunkach. Cztery zadania nie dowodzą poprawy całego modelu.

Raport jakości zapisuje warunki, finish_reason, liczbę znaków rozumowania i błędy,
bez tekstu wewnętrznego rozumowania. Rozbieżny klient jest odrzucany przed oceną;
brak końcowej odpowiedzi lub limit tokenów oznacza niezaliczony przypadek.
Profil nie włącza speculative decoding i nie uruchamia aktualizacji wag.
Walidacja kodu v100.11: 475 testów zaliczonych, 63 pominięte; lint, formatter
i hooki przeszły. To testy kontrolera na CPU, nie pełna ocena modelu na V100.

## Badania finansowe i paper trading, v100.9

Osobne portfele A/B, decyzje zapisane przed kolejną obserwacją, research/critic,
koszty wejścia i wyjścia, limity dywersyfikacji oraz raporty HTML z wykresami
opisuje [PAPER_RESEARCH.md](PAPER_RESEARCH.md). Pierwszy krok: `paper-init`.
Brak zweryfikowanych opłat lub świeżych danych blokuje transakcję. Dostępne są
odczyty Coinbase spot i metadanych SEC; akcje, instrumenty z dźwignią i zakłady
wymagają osobnego zatwierdzonego feedu. Wynik paper nie jest automatyczną etykietą
do treningu wag. Harmonogram raportów działa tylko przy uruchomionej lokalnej pętli.

v100.10 dodaje `learn-loop --paper-config FILE`: research paper/innych legalnych
dochodów może zasilać istniejący trening A/B przez referencyjnie zweryfikowane
ćwiczenia. Dane rynkowe i raporty są obserwowane również podczas treningu.
`paper-learning-status --datasets DIR` sprawdza dostępne wejścia bez ich odczytu
lub uruchamiania modelu. Szczegóły i ograniczenia opisuje dokument paper.

## Czytelny dziennik pracy, v100.8

Logi powstają w `research/logs/activity/RRRR-MM-DD/`. `timeline.jsonl` zawiera
wspólną chronologię. Osobne katalogi `A`, `B`, `shared`, `controller` zawierają
podkatalogi `model`, `trainer` oraz `researcher`, `tester` lub `critic`, zależnie
od uczestnika. Pliki `decisions`, `steps`, `tools`, `research`, `training`,
`metrics`, `errors` mają wersję JSONL do analizy oraz Markdown do czytania.
Powstają tylko pliki kategorii, w których wystąpiły zdarzenia.

Dziennik obejmuje jawne uzasadnienia wyboru danych/hiperparametrów, wyniki
testerów, akceptacje i odrzucenia, start/wynik narzędzia, czas inferencji,
zużycie tokenów i natywne liczniki draftu. Kroki mają czas UTC, identyfikatory
i powiązania request/step; wpisy pamięci publicznej mają numer sekwencji SQLite.
Kontekst inferencji wskazuje wersję modelu, docelowy plik i draft. Trening ma
osobny dziennik loss/eval_loss, KL, LR, grad_norm i VRAM, gdy metrykę poda Trainer,
oraz ukończone checkpointy i najlepszy checkpoint na końcu treningu.
Oryginalny `metrics.jsonl` w katalogu danego treningu nadal pozostaje źródłem.

To jawne decyzje i obserwowalne kroki, nie zapis prywatnego toku rozumowania.
Logger nie zapisuje promptów żądań, system promptów, nagłówków HTTP, kluczy
z konfiguracji ani `reasoning_content`. Znane formaty sekretów w treści są
maskowane; nie umieszczaj haseł w danych/modelowych odpowiedziach. Wpisy są
ograniczone objętościowo, rozdzielane dziennie, a zapis współbieżny blokowany
na czas dopisania. Pliki tworzone są z uprawnieniami 0600. Nie są automatycznie
usuwane: archiwizuj starsze dni. Dzienniki nie stają się danymi treningowymi
ani publiczną pamięcią modelu. Eksport dziennika pamięci publicznej następuje
po zatwierdzeniu SQLite; przerwanie między zapisami może pozostawić lukę w eksporcie.

## Speculative decoding i większy draft

`prepare-mtp` zachowuje oryginalny profil i tworzy osobne profile baseline,
MTP2, MTP4, MTP8 i MTP16 z przypiętym kompatybilnym asystentem Gemma Q8_0.
Dotychczasowe profile nie są nadpisywane. `test-mtp` domyślnie mierzy wszystkie
te rozmiary sekwencyjnie; `--draft-tokens` ogranicza sweep do wybranych rozmiarów.
Wymaga wolnej GPU i portu 8089, zapisuje natywny log serwera, CSV GPU i raport
każdej próby. Błąd większego draftu zachowuje logi i pozwala sprawdzić następny.

`comparison.json` pokazuje szybkość i zaakceptowane drafty; wariant bez natywnych
liczników draftu nie jest uznawany za działające MTP. `failures.json` zbiera
nieudane warianty. `recommendation.json` wskazuje kandydata szybszego o minimum
5%, z identycznymi odpowiedziami w tym benchmarku i bez spowolnienia żadnego
przypadku o więcej niż 5%. To kandydat wydajnościowy, nie certyfikat jakości:
wymaga osobnej bramki jakości przed uruchomieniem. Nie ma automatycznej promocji.
Większy draft nie musi przyspieszać: koszt odrzuconych tokenów i VRAM może rosnąć.

Po zmianie wag, architektury, draftu lub runtime przygotuj osobne profile testowe
z nowym targetem i powtórz test. Dotychczasowe profile nie są automatycznie
przepisywane na nowe wagi. Podczas eksportu i oceny nowego adaptera draft nadal jest
wyłączany do czasu ponownego sprawdzenia zgodności/wydajności. MTP przyspiesza
inferencję; nie zastępuje ani nie przyspiesza samo w sobie backward treningu.
Ta wersja nie została uruchomiona na Twojej V100: wynik wymaga pomiaru na Debianie.
Zmiana kodu treningu zmienia manifest wznowienia; starego checkpointu poprzedniej
wersji nie wznawiaj nowym kodem, rozpocznij nową rundę z zaakceptowanego adaptera.

## Twój cel i współpraca A/B

`set-goal TEXT --suite DEV_SUITE` ustawia wspólny cel opisany Twoimi słowami
i wiąże go z SHA256 zewnętrznego zestawu testów. Kolejne cele zachowują osobne
snapshoty. Planner, testerzy i researcherzy otrzymują cel i publiczne wnioski
obu gałęzi. A/B mogą proponować różne rozwiązania albo współpracować.
Wygrywa liczba niezależnie zaliczonych zadań po bramkach zachowania dawnych
wyników; model nie może zmienić kryterium ani zatwierdzić swojego wyniku.
Sam tekst celu nie tworzy poprawnego benchmarku: testy muszą mierzyć ten cel.

Koszt mierzy kontroler: trening, eksport, ocena i całkowity czas próby.
Przy równej jakości kontynuuje gałąź szybsza o co najmniej 5%; mniejsza różnica
lub brak poprawnych pomiarów daje jawne kontynuowanie A. Obie wersje zostają.
To heurystyka z pojedynczej próby, nie statystyczny dowód najlepszego ustawienia.
Raport `performance.json` jest powiązany z wagami i raportem jakości przez SHA256.

## Zachowanie wcześniejszych wersji

`protect-baseline ID --description ...` zapisuje niezależne kopie oryginalnego
GGUF, llama-server i jego lokalnych bibliotek. `serve --expert ID` weryfikuje
ich SHA256 i uruchamia przypięte ustawienia. Pliki są tylko do odczytu, rejestracja
istniejącego ID i zapis treningu w chronione ścieżki są odrzucane. Wymagane miejsce
na dysku: kolejna kopia modelu i bibliotek na każdego eksperta.

Ochrona oznacza zachowanie poprzednich plików i jawnej ścieżki uruchomienia.
Nie dowodzi braku pogorszenia odpowiedzi nowego kandydata ani poprawności routera.
Zmiana systemowych bibliotek, sterownika, promptów lub źródeł może zmienić zachowanie.
Właściciel konta może zmienić uprawnienia plików; to nie ochrona przed administratorem.

## Wagi rzeczywiście się uczą

`train DATASET` trenuje osobnego kandydata LoRA na zatwierdzonym feedbacku lub
kanonicznych przykładach ponownie sprawdzonych przez zewnętrzny kalkulator. Eksport
pamięci obejmuje również poprzednie zatwierdzone przykłady jako replay. Model
nie zatwierdza własnych odpowiedzi. Tylko parametry kandydata LoRA są trenowalne.
Źródła pozostają przypisane do train/validation w SQLite między rundami; próba
połączenia źródła treningowego i walidacyjnego kończy się błędem. Identyfikatory
niezależnego audytu rezerwujemy przed treningiem przez `reserve-audit`.

Profil ustawia `distillation_weight=0.1`, `distillation_temperature=1.0`.
Dodatkowy forward zamrożonego poprzednika dostarcza KL na nadzorowanych tokenach
odpowiedzi. Baza jest wspólna, bez drugiej kopii 12B na GPU. Nauczyciel to
`teacher_adapter`, w przeciwnym razie `init_adapter`, a w pierwszej rundzie baza
bez adaptera. Większa kara nie oznacza automatycznie lepszych wyników; dodatkowy
forward kosztuje czas i pamięć. To eksperymentalna regularyzacja, nie gwarancja.

`metrics.jsonl` pokazuje loss, supervised_loss, preservation_kl, eval_loss,
learning rate, grad_norm oraz peak_vram_gib, gdy te pola występują w logach Trainer.
Checkpoint obejmuje adapter, optimizer, scheduler, RNG i stan Trainer.
`train DATASET --resume` wybiera ostatni kompletny checkpoint; nie pozwala zmienić
zbioru, bazy, początkowego/nauczycielskiego adaptera, ustawień ani kodu między wznowieniami.
Nowa runda ma nowe `training.output`. Stare checkpointy bez nowego manifestu nie są
automatycznie zgodne. Dziedziczenie wymaga adaptera z `v100-adapter.json`.

Od v100.5 zapis co `save_steps` wybiera najlepszy kompletny checkpoint według
`eval_loss`. Wymagamy `save_steps == eval_steps <= max_steps`, domyślnie 25 kroków.
Trainer zachowuje najlepszy przy rotacji (limit 3 checkpointy), a na końcu wczytuje
jego wagi do `candidate`. `best.json` zawiera ścieżkę, loss i identyfikator manifestu.
Wznowienie nadal używa najnowszego kompletnego checkpointu z optimizerem/RNG,
nie samego eksportu wag najlepszego. „Najlepszy loss” nie jest werdyktem jakości.

## A/B i pomocnicy CPU podczas treningu

`plan-duel POOL --output DIR` pyta lokalny model o dwie hipotezy, przykłady z
zatwierdzonego zbioru i ograniczone hiperparametry. Model widzi pytania treningowe,
bez odpowiedzi walidacyjnych. Wybiera learning rate, rank, długość sekwencji,
akumulację gradientów, liczbę kroków i siłę KL w budżecie. A/B mają osobne profile,
seedy, adaptery, checkpointy i logi. Split całego zbioru powstaje przed wyborem
curriculum. Walidacja jest identyczna w obu gałęziach. `--replay PREVIOUS_POOL`
wymusza zachowanie wcześniejszych przykładów treningowych w następnej rundzie.
Rozmowy, hipotezy i wyniki development trafiają do wspólnego
`research/state/competition.sqlite3`; do kontekstu trafiają ograniczone fragmenty.
Odpowiedzi pomocników nie stają się automatycznie prawdziwymi etykietami treningu.
Nowy mechanizm formalnych ćwiczeń opisano poniżej.

`prepare-researcher` przygotowuje osobny profil `research/researcher-cpu.toml`
na porcie 8090. Pobiera oficjalny Qwen3-0.6B Q8_0 GGUF, przypina rewizję i sprawdza
SHA256 artefaktu. Używa tego samego skompilowanego llama-server, `gpu_layers=0`,
4 wątków, kontekstu 4096 dla nowych profili i nice=10. Istniejący profil nie jest
nadpisywany. Kontroler wyłącza widoczność CUDA dla tego
procesu. Rozmiar pliku wag około 639 MB nie obejmuje KV cache i buforów procesu.

`run-duel DIR --suite DEV_SUITE --baseline-report BASELINE --researcher-profile CPU_PROFILE`
trenuje A, eksportuje Q6_K, ocenia A, a następnie robi to samo z B. Na V100 jest
jedna duża gałąź naraz. Mały model CPU pozostaje uruchomiony równolegle z treningiem
GPU i czyta aktualne, kompletne wiersze metryk. Planner może zlecić do 3 zadań na
gałąź jako tester/researcher/critic. To różne role korzystające z jednego serwera
0.6B, nie niezależnie trenowane sieci potomne. Dodatkowo mogą tworzyć i trenować
osobne małe sieci przez izolowane narzędzia poniżej. Nie są sędziami jakości. Błąd/timeout pomocnika
zapisuje się jawnie, bez przerwania poprawnie działającego treningu.

Model główny po eksporcie przegląda wyniki pomocników, może odrzucić wszystkie i
zapisać własny wniosek. Zadania nie zmieniają hiperparametrów w środku bieżącego
checkpointu; wnioski służą planowaniu kolejnej rundy. Przed startem pomocnika i
zleceniem zadania wymagamy co najmniej 6 GiB dostępnego RAM. To kontrola zapasu,
nie twardy limit pamięci ani gwarancja braku OOM. Na 31 GiB RAM trzeba zmierzyć
współbieżny szczyt, zwłaszcza ładowanie/eksport FP16 i cache dyskowy. Prędkość
Qwen na starym CPU bez AVX2 wymaga pomiaru; nie zakładamy automatycznego przyspieszenia.

Kontroler wypisuje na żywo nowe wiersze metryk z nazwą gałęzi oraz dostępnym RAM,
a pełny stdout/stderr treningu zachowuje w `A/training.log` i `B/training.log`.

Kontroler wymaga 28 GiB wolnego VRAM przed treningiem/eksportem. Nie zatrzymuje
sam cudzych serwerów. W szczególności trzeba wcześniej zatrzymać własny serwer
8088/8089. Zatrzymuje wyłącznie procesy, które sam uruchomił. Gałęzie i pomocnik
korzystają z różnych portów. Cała runda ma limit czasu, domyślnie 7200 s na gałąź.
Przerwany trening można wznowić przez jego profil i `train --resume`; `run-duel`
nie jest obecnie pełnym wznawialnym workflow eksportu i ewaluacji.

Sędzia używa stałych walidatorów exact/contains poza modelem. Dopuszcza gałąź tylko
bez utraty któregokolwiek wcześniejszego zaliczonego przypadku i wybiera liczbę
zaliczonych zadań; równe wyniki to remis, obie regresje to brak zwycięzcy.
Można podać wiele `--baseline-report` dla wcześniejszych rodziców.
Raporty są związane z SHA modelu, suite i warunkami wykonania. Wynik zapisuje się
w `judgment.json`. Nie promuje automatycznie eksperta. Suite development jest
jawna i podatna na przeuczenie przy wielu rundach, więc końcowy audit musi pozostać
oddzielny. Eksperymenty nowych architektur mają osobny protokół; ten sędzia
duelu ocenia wyłącznie kandydatów obecnej głównej ścieżki Gemma/LoRA.

`evolve POOL --output DIR --suite DEV_SUITE --baseline-report BASELINE --researcher-profile CPU_PROFILE`
automatyzuje 1-4 takich pokoleń (domyślnie 2). Po zakończeniu pokolenia unloaduje
GPU, a kolejną parę planuje jego zwycięzca; używa jego adaptera jako inicjalizacji
i zamrożonego nauczyciela. Zachowuje poprzedni pool jako obowiązkowy replay oraz
raporty wszystkich wcześniejszych dopuszczonych gałęzi jako bramki regresji.
Przy remisie jakości kontynuuje według pomiaru czasu opisanego wyżej,
zachowując również drugą gałąź; przy braku dopuszczonego
zwycięzcy kończy cykl. `evolution.json` zapisuje wyniki każdego pokolenia.
Nie podmienia głównego profilu ani nie promuje eksperta. Pobiera nowe formalne
przykłady wyłącznie po niezależnym sprawdzeniu i przyjęciu przez rodzica.
To ograniczona ewolucja adapterów/hiperparametrów, bez automatycznej
przebudowy architektury ani krzyżowania wag. Cross breeding jest osobną komendą.
Pełny wielopokoleniowy cykl nie ma jeszcze automatycznego wznowienia po awarii.

## Samodzielne wnioski, internet i zmiana wag

Pomocnik proponuje do dwóch nowych ćwiczeń: ograniczoną arytmetykę całkowitą
lub równanie liniowe. Stały kalkulator hosta liczy odpowiedź, nie wykonując
zaproponowanego Pythona. Wpis trafia do kolejki pending w
`research/state/verified-insights.sqlite3`. Rodzic może go odrzucić; dopiero
sprawdzone i przyjęte ćwiczenie wchodzi do nowego snapshotu danych.
Dedup, ponowna walidacja etykiet i stałe podziały źródeł obejmują kolejne rundy.
Ta wersja nie potwierdza automatycznie dowolnej tezy naukowej, faktu z WWW ani
swobodnego podsumowania. Takie wnioski zostają w pamięci jako hipotezy.

Researcher wybiera narzędzia, maksymalnie dwa wywołania na zadanie. Może czytać
publiczny HTTPS tekst/HTML/JSON z limitem 256 KiB i wykonywać stałe testy
istniejącego kandydata kodu przez bubblewrap. Czytnik zachowuje URL, SHA i tekst,
blokuje adresy lokalne/prywatne oraz ponownie sprawdza przekierowania.
Nie jest wyszukiwarką, przeglądarką z logowaniem ani czytnikiem PDF.
Nie ma konsultacji przez płatne API. Automatyzacja darmowego LLM przez stronę,
tworzenie kont, logowanie i CAPTCHA nie są podłączone; wymagają konkretnej
usługi i jej obsługi. Nie ma obchodzenia limitów przez kolejne konta.

Od v100.7 model może przez `list_free_services`, `propose_free_service` oraz
`choose_free_service` sam badać i zmieniać preferowaną stronę do danego zadania.
Nie narzucamy jednej witryny. Dokumentację pobiera istniejącym czytnikiem;
URL, SHA i czas trafiają do osobnego SQLite. Dowód starszy niż 24 h lub zmieniony
tekst wymagają ponownego researchu. Wybór jest publiczny dla A/B, ale nie oznacza
zweryfikowanej darmowości ani uruchomionej konsultacji przez przeglądarkę.

Osobny działający protokół `list_free_models` / `consult_free_model` obsługuje
OpenRouter wyłącznie w wariantach `:free`. A/B i ich testerzy mogą wybierać model
z aktualnej listy; katalog nie jest rankingiem inteligencji. Przed każdym POST
kontroler sprawdza zero we wszystkich polach pricing oraz konto `is_free_tier`.
Żądanie ma max_price prompt/completion/request równe 0, bez fallbacków i pluginów,
z wymogiem distillable text. Płatna trasa, nieznana cena lub finansowane konto
są blokowane. To poleganie na zadeklarowanych cenach i egzekwowaniu limitu przez
dostawcę, nie możliwość kontrolowania jego rozliczeń od strony klienta.

Konsultacje potrzebują darmowego konta OpenRouter i klucza w lokalnym
`V100_FREE_ROUTER_KEY`. Żadnego konta ani klucza nie utworzono w tej rozmowie,
nie wykonano rzeczywistej konsultacji. Nie dodajemy karty, kredytów ani billing.
Jeden klucz służy wszystkim gałęziom: do 40 prób dziennie UTC, minimum 5 s
odstępu. Błąd/quota zatrzymuje wywołanie bez płatnej zmiany modelu i automatycznych
retry. Zewnętrzna odpowiedź jest hipotezą, nie etykietą treningową ani werdyktem.
Niezerowy lub niepoprawny koszt zgłoszony przez usługę blokuje dalsze konsultacje
do przeglądu przez operatora; nie ponawia automatycznie problematycznej trasy.
Ślad zawiera model, SHA pytania/odpowiedzi i czas; klucz nie trafia do trace.
Testy używają zastępczego transportu. Logowanie i konfiguracja na docelowym
Debianie pozostają do wykonania; kod nie daje gwarancji dostępności darmowych modeli.

Źródła protokołu: [OpenRouter free models](https://openrouter.ai/collections/free-models),
[provider routing](https://openrouter.ai/docs/guides/routing/provider-selection),
[status klucza](https://openrouter.ai/docs/api/api-reference/api-keys/get-current-api-key).

`learn-loop POOL --output DIR --suite DEV_SUITE --baseline-report BASELINE --researcher-profile CPU_PROFILE`
powtarza R&D A/B i weryfikację. Tylko NOWE sprawdzone, przyjęte przykłady
uruchamiają kolejną rundę optimizer/backward z replay i KL. Między rundami
serwuje bieżącą wersję; przed treningiem zatrzymuje tylko własne serwery GPU.
CPU pomocnik może pracować podczas treningu. Nie zapewnia równoczesnej
dostępności głównego modelu w trakcie całej aktualizacji.

Domyślnie wykonuje 4 cykle; `--cycles 0` działa do przerwania. Bez nowych danych
czeka `--interval` sekund (domyślnie 600), bez trenowania identycznego zbioru.
Wymaga 160 GiB wolnego dysku przed parą eksportów, limitu 10000 rekordów replay,
28 GiB wolnego VRAM przed fazą treningu i stałego celu/suite. Zmiana celu
przerywa tę serię. Nie śledzi dowolnych późniejszych zmian seedowego pliku.
Przechodząca wersja jest wybierana do następnej fazy serwowania, odrzucona nie
zastępuje poprzedniej. Zachowuje wszystkie historyczne bramki jakości.
Po skończonej liczbie cykli własne serwery są zatrzymane; `live.json` zawiera
wybrany profil do późniejszego `serve`, `ask` i kolejnej rundy. Oryginalny profil
i wrapper pozostają bez zmian. `state.json` zapisuje przebieg, ale cały loop
nie ma jeszcze automatycznego wznowienia po awarii. Wznowienie pojedynczego
kompletnego treningu działa przez jego profil i zgodny manifest.

Wagi LoRA zmieniają się w krokach optymalizatora, a eksport scala je z bazą.
Samo zapisanie wniosku nie aktualizuje parametrów. Ten cykl automatyzuje drogę
od sprawdzonego wniosku do danych i rzeczywistego treningu. Nie gwarantuje
całkowitego braku catastrophic forgetting poza skończonymi testami.

## Nowe sieci i własne architektury

Narzędzia `create_submodel`, `support_submodel`, `test_submodel` pozwalają A/B
napisać cały kod małej sieci PyTorch i trenować wszystkie jej wagi od zera.
Może to być np. RNN, GRU lub mały transformer. Kontrakt jest stały: build(config)
zwraca nn.Module, wejście to tokeny bajtowe [B,T], wyjście logits [B,T,257].
To inny tokenizer i backend niż Gemma. Kod musi sam zapewnić przyczynowość.
Nie ma dowodu poprawności dowolnego wygenerowanego algorytmu.

Wspólna próba wymaga osobnych zgód A i B na identyczny kod, cel oraz budżet.
Zmiana budżetu unieważnia poprzednie zgody. Propozycje i wyniki są publiczne
w pamięci obu gałęzi, więc mogą rozwijać pomysły partnera. Model nie zmienia
system promptu, zewnętrznego sędziego ani zaakceptowanych poprzednich wag.

`prepare-submodels POOL --suite DEV_SUITE` przygotowuje kontrolowane snapshoty.
Loop robi to automatycznie. Samodzielny tester CPU dostaje domyślnie 2 wątki,
2M parametrów, 256 bajtów kontekstu, 40 kroków, 120 s na każdą fazę,
8 GiB limitu przestrzeni wirtualnej i 512 MiB artefaktów. Próbę odkłada, jeśli
obok rodzica pozostaje mniej niż 12 GiB dostępnego RAM. Jeden taki trening CPU
naraz. To dodatki do głównego GPU, a nie jego zastąpienie.

`run-submodel ID POOL --suite DEV_SUITE --budget BUDGET_JSON` pozwala jawnie
wybrać większy budżet, także CUDA, do granic zewnętrznego kontrolera. CUDA
wymaga wolnej V100 i korzysta z tej samej blokady co run-duel. Nie ładuje dwóch
pełnych głównych Gemm naraz. V100 nie ma MIG: nie tworzymy sprzętowych partycji
VRAM ani nie gwarantujemy, że niekooperujący kod nie przekroczy limitu chwilowo.
Limit allocatora PyTorch, okresowe monitorowanie RAM/VRAM/artefaktów oraz
zatrzymanie grupy procesów po przekroczeniu budżetu ograniczają próbę.
CPU dodatkowo ma prlimit AS/CPU/rozmiaru pliku; CUDA nie ma limitu przestrzeni
wirtualnej ze względu na rezerwacje adresowe sterownika. Wszystkie próby wymagają
bubblewrap i prlimit, bez sieci i zastępczego wykonania na hoście.

Trening widzi wyłącznie część train. Ocena dostaje pytania bez gold odpowiedzi,
a exact/contains liczy zewnętrzny kontroler. `trial/quality.json` wiąże wynik
z SHA kodu, wag i suite. Loss w train.log jest tylko metryką pomocniczą.
Wybrane wagi zapisują się jako safetensors według najmniejszego obserwowanego
training loss; te krótkie próby nie mają jeszcze pełnego wznowienia optimizer/RNG.
Nie ma automatycznej podmiany głównej Gemmy, eksportu dowolnej architektury do
GGUF ani krzyżowania niezgodnych sieci. Ich wyniki mogą kierować dalszym R&D;
produkcja nowego backendu wymaga kolejnego wdrożenia i bramek jakości.

## Kandydaty zmian własnego kodu

`propose-code REPOSITORY FILE --output DIR` pozwala modelowi zmienić jeden
śledzony plik algorytmu przez jedno dokładne zastąpienie tekstu, z hipotezą.
Rodzic i zamrożona kopia testów zostają identyczni. Pliki promptów oraz zewnętrzny
kontroler nie są modyfikowane. Wygenerowany kod nie jest importowany na hoście.
`check-code DIR` uruchamia widoczne testy w bubblewrap: osobne namespaces, bez
sieci i CUDA, tylko odczyt źródeł/testów, prywatne `/tmp`, minimalne środowisko.
Wymaga opcjonalnego zestawu zależności `selflab` oraz działającego bubblewrap w OS.
Brak izolacji lub błąd testów blokuje wykonanie; nie ma wykonania zastępczego na hoście.

`run-duel --code-a DIR` lub `--code-b DIR` może użyć takiego sprawdzonego kandydata
podczas treningu w tej samej izolacji, z jawnym dostępem do V100. Kandydat dostaje
tylko prywatny output i kopię ledgera do zapisu; baza, dataset i adaptery rodziców
są do odczytu. Zwykły eksport i ocena pozostają w kontrolerze spoza kandydata.
Przejście widocznych testów nie dowodzi braku manipulacji przez wygenerowany kod;
jego własny loss może być niewiarygodny. Potrzebna pozostaje niezależna ocena
wynikowego modelu. Nie dostaje dostępu do całego konta Linux ani ukrytego audytu.

## Cross breeding

`breed-adapters FIRST SECOND --output ROUND --alpha W` tworzy
`ROUND/candidate` z dwóch zgodnych adapterów tej samej dokładnej bazy.
SHA256 wag, konfiguracji i tokenizera identyfikują bazę. Ranki rodziców mogą się różnić;
pozostałe ustawienia LoRA muszą być zgodne. Obsługiwane są standardowe liniowe LoRA
bez dodatkowych trenowanych embeddingów, bias, DoRA, RSLoRA i wzorców rank/alpha.

Metoda: konkatenacja faktorów, która daje dokładnie
`delta_child = W * delta_first + (1-W) * delta_second`, z uwzględnieniem skalowania
alpha/r każdego rodzica. To nie jest średnia faktorów A i B, która dodawałaby
niezamierzone iloczyny. Obliczenia dotyczą adapterów na CPU, bez ładowania 12B.
Rank potomka jest sumą ranków; dalszy trening będzie więc droższy. Nie wprowadzono
SVD, TIES ani DARE, bo wymagają osobnych pomiarów strat i jakości.

Rodzice pozostają identyczni. Potomek jest tylko kandydatem. Można utworzyć kilka
osobnych mieszanek (np. 0.25, 0.5, 0.75), wybrać je na development set, a potem
kontynuować trening na zweryfikowanych przykładach obu rodziców. W profilu następnej
rundy `init_adapter` wskazuje `ROUND/candidate`, a `training.output` nowy katalog.
Możliwe jest ograniczone planowanie pokoleń przez `evolve`, ale nie wykonano tego
cyklu na Twoim serwerze. Nie ma samopotwierdzania danych.
Nie ma obecnie gotowych dwóch wytrenowanych rodziców na Twoim serwerze.

`export-model` scala kandydata z bazą do osobnego FP16 i Q6_K. Dla potomka trzeba
najpierw wyeksportować obu rodziców. Zapis eksportu wiąże ich adaptery z konkretnymi
SHA modeli GGUF; ręcznie podmienione wagi/eksporty są odrzucane.

## Bramka jakości

`evaluate-suite SUITE --output REPORT` sprawdza deterministyczne zadania z JSONL.
Każdy rekord ma `id`, `skill`, `messages`, `expected`, `match` (exact albo contains).
To ograniczone walidatory, nie ogólny sędzia poprawności. Sam expected/contains nie
wystarczy do oceny dużych programów, rozumowania i fałszywych cytowań.

Raport wiąże suite SHA, model SHA, warunki generacji i wykonania. Serwer musi być
uruchomiony tą wersją kontrolera, aby mieć lokalny receipt procesu. Profile i
warunki generacji obu porównań muszą się zgadzać. `compare-quality` odrzuca każdy
przypadek wcześniej poprawny, który teraz jest błędny, nawet jeśli średnia rośnie.

`register-expert ID --description ... --baseline-report OLD --candidate-report NEW`
przyjmuje nowego eksperta dopiero po tej bramce. Dla krzyżowanego potomka podaj
`--baseline-report` dla każdego rodzica. Oba raporty muszą dotyczyć rzeczywistych
eksportów rodziców i tej samej połączonej suite; potomek musi zachować każdy
przypadek zaliczony przez któregokolwiek rodzica. Dotyczy to wyłącznie skończonych
prób, nie wszystkich możliwych wejść. Spadek loss i wzrost tok/s nie dowodzą jakości.

Do wybierania mieszanek używamy development set; oddzielny ukryty audit służy
końcowej ocenie. Ten kontroler nie zapewnia procesu ukrywania audytu przed operatorem
ani odporności na manipulowanie raportami przez właściciela plików.

## Pamięć i lokalne narzędzia, bez Jev

`prepare-embeddings` pobiera przypiętą wersję
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` i zapisuje SHA plików.
`index-memory` indeksuje oryginalne fragmenty do wersjonowanych wektorów w SQLite.
Encoder działa domyślnie na CPU, aby nie zabierać VRAM V100. Długie fragmenty dzieli
na okna zamiast obcinać; retrieval łączy cosine i FTS przez reciprocal rank fusion.
Wyszukiwanie cosine jest dokładne i liniowe względem rozmiaru indeksu, bez FAISS
wymagającego dodatkowych binarek. Dla dużych zbiorów trzeba zmierzyć koszt i dodać ANN.

Po przygotowaniu i indeksowaniu ustaw `memory.retrieval=hybrid`; domyślny profil
pozostaje lexical, aby instalacja działała bez dodatkowego pobrania encodera.
Nowo dodane źródła trzeba zaindeksować przed wyszukiwaniem hybrid. Zmiana encodera
wymaga nowej wersji indeksu. `backup-memory DEST` atomowo zapisuje niezależny snapshot
SQLite przez backup API, z uwzględnieniem WAL. Kopię trzeba też przenieść na inny dysk.

`ask QUESTION --agent` pozwala lokalnej Gemmie wybrać `search_memory` i `read_source`.
Narzędzia czytają źródła, mają limit tur, walidację argumentów i budżet kontekstu.
Nie wykonują powłoki ani wygenerowanego Python i nie zmieniają wag.
To nie pełna ochrona przed prompt injection; model może błędnie wybrać źródła/narzędzia.
Ślad wyboru zapisuje się w runie. Rozmowa agentowa nie jest automatycznie materiałem treningowym.

`route-expert QUESTION` wybiera tylko z zarejestrowanych ekspertów, przez lokalny model.
Jawne `--expert ID` ma pierwszeństwo. `ask --auto-expert` sprawdza, czy wybrany ekspert
jest rzeczywiście uruchomiony. Jedna V100 nie ma tu automatycznie załadowanego całego
banku; jeśli działa inny ekspert, komenda odmawia zamiast udawać poprawne przełączenie.
Nie zaimplementowano automatycznej orkiestracji restartów/model swapping.

## Stan walidacji

Mechanizmy sprawdzono lokalnie na CPU, w tym prawdziwy trening małej Llamy z PEFT,
checkpointowanym backward i KL: baza oraz nauczyciel nie zmieniły się, kandydat się zmienił.
Testy obejmują dokładność sumy delt, niezmienność rodziców, split leakage, backup,
semantyczny retrieval na mock encoderze, narzędzia, snapshoty i odmowę regresji obu rodziców.
Nie uruchomiono nowego treningu Gemma 12B, rzeczywistego encodera ani tool calling Gemmy
na Twojej V100. Nie ma zmierzonego wzrostu jakości lub szybkości tej wersji.
Bazowy pomiar użytkownika nadal wynosi około 48 tok/s, bez spekulacji.
Walidacja v100.4: 323 testy zaliczone, 63 pominięte; wszystkie hooki pre-commit
zaliczone. Mały encoder Bert sprawdza też rzeczywisty tokenizing i forward na CPU,
nie tylko mocki wyszukiwania.
Nowe testy v100.5 obejmują rzeczywiste zachowanie Trainer na małym GPT2 CPU:
najlepszy wczesny checkpoint przetrwał rotację, jego wagi zostały wczytane na końcu,
a najnowszy zachował stan wznowienia. Test współbieżności używa zastępczego procesu
treningowego i prawdziwego wątku pomocnika. Sprawdzamy również replay, budżety,
wspólną pamięć, odmowę regresji wszystkich rodziców i brak wykonania kodu bez izolacji.
Nie zmierzono tego workflow na V100 ani przepustowości pomocnika Qwen na Twoim CPU.
Bubblewrap jest niedostępny funkcjonalnie w środowisku walidacji (brak wymaganego
pliku proc przy tworzeniu namespace). Konstrukcję izolacji sprawdzono jednostkowo,
ale jej realne wykonanie, zwłaszcza GPU, wymaga sprawdzenia na docelowym Debianie.
Wynik v100.5: 345 zaliczonych, 63 pominięte. Hooki ruff, formatter i ty zaliczone
przy wskazaniu Python 3.11. Wielopokoleniową orkiestrację sprawdzono przez zastępcze
treningi/serwery: zachowuje wszystkie dopuszczone raporty rodziców, ustawia
poprzednika jako init/nauczyciela, wymusza replay i kończy po braku zwycięzcy.
Wynik v100.6: 382 zaliczone, 63 pominięte; hooki ruff, formatter i ty zaliczone.
Wynik v100.7: 404 zaliczone, 63 pominięte; te same hooki zaliczone. Nowe testy
sprawdzają aktualność źródeł, zmianę wyboru usług, filtrowanie cen, odmowę
finansowanego konta, wspólny limit konsultacji, brak płatnego fallbacku i
blokadę dalszych wywołań po naruszeniu kontroli kosztu. Transport jest zastępczy;
nie wykonano konsultacji z prawdziwym kluczem ani rejestracji kont.
Wynik v100.8: 416 zaliczone, 63 pominięte; hooki ruff, formatter i ty zaliczone.
Nowe testy obejmują rozdzielenie dzienników, maskowanie znanych sekretów,
współbieżny zapis, powiązania kroków, MTP8/MTP16 i kontynuowanie sweepu po błędzie.
Pomiary serwerów MTP są zastępcze; nie dowodzą przyspieszenia na V100.
Wynik v100.9: 443 zaliczone, 63 pominięte. Ruff i formatter oraz dotychczasowe
hooki pre-commit zaliczone. Testy paper sprawdzają chronologię decyzji, koszty,
partial fills, ryzyko, funding/likwidację, podatki zakładów, splity/dywidendy,
audyt, raporty i odmowę brakujących danych. Transport i modelowi researcherzy są
zastępczy; nie wykonano rzeczywistego paper tradingu ani testu V100 tej funkcji.
Wynik v100.10: 460 zaliczone, 63 pominięte. Ruff, formatter i dotychczasowe hooki
pre-commit zaliczone; zmienione moduły paper/continuous sprawdzone też przez ty
bez ignorowania błędów. Nowe testy sprawdzają bridge R&D do poolu, zwolnienie własnej
inferencji przed zastępczym treningiem, obserwację w czasie treningu, zachowanie
portfeli/checkpointów, blokadę dwóch loopów, niezmienność celu i błąd obserwatora.
Nie jest to rzeczywisty trening Gemmy na V100 ani dowód poprawy finansowej strategii.
Nowe testy sprawdzają rzeczywiste zmiany LoRA od automatycznie wyliczonych etykiet
oraz zmianę wszystkich parametrów zaufanej małej sieci CPU. Orkiestracja loopu
jest testowana na zastępczych serwerach; ocena nowych architektur sprawdza oddzielenie
gold odpowiedzi, dwie zgody na wspólny budżet i brak wykonania bez bubblewrap.
Nie zmierzono jeszcze tych nowych funkcji na Twoim Debianie/V100. Rzeczywista
izolacja wymaga testu docelowego; nie zastępujemy jej wykonywaniem kodu na hoście.
Ladder side network oraz automatyczna przebudowa i wymiana głównej Gemmy
pozostają kolejnymi eksperymentami. Granice badań: [FORGETTING_RESEARCH.md](FORGETTING_RESEARCH.md).
# Opcjonalny dodatkowy GPU do researchu

Wersja v100.15 obsługuje izolowanego researchera Ollama na Windows/RTX 3090,
również w czasie treningu V100. Instalacja i ograniczenia:
[RTX3090_HELPER.md](RTX3090_HELPER.md). Aktualizacja pakietu sama nie podłącza
zdalnego GPU; potrzebny jest udany smoke test i przygotowany profil z pełnym digestem.
