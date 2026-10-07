# V100 continual learning, v100.8

Nowy workflow ma osobne środowisko `venvs/v100-continual`, komendę
`bin/v100-continual`, profil `research/v100-continual.toml`, kopię pamięci SQLite
i port 8089. Instalator `install-continual-v100.sh` kopiuje zależności działającego
środowiska treningowego, zachowując Torch 2.6.0 CUDA 12.4. Nie aktualizuje `train`,
`memory-lab`, `v100-lab` ani istniejącego profilu i nie uruchamia treningu/serwera.
To pełne osobne środowisko, nie optymalizacja globalnych sterowników lub CUDA.
Kod v100.8 jest przygotowany do instalacji; sam commit nie aktualizuje serwera
Debian ani nie uruchamia procesu. Nowe zależności opcjonalne `selflab` obejmują
pytest i psutil; trening wymaga istniejącego Torch/Transformers/PEFT.

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
Nowe testy sprawdzają rzeczywiste zmiany LoRA od automatycznie wyliczonych etykiet
oraz zmianę wszystkich parametrów zaufanej małej sieci CPU. Orkiestracja loopu
jest testowana na zastępczych serwerach; ocena nowych architektur sprawdza oddzielenie
gold odpowiedzi, dwie zgody na wspólny budżet i brak wykonania bez bubblewrap.
Nie zmierzono jeszcze tych nowych funkcji na Twoim Debianie/V100. Rzeczywista
izolacja wymaga testu docelowego; nie zastępujemy jej wykonywaniem kodu na hoście.
Ladder side network oraz automatyczna przebudowa i wymiana głównej Gemmy
pozostają kolejnymi eksperymentami. Granice badań: [FORGETTING_RESEARCH.md](FORGETTING_RESEARCH.md).
