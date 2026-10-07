# V100 continual learning, v100.5

Nowy workflow ma osobne środowisko `venvs/v100-continual`, komendę
`bin/v100-continual`, profil `research/v100-continual.toml`, kopię pamięci SQLite
i port 8089. Instalator `install-continual-v100.sh` kopiuje zależności działającego
środowiska treningowego, zachowując Torch 2.6.0 CUDA 12.4. Nie aktualizuje `train`,
`memory-lab`, `v100-lab` ani istniejącego profilu i nie uruchamia treningu/serwera.
To pełne osobne środowisko, nie optymalizacja globalnych sterowników lub CUDA.

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

`train DATASET` trenuje osobnego kandydata LoRA na zatwierdzonym feedbacku. Eksport
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
Nie jest to samoczynne generowanie prawdziwych odpowiedzi ani przyjmowanie
niezweryfikowanych danych od pomocników do treningu.

`prepare-researcher` przygotowuje osobny profil `research/researcher-cpu.toml`
na porcie 8090. Pobiera oficjalny Qwen3-0.6B Q8_0 GGUF, przypina rewizję i sprawdza
SHA256 artefaktu. Używa tego samego skompilowanego llama-server, `gpu_layers=0`,
4 wątków, kontekstu 2048 i nice=10. Kontroler wyłącza widoczność CUDA dla tego
procesu. Rozmiar pliku wag około 639 MB nie obejmuje KV cache i buforów procesu.

`run-duel DIR --suite DEV_SUITE --baseline-report BASELINE --researcher-profile CPU_PROFILE`
trenuje A, eksportuje Q6_K, ocenia A, a następnie robi to samo z B. Na V100 jest
jedna duża gałąź naraz. Mały model CPU pozostaje uruchomiony równolegle z treningiem
GPU i czyta aktualne, kompletne wiersze metryk. Planner może zlecić do 3 zadań na
gałąź jako tester/researcher/critic. To różne role korzystające z jednego serwera
0.6B, nie niezależnie trenowane sieci potomne. Pomocnicy proponują hipotezy i testy;
nie wykonują dowolnego kodu ani nie są sędziami jakości. Błąd/timeout pomocnika
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
oddzielny. Zmiana architektury i trening sieci pomocników pozostają kolejnymi
eksperymentami, nie gotowymi funkcjami tej wersji.

`evolve POOL --output DIR --suite DEV_SUITE --baseline-report BASELINE --researcher-profile CPU_PROFILE`
automatyzuje 1-4 takich pokoleń (domyślnie 2). Po zakończeniu pokolenia unloaduje
GPU, a kolejną parę planuje jego zwycięzca; używa jego adaptera jako inicjalizacji
i zamrożonego nauczyciela. Zachowuje poprzedni pool jako obowiązkowy replay oraz
raporty wszystkich wcześniejszych dopuszczonych gałęzi jako bramki regresji.
Przy remisie kontynuuje jawnie A, zachowując również B; przy braku dopuszczonego
zwycięzcy kończy cykl. `evolution.json` zapisuje wyniki każdego pokolenia.
Nie podmienia serwowanego modelu, nie promuje eksperta i nie zatwierdza odpowiedzi
pomocników. To ograniczona ewolucja adapterów/hiperparametrów, bez automatycznej
przebudowy architektury ani krzyżowania wag. Cross breeding jest osobną komendą.
Pełny wielopokoleniowy cykl nie ma jeszcze automatycznego wznowienia po awarii.

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
Ladder side network, przebudowa Gemmy i inne metody pozostają eksperymentami do porównania,
nie istniejącą funkcją. Granice badań: [FORGETTING_RESEARCH.md](FORGETTING_RESEARCH.md).
