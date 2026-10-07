# V100 continual learning, v100.4

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
Nie uruchomiono automatycznej ewolucji, selekcji bez oceny ani samopotwierdzania danych.
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
Walidacja tej wersji: 323 testy zaliczone, 63 pominięte; wszystkie hooki pre-commit
zaliczone. Mały encoder Bert sprawdza też rzeczywisty tokenizing i forward na CPU,
nie tylko mocki wyszukiwania.
Ladder side network, przebudowa Gemmy i inne metody pozostają eksperymentami do porównania,
nie istniejącą funkcją. Granice badań: [FORGETTING_RESEARCH.md](FORGETTING_RESEARCH.md).
