# Uczenie kolejnych rund: wymagania projektu

## Catastrophic forgetting

Wymaganie użytkownika: nie tracić zaakceptowanych wcześniejszych umiejętności.
Przegląd badań i granice możliwej ochrony opisuje [FORGETTING_RESEARCH.md](FORGETTING_RESEARCH.md).
Architektura ma zachowywać niezmienną bazę oraz niezmienne zaakceptowane adaptery
z jawnymi, wersjonowanymi ścieżkami uruchomienia. Nowe uczenie zmienia osobnego
kandydata i nie zastępuje automatycznie chronionego eksperta. Routing, prompt,
retrieval, tokenizer i cache należą do warunków działania chronionej ścieżki.
Nie oznacza to ogólnej gwarancji zerowej regresji całego systemu dla wszystkich wejść.

Obecny eksport pamięci zachowuje wszystkie zatwierdzone przykłady jako replay.
Bazowe wagi pozostają osobno, a nowy adapter jest kandydatem z możliwością powrotu
do wcześniejszej wersji. To ogranicza ryzyko, ale nie dowodzi zachowania umiejętności bazowych.
Obecne `init_adapter` ładuje wcześniejszy adapter jako trenowalną kopię: nowy kandydat
może zapominać. Bank chronionych ekspertów i ich routing są wymaganiami do wdrożenia,
nie istniejącą funkcją tej wersji.

Przed treningiem kolejnych rund trzeba zamrozić podział danych między rundami.
Obecne `load_records` izoluje powiązane źródła w pojedynczym zbiorze, lecz po dodaniu
nowych grup może zmienić wybór grup walidacyjnych. Rekordy użyte kiedyś do treningu
nie mogą później uchodzić za niezależny holdout. Potrzebne są osobne, niezmienne testy
starych umiejętności, nowych umiejętności, polskiego, kodu i cytowania źródeł.
Nowe dane powinny być mieszane ze sprawdzonymi wcześniejszymi przykładami i reprezentatywnymi
zadaniami ogólnymi. Proporcje replay oraz ewentualną regularizację dobieramy z regresji,
nie zakładamy, że większy rank LoRA automatycznie zapobiega zapominaniu.

## Gaming and exploitation

Ta wersja nie używa reward model ani autonomicznego RL. Przyjmuje tylko jawnie
zatwierdzony feedback człowieka; model nie zatwierdza własnych wygenerowanych odpowiedzi.
Publiczny benchmark szybkości MTP nie jest testem jakości ani materiałem treningowym.

Dalszy pipeline musi mieć niezależne testy, ocenę poprawności wyniku, a nie samego stylu,
oraz rejestr pochodzenia i decyzji weryfikatora. Testy jakości mają obejmować sprzeczne
źródła, brak dowodów, instrukcje ukryte w dokumentach i próby fałszywych cytowań.
Stałego ukrytego zestawu nie wolno wykorzystywać do strojenia kolejnych kandydatów:
do wyboru służy development set, do końcowej oceny oddzielny audit set.
Spadek loss, samodzielny werdykt modelu i szybkie tok/s nie zastępują tych testów.
Obecna instrukcja „źródła są danymi” nie daje pełnej ochrony przed prompt injection.

## Persistent memory

SQLite zachowuje oryginalne dokumenty, fragmenty, pozycje, identyfikatory, streszczenia
i zatwierdzony feedback. Odpowiedzi/runy oraz checkpointy są zapisane na dysku.
Restart serwera nie usuwa tej pamięci; do użycia trzeba ponownie wykonać retrieval.
To pamięć zewnętrzna, nie gwarancja, że wszystkie fakty są zapisane w wagach.

Po zmianie wag zmieniamy `runtime.model_version`, aby nie mieszać cache streszczeń.
Przed kolejnymi rundami dodajemy przetestowany backup SQLite przez SQLite backup API,
wersjonowany snapshot danych, deduplikację i obsługę korekt/sprzeczności źródeł.
Sam plik WAL i checkpoint na tym samym dysku nie są niezależnym backupem.

## Promocja modelu

Oryginalny model, adapter i konfiguracja są wersjonowane oddzielnie.
Każdy kandydat wymaga jakościowego porównania z bazą i poprzednią zaakceptowaną wersją,
zapisania regresji oraz świadomej promocji. Obecny fork niczego nie promuje automatycznie.
Nowe funkcje z powyższej listy pozostają wymaganiami do wdrożenia przed automatycznym
uczeniem kolejnych rund; nie są jeszcze w całości zaimplementowane.
