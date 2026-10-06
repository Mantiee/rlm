# Uczenie kolejnych rund: wymagania projektu

## Catastrophic forgetting

Obecny eksport pamięci zachowuje wszystkie zatwierdzone przykłady jako replay.
Bazowe wagi pozostają osobno, a nowy adapter jest kandydatem z możliwością powrotu
do wcześniejszej wersji. To ogranicza ryzyko, ale nie dowodzi zachowania umiejętności bazowych.

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
