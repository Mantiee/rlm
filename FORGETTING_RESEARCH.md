# Uczenie bez utraty wcześniejszych umiejętności: przegląd i propozycje

Stan badań sprawdzony 6 października 2026 UTC (7 października w Polsce).
Dokument opisuje propozycje projektu i wymagania,
nie wdrożoną gwarancję. Nie uruchamia treningu ani niczego na serwerze użytkownika.

## Co rzeczywiście możemy chronić

Nie znalazłem metody dającej ogólną gwarancję zerowego spadku jakości na wszystkich
wcześniejszych zadaniach dla jednego stale aktualizowanego LLM. Wynik na skończonym
benchmarku nie jest taką gwarancją. Nawet poprawa średniej może ukryć utratę pojedynczej
umiejętności.

Możemy natomiast chronić wcześniejsze funkcje przez izolację: niezmienna baza,
niezmienne zaakceptowane adaptery i zachowany sposób ich uruchamiania. Nowy trening
zmienia wyłącznie nowego kandydata. Jest to decyzja projektowa inspirowana izolacją
parametrów, a nie twierdzenie, że dowolny system z LoRA nie zapomina.

Trzeba odróżnić trzy wymagania:

| Wymaganie | Zakres ochrony |
| --- | --- |
| Nie nadpisywać starej wiedzy w wagach | Niezmienne artefakty, hashe, osobne katalogi, brak optymalizacji chronionych parametrów. |
| Zachować działanie starego eksperta | Niezmienna baza i adapter, tokenizer, szablon, prompt, kontekst, routing oraz wykonanie. Losowanie i zmiany numeryczne ograniczają identyczność odpowiedzi. |
| Cały rozwijający się system nigdy nie odpowie gorzej | Brak ogólnej gwarancji: wybór eksperta, retrieval i nowe kombinacje zadań mogą się pomylić. |

Nie wolno przedstawiać backupu jako rozwiązania aktywnego zapominania. Backup daje
możliwość odzyskania. Dostępna, niezmienna ścieżka eksperta zachowuje wcześniejszy model.

## Co pokazują badania

### Izolacja parametrów: Progressive Neural Networks

[Rusu et al., 2016](https://arxiv.org/abs/1606.04671) zamrażają wcześniejsze kolumny
sieci i dodają nowe, korzystające z wcześniejszych reprezentacji. To mocny wzorzec
ochrony starych parametrów. Eksperymenty dotyczą uczenia ze wzmocnieniem, nie Gemmy.
Kopiowanie całego modelu 12B dla każdego zadania byłoby kosztowne; nasza propozycja
zastępuje pełne kolumny małymi adapterami, zachowując odrębne wersje wykonania.

### O-LoRA: przydatna, ale nie zerowe zapominanie

[Wang et al., EMNLP 2023](https://aclanthology.org/2023.findings-emnlp.715.pdf)
uczą nowe adaptery z karą za nieortogonalność wobec wcześniejszych podprzestrzeni.
W tabeli 3 Alpaca-LoRA ma MMLU 37,5; zwykłe continual learning 23,3; O-LoRA 33,6.
To znaczna poprawa względem zwykłego treningu, lecz nadal spadek o 3,9 punktu.
Sumowanie wszystkich aktualizacji może zmieniać działanie starych zadań mimo
zamrożenia plików wcześniejszych adapterów.

**O-LoRA z tej pracy to inna metoda niż OLoRA inicjalizująca adapter przez QR.**
Ustawienie PEFT `init_lora_weights="olora"` nie wdraża tego algorytmu continual learning.

### InfLoRA: ostrożnie z przenoszeniem wyników

[Liang i Li, CVPR 2024](https://arxiv.org/html/2404.00228v3) dobierają przestrzeń
nowych aktualizacji z uwzględnieniem wcześniejszych reprezentacji. Eksperymenty
wykorzystują ViT-B/16 i klasyfikację obrazów. Ochrona opiera się na przybliżeniu
wcześniejszej przestrzeni; autorzy omawiają błąd tego przybliżenia. Nazwa
„Interference-Free” nie stanowi dowodu zerowej regresji dla Gemmy 12B. Port i jego
koszt na V100 wymagałyby osobnego eksperymentu.

### CorDA: obiecujący eksperyment dla nowego kandydata

[Yang et al., NeurIPS 2024](https://arxiv.org/html/2406.05223v3) wykorzystują
statystyki kontekstu do dekompozycji wag. Tryb KPM zachowuje główne komponenty
i dostraja pozostałe. Po treningu matematyki Gemma-2-9B ma NQ-open 12,85 przed
treningiem, 9,28 z LoRA oraz 10,17 z CorDA KPM. Ochrona jest częściowa.
To inna Gemma niż nasz model. Obliczanie kowariancji i dekompozycji wymaga pomiaru
pamięci i czasu. Dekompozycję wykonujemy wyłącznie na kopii kandydata, z poprawnym
eksportem względem oryginalnej bazy.

### FOREVER: ciekawy replay z 2026 roku

[Feng et al., ACL 2026](https://aclanthology.org/2026.acl-long.1144/)
wyznaczają harmonogram replay według wielkości zmian parametrów przez optymalizator,
zamiast wyłącznie numeru kroku, i adaptują siłę regularizacji. Badania obejmują modele
0,6B–13B i trzy benchmarki continual learning. To kandydat do ograniczania regresji
w nowym adapterze, bez obietnicy zerowego zapominania. Własny prosty scheduler nie
powinien być oznaczany jako reprodukcja FOREVER bez weryfikacji całego algorytmu.

### Replay i gradienty: znaczenie doboru danych

[Abbes et al., PMLR 2026](https://proceedings.mlr.press/v330/abbes26a.html) badają
replay i efektywną wersję gradient alignment przy dalszym pretrainingu LLaMA na
bardzo dużych korpusach językowych. Wynik uzasadnia sprawdzenie tych składników,
nie gotową receptę na nasz mały zbiór i V100.

[Mahdaviyeh et al., PMLR 2026](https://proceedings.mlr.press/v330/mahdaviyeh26a.html)
pokazują przypadki, w których replay może zwiększyć zapominanie, oraz brak prostej
monotonicznej zależności od rozmiaru bufora. Ich teoria nie jest dowodem, że replay
zaszkodzi Gemmie. Jest powodem, by oceniać skład i proporcje na niezależnych zadaniach.

### LoRA, EWC i edycja faktów

[Biderman et al., TMLR 2024](https://arxiv.org/html/2405.09673v2) pokazują kompromis
między uczeniem a zachowaniem wiedzy: LoRA często zapomina mniej, ale też może uczyć
się mniej. Większy rank nie jest automatyczną ochroną.

[EWC, Kirkpatrick et al.](https://arxiv.org/abs/1612.00796) ogranicza zmiany ważnych
parametrów przez regularizację. Skończona kara nie zamraża funkcji. Przechowywanie
jednej statystyki FP32 dla około 12 miliardów parametrów to około 48 GB; dlatego
ewentualną regularizację rozważamy na adapterze, nie całej bazie.

[AlphaEdit, Fang et al.](https://arxiv.org/html/2410.02355v3) projektuje edycje do
przestrzeni zerowej reprezentacji chronionej wiedzy. Ochrona dotyczy reprezentowanej
przestrzeni i założeń edycji, nie wszystkich zachowań autoregresywnego modelu.
To ciekawy kierunek dla korekt faktów, nie podstawowe rozwiązanie dla ogólnego uczenia.

## Embeddingi i przebudowa modelu zamiast samej LoRA

Embedding jest wektorową reprezentacją tekstu, użyteczną do wyszukiwania podobnych
znaczeniowo fragmentów. [Dokumentacja Sentence Transformers](https://sbert.net/examples/sentence_transformer/applications/semantic-search/README.html)
opisuje wyszukiwanie poprzez porównanie embeddingów zapytania i dokumentów.
Proponujemy indeks semantyczny uzupełniający wyszukiwanie słów, zawsze z zachowaniem
oryginału i identyfikatora wersji enkodera. Zmiana enkodera wymaga zgodnego ponownego
indeksowania. Sam embedding nie zapisuje bezstratnie dokumentu ani nie chroni wag
Gemmy przed regresją. Odnalezienie przykładu starego zadania nie gwarantuje, że model
nadal potrafi je wykonać.

[Memorizing Transformers](https://research.google/pubs/memorizing-transformers/)
pokazują bardziej wewnętrzną pamięć: kNN nad parami klucz–wartość reprezentacji
transformera. To odrębne rozwiązanie od zwykłego RAG i wymaga integracji z architekturą.

LoRA nie jest jedyną możliwością. [Ladder Side-Tuning](https://arxiv.org/abs/2206.06522)
trenuje małą sieć korzystającą z reprezentacji pośrednich zamrożonej sieci bazowej,
bez backpropagation przez bazę. [Ladder Up, Memory Down](https://arxiv.org/pdf/2512.14237)
(wersja z sierpnia 2026) rozwija ten kierunek dla LLM, opisuje xLadder i mniejszy
szczyt pamięci w eksperymentach. Badano m.in. Qwen, Llama i OPT; nie potwierdzono
naszej Gemmy na V100. Praca dotyczy efektywnego dostrajania, nie dowodu zerowego
zapominania. Nowa sieć boczna nadal może zmieniać wynik, a jej kolejne aktualizacje
mogą tracić wcześniejsze umiejętności.

[Titans](https://arxiv.org/html/2501.00663v1) uczą wagi modułu pamięci podczas
przetwarzania sekwencji. To pasuje do ambicji uczenia przy napływie informacji,
ale obejmuje mechanizm kontrolowanego zapominania pamięci i nie rozwiązuje wymogu
zachowania wszystkich umiejętności. [RMT](https://arxiv.org/abs/2207.06881) przenosi
trenowane reprezentacje pamięci między segmentami kontekstu. Długi kontekst i
ochrona umiejętności podczas kolejnych treningów pozostają odrębnymi problemami.

Można też dostrajać bezpośrednio wybrane macierze albo bloki Gemmy. Pełny standardowy
mixed-precision AdamW wymaga według [dokumentacji Transformers](https://huggingface.co/docs/transformers/v5.1.0/model_memory_anatomy)
około 18 bajtów na parametr plus aktywacje: około 216 GB dla 12B. Inne optymalizatory,
offload i trening podzbioru parametrów zmieniają ten rachunek. To ograniczenie
standardowego pełnego treningu, nie dowód, że wszelka przebudowa jest niemożliwa.
Offload na tym serwerze z 31 GiB RAM i starszym CPU również wymaga ostrożnego pomiaru.

Dodanie nowej gałęzi z możliwością pominięcia daje eksperymentalną rozbudowę modelu
i zachowaną ścieżkę oryginalną. Nie daje automatycznie braku regresji przy włączonej
gałęzi. Zmiana architektury wymaga treningu integracji oraz wsparcia inferencji;
nie zakładamy, że istniejący szybki backend GGUF uruchomi ją bez modyfikacji.
Wybór LoRA lub sieci bocznej pozostaje otwarty. Przed decyzją potrzebny jest mały
porównawczy eksperyment jakości, pamięci i szybkości.

## Proponowana architektura na V100 32 GB

Poniższe decyzje są wnioskami projektowymi z badań. Nie zostały jeszcze zmierzone
na tym sprzęcie ani w pełni wdrożone w forku.

1. **Baza i zatwierdzeni eksperci są niezmienni.** Wersja to manifest z hashami
   modelu, adaptera, tokenizera, szablonów i konfiguracji wykonania. Oryginalny Q6
   zachowuje własną ścieżkę. Żaden trening nie może pisać w katalogu zaakceptowanego eksperta.
2. **Każda runda tworzy kandydata.** Może zaczynać od kopii zaakceptowanego adaptera;
   trening tej kopii może zapominać. Nie zastępuje ona automatycznie poprzednika.
   Jej wagi faktycznie uczą się nowych umiejętności. Uczenie i serwowanie nie działają
   jednocześnie na tych samych obiektach parametrów.
3. **Jawny wybór eksperta.** Chroniony identyfikator umiejętności wskazuje na stałą
   wersję. Nowe zadanie dostaje nową wersję. Automatyczny router może pomagać, ale
   jego decyzja wymaga osobnej oceny i jawnej możliwości uruchomienia starego eksperta.
   Globalne przełączanie adaptera między równoczesnymi żądaniami jest niedopuszczalne.
4. **Jedna baza na GPU i ograniczona liczba aktywnych adapterów.** Pozostałe są
   przechowywane na dysku/RAM. Docelowo mierzymy natywne LoRA w llama.cpp; najpierw
   sprawdzamy konwersję dla dokładnej architektury Gemmy i zgodność wyników.
   KV cache musi być przypisany do wersji eksperta lub unieważniony przy zmianie.
   Nie zakładamy braku narzutu ani zachowania dotychczasowych 48 tok/s.
5. **Oryginalne źródła i korekty pozostają w pamięci zewnętrznej.** Wersjonujemy
   dokumenty, pochodzenie, zatwierdzenia i snapshoty. Retrieval i streszczenie nie
   zastępują oryginału. Zmiana kontekstu jest zmianą warunków odpowiedzi starego eksperta.

[S-LoRA](https://arxiv.org/abs/2311.03285) jest inspiracją dla współdzielenia bazy
i obsługi wielu adapterów. Nie dowodzi zgodności swojego backendu z naszym sm70
ani wydajności natywnych adapterów Gemmy. [PEFT](https://huggingface.co/docs/peft/package_reference/peft_model)
udostępnia ładowanie i wybieranie adapterów; samo API nie wdraża ochrony starej ścieżki.

Kosztem ochrony jest rosnący bank adapterów, testów i metadanych. Nie otrzymujemy
jednego małego, wiecznie aktualizowanego zestawu wag z gwarantowanym brakiem regresji.
Łączenie adapterów albo destylacja banku do jednego modelu tworzy nowego kandydata
i ponownie wymaga oceny. Stare wersje pozostają dostępne.

## Jak stwierdzić, czy trening idzie dobrze

Logujemy training loss, validation loss, learning rate, grad norm, liczbę tokenów,
tok/s, pamięć GPU, checkpoint i pochodzenie danych. Najważniejszy jest oddzielny wynik
każdej chronionej umiejętności: polski, kod z wykonywalnymi testami, rozumowanie,
cytowanie, wcześniejsze zadania i poprawne odmawianie odpowiedzi bez dowodów.

Podział grup źródeł train/development/audit utrwalamy raz. Przykład wcześniej użyty
w treningu nie staje się później niezależnym testem. Development służy do wyboru
kandydatów; audit nie może stale sterować tuningiem. Potrzebne są też nowe niezależne
próby, aby nie dopasowywać systemu do jednego publicznego zestawu.

Wynik zapisujemy per zadanie i per przykład, ze zmianą względem zaakceptowanego
eksperta, rozrzutem i warunkami generacji. Kandydat z regresją chronionej umiejętności
nie zastępuje jej ścieżki. Brak wykrytej regresji oznacza zgodność na sprawdzonych
próbach, nie dowód dla wszystkich możliwych wejść. Sam spadek loss nie wystarcza.

## Stan kodu i kolejność prac

Obecny `training.py` zapisuje kandydatów i pełne checkpointy oddzielnie. Jednak
`init_adapter` wczytuje adapter jako `is_trainable=True`: nowa kopia może zapominać.
`load_records` przelicza podział grup po zmianie zbioru. Bank niezmiennych ekspertów,
chroniony routing i stały rejestr splitów nie są jeszcze zaimplementowane.

Proponowana kolejność: rejestr ekspertów i stałych podziałów, zakaz nadpisywania,
logowanie wersji oraz testy ochrony ścieżek. Następnie walidujemy konwersję i obsługę
LoRA w natywnym serwerze oraz wykonalność sieci bocznej. Potem porównujemy małe rundy
treningu: LoRA z dobranym replay, sieć boczna i regularizacja względem poprzednika;
dalej ewentualnie O-LoRA/CorDA/FOREVER.
Nie dokładamy wszystkich naraz, bo nie da się ustalić, co rzeczywiście pomaga.

Potwierdzony baseline sprzętowy to controller v100.2, oryginalna Gemma Q6
i około 48,23 tok/s. MTP jest
osobnym eksperymentem inferencji; nie rozwiązuje zapominania przy treningu.
