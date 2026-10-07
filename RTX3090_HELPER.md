# Pomocnik RTX 3090 na Windowsie

Opcjonalny, osobny serwer Ollama dla researchu/testerów A/B. Trening Gemmy,
główne profile i kontrola jakości pozostają na V100. Zdalny pomocnik nie
zmienia własnych wag w tej wersji i nie zastępuje niezależnej weryfikacji.
Nie dzielimy jednego modelu przez LAN ani nie używamy płatnej/cloud API.

## Windows, pierwszy krok

Uruchom PowerShell jako administrator
(tylko dodanie/usunięcie własnych reguł zapory wymaga tych uprawnień):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\start-rtx3090-helper.ps1
```

Skrypt używa `192.168.0.61:11435`, wpuszcza Debiana `192.168.0.68`, blokuje
pozostałe adresy na tym porcie, zachowuje istniejący serwer 11434 i ustawia
zmienne środowiska wyłącznie dla nowego procesu. Model
`qwen3.5:9b-q8_0` pobiera do `%USERPROFILE%\ai-v100-helper\models`.
Pobranie ma około 10 GB; nie jest to pomiar zużycia VRAM.

Pomocnik używa osobnej **Ollamy 0.40.0** z oficjalnych archiwów standalone
Windows amd64 i MLX CUDA, w `ai-v100-helper\runtime\ollama-0.40.0`.
Oba archiwa są przypięte do sum SHA-256 z oficjalnego wydania. Instalator nie
używa `ollama` znalezionej przez PATH, nie uruchamia globalnego instalatora
i nie aktualizuje aplikacji desktopowej. Wymagane 28 GiB wolnego miejsca;
archiwa pobrania są zachowane do ponownego użycia. Aktualny manifest Qwen3.5
zwrócił HTTP 412 przy Ollamie 0.21.2, dlatego ta wersja nie jest używana.
Po starcie sprawdzamy też wersję `/api/version`; rzeczywista zgodność modelu
i GPU nadal wymaga zakończenia smoke testu na Windowsie.

Test infrastruktury sprawdza JSON `2+2=4`, zakończenie odpowiedzi i rezydencję
GPU. Oddzielnie zachowujemy pierwotne trudniejsze działanie z wynikiem 10444,
teraz z budżetem 1024 tokenów. `direct_math_passed=false` jest jawnym wynikiem
jakościowym, a nie awarią transportu; prostszy test go nie zastępuje.
Nie oznacza to akceptacji wyników finansowych ani pominięcia późniejszych
testów jakości/admisji danych. Odpowiedź i liczniki smoke testu są w
`logs/smoke-json.response.json`, działanie w `logs/smoke-math.diagnostic.json`,
a pomiar modelu w `logs/smoke-gpu.loaded.json`. Zapisujemy liczbę znaków thinking,
bez jego treści. Przy ponownym uruchomieniu używamy zweryfikowanego runtime
i cache modelu; kontrola miejsca na dysku uwzględnia brakujące pobrania.

Jeden slot, kontekst początkowo 32768, Flash Attention i KV Q8. Rezerwacja
pozostałej pamięci jest wskazówką dla schedulera, **nie twardym limitem**.
Po odpowiedzi JSON testuje faktyczne `size_vram`, kontekst i pełne GPU
offload. Przy >12 GiB lub częściowym CPU offload przerywa, zatrzymuje wyłącznie
własny pomocnik i usuwa własne reguły zapory. W takim przypadku można
powtórzyć z `-Context 16384` albo `-Context 8192`. Test nie mierzy wszystkich
chwilowych pików ani jakości finansowego researchu. Pomocnik zostaje w VRAM
do zatrzymania, aby nie wygasł podczas długiego treningu V100.

Wynik: `helper-ready.json` z pełnym digestem, kontekstem, czasem i tok/s.
Logi: `ai-v100-helper\logs\server.stdout.log` i `server.stderr.log`.
Zatrzymanie bez usuwania pobranego modelu:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\start-rtx3090-helper.ps1 -Stop
```

Zatrzymanie obejmuje też osierocone `llama-server.exe` po zamknięciu serwera
Ollamy. Wybieramy wyłącznie procesy z wydzielonego katalogu runtime tego
pomocnika, sprawdzamy ścieżkę i czas uruchomienia przed zatrzymaniem i czekamy
na wyjście. To katalog przeznaczony dla jednego helpera, nie współdzielony
runtime innych serwerów. Ollama z `AppData\Local\Programs\Ollama` jest pomijana.

## Debian, po udanym teście Windows

Launcher v100.16 przyjmuje też `-Context 65536` i `-Context 131072`.
Dotychczasowy domyślny kontekst pozostaje 32768. Izolowany proces już otrzymuje
`OLLAMA_FLASH_ATTENTION=1` i `OLLAMA_KV_CACHE_TYPE=q8_0`; nowe okno nie oznacza
zmiany kwantyzacji wag Q8_0. Po zmianie trzeba zatrzymać wyłącznie helper,
uruchomić go z wybranym kontekstem i ponownie sprawdzić `/api/ps`, JSON i VRAM.
Pomiar nadal musi mieścić się w 12 GiB i pokazywać pełną rezydencję GPU.
To pomiar po smoke, nie twardy limit ani test pełnego okna. Nie dodajemy slotów
bez osobnego pomiaru, bo równoległe okna mogą zwiększać zużycie cache.

Przy zatrzymanej misji można zachować stary `researcher-rtx3090.json` pod inną
nazwą i użyć `prepare-remote-helper --context 131072` do przygotowania nowego
profilu z tym samym pełnym digestem. Istniejącego profilu komenda nie nadpisuje.

Po zainstalowaniu tej wersji forka przepisz **pełny digest i kontekst z wyniku
Windows**, nie skrócony identyfikator ze strony modelu:

```bash
~/ai-v100/bin/v100-continual prepare-remote-helper \
  --url http://192.168.0.61:11435 --context 32768 --digest PELNY_DIGEST_Z_WINDOWS
```

Komenda sprawdza dostępność, lokalną rodzinę Qwen byte-BPE/Q8, niezmienność
metadanych i rendererów, rezydencję GPU i limit pomiaru VRAM. Tworzy nowy
`research/researcher-rtx3090.json`; istniejącego pliku nie nadpisuje.
Nie zatrzymuje obecnej misji. Dopiero **następne mission-start** wybierze
ten profil. Nie trzeba powtarzać zgodnego, ukończonego baseline, ale
niedokończonego raportu nie wolno uznać za baseline. Zakończ bieżącą ocenę
przed kontrolowanym restartem misji.

Windows musi być włączony; zdalny helper nie jest uruchamiany ani zabijany
przez Debiana. Awaria/zmiana modelu daje jawny błąd, bez cichego przełączenia
na CPU. Usunięcie/przeniesienie wyłącznie profilu zdalnego, przy zatrzymanej
misji, przywraca wybór przygotowanego CPU researchera dla następnej misji.

## Budżet i logi

Ollama nie udostępnia dokładnego `/tokenize`. Dla tego konkretnego tekstowego
modelu używamy konserwatywnego oszacowania UTF-8 plus margines na szablon,
schema i role. Odrzucamy zbyt duże prośby przed wysłaniem, bez automatycznego
skracania źródeł. To oszacowanie, nie dokładny tokenizer ani dowód braku
każdego możliwego zachowania wewnętrznego Ollamy. Faktyczne liczniki tokenów
serwera sprawdzamy po odpowiedzi; limit odpowiedzi 1024, bez thinking.
Główny lokalny klient Gemmy nadal używa dokładnego tokenizera llama.cpp.

Helper używa dotychczasowych JSON tools, tej samej archiwizacji źródeł,
pamięci i recenzji parenta, również gdy V100 wykonuje backward. Nie przyjmuje
natywnych/tool/image wiadomości, zmiany modelu, chmury, przekierowań HTTP
ani zdalnego pull/delete/execute od kontrolera. Narzędzia działają na Debianie
w istniejących ograniczonych runnerach, nie jako dowolny Windows shell.

W logach aktywności `research/logs/activity/<data>/<A|B|controller>/tester/`
widać decyzje, wybrane narzędzia, błędy, rzeczywiste tokeny, tok/s,
`remote_vram_gib` i rodzaj oszacowania. Nie zapisujemy prywatnego toku
rozumowania. Snapshot misji: `research-helper.json`, połączenie:
`research-helper.log`. Pełne źródła pozostają na dysku; pomocnik nie zwiększa
jednego fizycznego okna Gemmy przez sumowanie VRAM obu kart.

Testy transportu/orchestracji wykonano z deterministycznymi serwerami
zastępczymi. Rzeczywiste Windows/3090, LAN, przepustowość i jakość wymagają
wyniku uruchomienia skryptu u użytkownika. Speculative decoding pozostaje
oddzielnym eksperymentem; ten pomocnik go nie włącza.

## Ograniczenia samego helpera, v100.20

Launcher ma domyślnie `-BatchTokens 64` (dozwolone 16/32/64), cztery wątki
CPU i odstęp między własnymi testami startowymi. Zaktualizowany klient Debiana
stosuje taki sam batch i cel 65% czasu aktywnych zapytań. Po zapytaniu trwającym
13 s rezerwuje 7 s przerwy przed następnym. Wynik oddaje od razu, więc V100
może pracować podczas tej przerwy. Lease i zapisany deadline są wspólne dla
klientów i procesów tego labu na Debianie, nie tylko jednego obiektu klienta.

Stare profile otrzymują domyślnie batch 64 i cel 65%. Opcjonalne pola resources
`helper_batch_tokens` i `helper_duty_percent` pozwalają zmniejszyć te wartości.
Nie trzeba zmieniać digestu, metadanych ani kontekstu 131072. Log zdarzenia
`helper-workload-reservation` podaje czas zapytania, zaplanowaną przerwę i
rzeczywiste oczekiwanie przed kolejnym zapytaniem. Potrzebna jest aktualizacja
launchera Windows i klienta Debiana; sam launcher nie ogranicza całego ruchu API.

To **nie jest twardy limit 65% GPU-Util, watów ani pików VRAM**. Jedno zapytanie
nadal może chwilowo wykorzystać pełną kartę. Inne aplikacje na RTX nie dostają
limitu i nie zmieniamy globalnej mocy karty, zegarów, sterownika ani TDR.
Nie uznajemy tego za potwierdzone rozwiązanie blackscreen; wcześniejsze awarie
przy dużym obciążeniu nadal wymagają diagnozy i kontrolowanego testu.

## Stabilne metadane i okno helpera, v100.21

Po aktualizacji klienta, przy zatrzymanej misji i działającym helperze, polecenie
`v100-continual repair-helper-metadata` migruje stary hash do schematu pomijającego
wyłącznie `modified_at`. Wymaga dotychczasowego digestu manifestu, runtime 0.40.0
i właściwego kontekstu/rezydencji GPU. Zapisuje backup profilu i pełny snapshot
metadanych w `research/helper-metadata/migration-*/`. Wagi, kontekst i cel 65%
czasu zapytań pozostają bez zmian. Zmiany szablonu lub parametrów po migracji
nadal zatrzymują pracę; profile już zmigrowane nie są ponownie przepinane.

Stary hash bez oryginalnego snapshotu nie pozwala ustalić przyczyny różnicy.
Migracja jawnie rejestruje ten brak, zamiast uznawać go za dowód zmiany samej daty.

Nowy launcher ukrywa puste okno natywnego serwera. Opcjonalne `-DebugLogs` włącza
debug wyłącznie w procesach tego helpera; `OLLAMA_DEBUG_LOG_REQUESTS` pozostaje
wyłączone. Log serwera: `$env:USERPROFILE\ai-v100-helper\logs\server.stderr.log`.
Podgląd w osobnym PowerShellu:

```powershell
Get-Content -LiteralPath "$env:USERPROFILE\ai-v100-helper\logs\server.stderr.log" -Tail 30 -Wait
```

To log transportu/serwera, nie raport uczenia ani wyników zarabiania. Te raporty
są dostępne przez `v100-continual mission-watch` i `mission-report` na Debianie.
Zmiany Windows sprawdzono w kodzie; rzeczywisty test PowerShell/GPU wymaga
uruchomienia launchera na komputerze użytkownika.
