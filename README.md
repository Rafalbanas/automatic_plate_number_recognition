# Plates — demonstrator ALPR

Publiczne demo rozpoznawania tablic rejestracyjnych dla
[`plates.banas.dev`](https://plates.banas.dev). Aplikacja wykrywa tablicę na
zdjęciu i odczytuje tekst lokalnie na VPS, bez zewnętrznej usługi OCR.

## Stan i architektura

Ścieżka produkcyjna używa dwóch małych modeli ONNX ładowanych raz przy starcie
jednego procesu:

- detektor `yolo-v9-t-384-license-plate-end2end` z projektu
  [Open Image Models](https://github.com/ankandrew/open-image-models);
- OCR `cct-xs-v2-global-model` z projektu
  [Fast Plate OCR](https://github.com/ankandrew/fast-plate-ocr);
- integracja przez [FastALPR](https://github.com/ankandrew/fast-alpr).

Wszystkie trzy projekty są publikowane na licencji MIT. Modele pobiera jednostka
setup i przechowuje w `/var/cache/plates-web/.cache`; nie są commitowane.
Detektor jest modelem ogólnym dla tablic wielu państw, a OCR obsługuje alfabet
łaciński i cyfry. Nie jest to walidator polskich numerów i nie należy używać
wyniku jako podstawy decyzji administracyjnej lub bezpieczeństwa.

Repozytorium zawiera też niezależny, powtarzalny pipeline treningu detektora.
Trening nie jest potrzebny do działania bieżącego demo, ale pozwala odtworzyć
własne wagi i rzetelnie ocenić je na niewidzianym zbiorze testowym.

### Zweryfikowany stan wdrożenia (2026-10-01)

- HTTPS, health check i prawdziwy upload działają pod `plates.banas.dev`;
- na VPS przykład `TEST123` został wykryty i odczytany poprawnie w 92 ms;
- przykład `DEMO456` został wykryty w 94 ms, ale OCR zwrócił `DEM0456` — to
  rzeczywisty błąd `O/0`, a nie poprawiony lub przypisany z góry wynik;
- biały obraz zwrócił pustą listę tablic w 91 ms, fałszywy JPEG HTTP 400,
  plik 5 MiB HTTP 413, a test sześciu równoległych wywołań: jedno HTTP 200 i
  pięć HTTP 429;
- po inferencji jednostka WWW zajmowała ok. 123 MiB RAM i 0 B swapu przy
  limitach 640/768 MiB (`MemoryHigh`/`MemoryMax`).
- przygotowanie osiągnęło 267,1 MiB w pomiarze co 5 s (minimum
  `MemAvailable`: 2455,3 MiB); smoke test trwał 53,6 s i zapisał checkpointy w
  `/var/lib/plates-training/runs/smoke/weights`;
- trening ukończył 40/40 epok w 2 h 57 min, zapisał `last.pt`, `best.pt` oraz
  checkpoint każdej epoki. Szczyt całego zadania wyniósł 1030,9 MiB, minimum
  `MemAvailable` 1971,4 MiB, swap pozostał równy 0 B;
- końcowa ocena na 137 niewidzianych obrazach testowych: precision 0,8044,
  recall 0,7381, mAP50 0,7085 i mAP50-95 0,4846. Ocena trwała 18,955 s,
  średnio 138 ms/obraz razem z narzutem uruchomienia i metryk; raport znajduje
  się w `/var/lib/plates-training/evaluation.json`.

Te dwa obrazy demonstracyjne nie są zbiorem statystycznym. Powyższe metryki
dotyczą wyłącznie detekcji tablic. Zbiór Open Images nie ma tekstowych etykiet
OCR, więc nie wolno z niego wyliczać dokładności odczytu numerów.

## Ustalenia z audytu pierwotnego projektu

- `scripts/trening.ipynb` zakładał Colab, GPU, `yolo26s`, obrazy 1280 px,
  automatyczny batch oraz 400 epok. Nie jest to konfiguracja odpowiednia dla
  VPS z jednym dostępnym vCPU i 4 GB RAM.
- W repozytorium nie było wag `.pt`, `.pth` ani `.onnx`.
- `LPRNet_Pytorch` był pustym gitlinkiem do commita bez wpisu w `.gitmodules`.
  Kod nie był możliwy do pobrania jako submodule. Ścieżka produkcyjna nie używa
  LPRNet.
- 195 plików w `lpr_data/images` to wycinki 94×24. Etykiety w `train.txt` i
  `val.txt` zawierają literalne `PLATE`, a nie numery. Nie nadają się ani do
  detekcji na pełnych zdjęciach, ani do treningu OCR.
- Skrypty projektu wskazywały Kaggle dataset
  `piotrstefaskiue/poland-vehicle-license-plate-dataset`. Zbiór jest dostępny i
  zawiera pełne zdjęcia oraz CVAT XML (ok. 604 MB), lecz Kaggle oznacza licencję
  jako „Other”, a opis nie podaje warunków wykorzystania. Pipeline nie pobiera
  go automatycznie, bo nie wolno domyślać zgody na trening/redystrybucję.

## Dane treningowe i rozdzielenie zbiorów

Zamiennikiem jest wyłącznie oznaczona część klasy `Vehicle registration plate`
z Open Images, pobierana z publicznego
[mirrora Hugging Face](https://huggingface.co/datasets/shravya11/vehicle-registration-plate).
Katalog `harvested_plates` o niejasnym pochodzeniu jest celowo pomijany.

- karta mirrora deklaruje całość jako CC BY 4.0; oficjalna dokumentacja Open
  Images podaje CC BY 2.0 dla obrazów i CC BY 4.0 dla adnotacji. Open Images
  zastrzega, że nie gwarantuje statusu licencji każdego obrazu z osobna;
- domyślnie 600 obrazów z oryginalnego `train` oraz 240 z `validation`;
- train pozostaje train; grupy z validation są deterministycznie dzielone na
  val/test;
- identyczne SHA-256 i bliskie dHash (odległość Hamminga ≤ 3) są grupowane, a
  cała grupa trafia do jednego splitu;
- manifest z ID, hashem i przypisaniem znajduje się w
  `/var/lib/plates-training/dataset/manifest.json`.

To ogranicza przeciek duplikatów i bardzo podobnych kadrów, ale Open Images nie
publikuje identyfikatorów pojazdów/serii. Nie można więc zagwarantować pełnego
rozdzielenia według fizycznego pojazdu.

## Uruchomienie lokalne

Wymagany jest Python 3.10–3.12.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-web.txt
.venv/bin/python scripts/warm_models.py
.venv/bin/python -m webapp.server
```

Serwer nasłuchuje domyślnie na `127.0.0.1:20145`.

Testy walidacji:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Przygotowanie danych, trening i wznowienie

Na serwerze wszystkie procesy potomne przygotowania/treningu są zamknięte w
cgroup systemd. Jednostki mają:

```ini
MemoryHigh=1536M
MemoryMax=2048M
MemorySwapMax=0
OOMPolicy=stop
Restart=no
CPUQuota=80%
```

Trening zaczyna się od `yolo11n.pt`, `batch=1`, `workers=0`, `cache=False`,
`imgsz=416`, CPU. Najpierw wykonywany jest smoke test 1 epoki na 3% danych,
potem trening właściwy do 40 epok. Checkpoint jest zapisywany co epokę.
Jednostka instaluje jawnie `torch==2.14.1+cpu` z oficjalnego indeksu PyTorch;
nie pobiera bibliotek CUDA.

```bash
sudo systemctl start plates-prepare.service
sudo systemctl start plates-training.service
```

Jednostka automatycznie wznawia run z
`/var/lib/plates-training/runs/plate-detector/weights/last.pt`. Po treningu
uruchamia ocenę na niewidzianym splicie `test` i zapisuje prawdziwe metryki do
`/var/lib/plates-training/evaluation.json`.

Status, logi, pamięć i zatrzymanie:

```bash
systemctl status plates-training.service
journalctl -fu plates-training.service
cat /var/lib/plates-training/resource-status.json
cat /var/lib/plates-training/prepare-resource-status.json
systemctl show plates-training.service -p MemoryCurrent -p MemoryPeak -p MemoryHigh -p MemoryMax -p MemorySwapCurrent
systemd-cgtop --depth=2
sudo systemctl stop plates-training.service
```

`scripts/resource_guard.py` obejmuje cały pipeline przygotowania oraz trening i
sprawdza pamięć co 5 sekund. Jeśli efektywne
`MemAvailable` pozostaje poniżej 768 MB przez 30 sekund, wysyła SIGTERM do całej
grupy procesu i nie uruchamia go ponownie. Nie tworzy ani nie włącza swapu.

Ręczne odtworzenie (po przygotowaniu danych):

```bash
/var/lib/plates-training/venv/bin/python scripts/train_detector.py \
  --data /var/lib/plates-training/dataset/dataset.yaml \
  --project /var/lib/plates-training/runs \
  --name plate-detector --epochs 40 --resume
```

Ultralytics i jego wagi są dostępne na AGPL-3.0; wygenerowanych checkpointów nie
należy przenosić do zamkniętego produktu bez sprawdzenia obowiązków licencyjnych.

## Prywatność i limity aplikacji

Przeglądarka wysyła surowe bajty JPEG/PNG/WebP, nie formularz multipart. Serwer
odczytuje ciało do pamięci procesu, sprawdza MIME, sygnaturę/dekodowanie,
pojedynczą klatkę, wymiary oraz liczbę pikseli. Aplikacja nie zapisuje uploadu,
wyniku OCR ani nazwy pliku. Endpoint `/api/analyze` ma wyłączony access log
Nginx; log aplikacji zawiera tylko losowe ID żądania, status i czas.

Limity:

- 4 MiB pliku (aplikacja i Nginx);
- JPEG/PNG/WebP, pojedyncza klatka;
- maks. 12 megapikseli i 4096 px na bok;
- jedna inferencja równocześnie, kolejne żądanie otrzymuje HTTP 429;
- 15 s po stronie aplikacji, 18 s w Nginx;
- 6 żądań/min/IP z burst 2;
- usługa WWW: `MemoryHigh=640M`, `MemoryMax=768M`, brak swapu.

Nginx ma `proxy_request_buffering off` dla uploadu, więc nie buforuje ciała do
pliku tymczasowego. `PrivateTmp=true` izoluje ewentualne pliki robocze bibliotek.
Nie deklarujemy „wyłącznie RAM”: system operacyjny i biblioteki runtime mogą
tworzyć własne metadane/cache modeli, ale nie zawierają one zdjęć użytkownika.
Publiczny rekord DNS jest obecnie proxy'owany przez Cloudflare, więc upload
przechodzi również przez jego infrastrukturę zgodnie z konfiguracją i zasadami
tego konta. Cloudflare nie wykonuje OCR, ale repozytorium nie może zagwarantować
jego polityki buforowania. Informacja ta jest widoczna także w interfejsie.
Sprawdzony publiczny endpoint API zwraca `Cache-Control: no-store` i
`CF-Cache-Status: DYNAMIC`; HTML także ma `no-store`. Publiczne CSS, JavaScript
i przykłady mogą być cache'owane (Cloudflare stosował do 4 godzin), ponieważ
nie zawierają danych użytkowników.

## Wdrożenie

Pliki produkcyjne:

- `/opt/plates-web` — kod tylko do odczytu;
- `/var/lib/plates-web/venv` — środowisko aplikacji;
- `/var/cache/plates-web` — publiczne modele ONNX/cache;
- `/var/lib/plates-training` — dane, venv, runy i checkpointy treningu;
- `/etc/systemd/system/plates-web.service`;
- `/etc/systemd/system/plates-{prepare,training}.service`;
- `/etc/nginx/sites-available/plates.conf`;
- `/etc/nginx/conf.d/plates-limits.conf`.

Przed każdym przeładowaniem:

```bash
sudo nginx -t && sudo systemctl reload nginx
```

Health check:

```bash
curl -fsS https://plates.banas.dev/healthz
```

## Przykłady i ograniczenia

`webapp/static/examples/test123.png` i `demo456.png` są oryginalnymi,
wygenerowanymi na potrzeby projektu obrazami z fikcyjnymi numerami. Nie są
danymi treningowymi. Są celowo analizowane przez ten sam endpoint co upload —
UI nie zawiera przypisanych z góry wyników.

Najczęstsze ograniczenia: bardzo mała tablica, silna perspektywa, rozmycie,
odblaski, zasłonięcie, tablice dwurzędowe i pomyłki podobnych znaków (`O/0`,
`I/1`, `B/8`). Model może wykryć tablicę bez wystarczająco pewnego odczytu;
wtedy UI pokazuje komunikat zamiast zgadywać.
