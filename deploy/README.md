# Wdrożenia aplikacji Plates

Pipeline wzorowany na `Rafalbanas/banasdev` i `Rafalbanas/weronikarafal`.
Przesyła tylko kod aplikacji i publiczne przykłady. Nie przesyła modeli,
zdjęć Kaggle, danych użytkowników, venv ani skryptów wykonywanych jako root.
Nie zmienia modelu detekcji ani OCR. Dane i wagi nadal pozostają na `/storage`.

## Bezpieczeństwo i granice

- CI testuje i tworzy deterministyczny ZIP z jawną listą plików.
- CD pobiera dokładnie ten artefakt. Nie buduje innej paczki na VPS.
- Tylko `main`, nigdy PR; domyślnie CD jest **wyłączone**. Wymaga zmiennej
  repozytorium `PLATES_CD_ENABLED=true` i środowiska GitHub `production`.
  Zalecana jest ochrona `main` oraz zatwierdzanie środowiska przez człowieka.
- Oddzielny klucz SSH ma forced command: tylko `status` i `deploy SHA DIGEST`.
  Nigdy nie używaj prywatnego klucza administratora ani klucza stacjonarki.
- Root-owned odbiornik sprawdza SHA-256, rozmiary, dokładną listę plików,
  typy plików, składnię Python i zatwierdzony hash `requirements-web.txt`.
  Nie wykonuje kodu aplikacji jako root i nie uruchamia `pip`.
- Odbiornik przełącza symlink `current`, restartuje tylko `plates-web.service`
  i sprawdza `/healthz`: gotowość modeli, SHA commita i hash HTML z procesu.
  Niepowodzenie po przełączeniu powoduje powrót do poprzedniego wydania,
  restart i ponowne sprawdzenie jego zdrowia. Nieudany rollback jest błędem.
- Wdrożenia są blokowane po stronie CI (concurrency) i VPS (flock).
  Workflow odrzuca stary commit, który nie jest już HEAD `main`, i cofnięcie
  względem aktualnie wdrożonej historii Git.

Kod odbiornika, wrapper i pliki systemd aktualizuje administrator po przeglądzie,
poza automatycznym CD. Sam klucz CD nie może zmienić tych plików ani zależności.

## Jednorazowe przygotowanie VPS — ręcznie, w oknie wdrożeniowym

Nie wykonuj poniższych kroków na produkcji bez zatwierdzenia. Najpierw przetestuj
odbiornik i rollback jako nieuprzywilejowany użytkownik na katalogu tymczasowym.
Dotychczasowa aplikacja nie jest migrowana samym dodaniem workflow do repo.

1. Zachowaj działającą konfigurację jednostki i `/opt/plates-web`. Sprawdź
   dostępność istniejących modeli w `/var/cache/plates-web` i runtime w
   `/var/lib/plates-web/venv`. **Nie podmieniaj `/opt/plates-web` na symlink**:
   korzystają z niego istniejące usługi treningu i przygotowania.
2. Utwórz root-owned `/storage/plates-deploy/releases` (katalogi 0755), plik
   `/etc/plates/deploy.json` na podstawie `deploy.example.json` (root:root 0644).
   Zatwierdzony hash zależności oblicz przez
   `sha256sum requirements-web.txt`. Każda ich zmiana wymaga osobnego przeglądu,
   instalacji runtime i aktualizacji hasha przez administratora.
3. Po scaleniu zatwierdzonego PR, z czystego checkoutu commita **z `main`**
   utwórz paczkę (ważne także przy squash merge — SHA z gałęzi PR nie musi
   należeć do późniejszej historii `main`):

   ```bash
   python3 scripts/package.py "$(git rev-parse HEAD)" /storage/plates-deploy/bootstrap.zip
   sha256sum /storage/plates-deploy/bootstrap.zip
   ```

   Zasiej pierwsze wydanie jako root-owned katalog
   `/storage/plates-deploy/releases/<SHA40>-<pierwsze16_SHA256_paczki>`:
   tylko pliki z allowlisty paczki, katalogi 0755, pliki 0644. Użyj sprawdzonej
   lokalnej paczki, nie niezweryfikowanego ZIP z sieci. Ustaw `current` jako
   symlink do tego wydania. Odbiornik celowo odmawia wdrożenia bez istniejącego
   `current` — nie potrafi bezpiecznie wycofać niezainicjalizowanej instalacji.
4. Zainstaluj `plates-web-release.conf` jako
   `/etc/systemd/system/plates-web.service.d/release.conf`. Po `daemon-reload`
   i restarcie sprawdź lokalnie `/healthz` oraz prawdziwą inferencję fikcyjnego
   przykładu. Health musi zawierać dokładny SHA wdrożonego commita i
   `index_sha256` zgodny z wydaniem. W razie błędu usuń tylko nowy drop-in,
   przeładuj systemd i uruchom ponownie dotychczasową aplikację z `/opt`.
5. Zainstaluj `deploy.py` jako `/usr/local/sbin/plates-deploy` oraz
   `ssh-command.sh` jako `/usr/local/sbin/plates-ssh-command` (root:root 0755).
   Skonfiguruj dedykowanego użytkownika `plates-deploy`, bez logowania hasłem.
   Jego katalog domowy i `authorized_keys` muszą być zarządzane przez root,
   a konto nie może zapisywać konfiguracji odbiornika ani wydań.
6. W root-owned `authorized_keys` umieść wyłącznie dedykowany klucz publiczny:

   ```text
   restrict,command="/usr/local/sbin/plates-ssh-command" ssh-ed25519 <PUBLIC_KEY> plates-cd
   ```

   Konto potrzebuje powłoki umożliwiającej uruchomienie forced command, nie
   `/usr/sbin/nologin`. Root-owned sudoers, zweryfikowany przez `visudo -cf`:

   ```text
   plates-deploy ALL=(root) NOPASSWD: /usr/local/sbin/plates-deploy
   ```

   Program odbiornika odrzuca inne argumenty i ma stałą ścieżkę konfiguracji.
   Nie przyznawaj kontu ogólnego `sudo systemctl`, `sudo python` ani powłoki root.

## Konfiguracja GitHub i aktywacja

W środowisku `production` ustaw sekrety:

| Sekret | Zawartość |
| --- | --- |
| `VPS_HOST` | Adres osiągalny z GitHub Actions |
| `VPS_PORT` | Port SSH |
| `VPS_USER` | `plates-deploy` |
| `VPS_SSH_KEY` | Prywatny, dedykowany klucz CD |
| `VPS_KNOWN_HOSTS` | Zweryfikowany niezależnie wpis hosta, z portem jeśli niestandardowy |

Nie pobieraj klucza hosta w workflow przez niezaufany `ssh-keyscan`. Nie wklejaj
sekretów do repo, issue ani rozmowy. Sprawdź, że `ssh ... status` tym kluczem
zwraca sam SHA40, a inne polecenia są odrzucane. Dopiero po udanym bootstrapie,
sprawdzeniu rollbacku, scaleniu zatwierdzonego PR i ustawieniu sekretów włącz
zmienną repo `PLATES_CD_ENABLED=true`. Uruchom workflow z `main` i sprawdź CI,
deploy, health i upload fikcyjnego przykładu. Nie scalać PR automatycznie.

## Rollback i utrzymanie

Nie wysyłaj starego commita jako nowego wdrożenia przez workflow. Awaryjny
rollback wykonuje administrator: odczytuje zapisane metadane poprzedniej wersji,
weryfikuje katalog, atomowo przełącza `current`, restartuje jednostkę i sprawdza
jej health. Zachowaj co najmniej bieżące i poprzednie wydanie; automatyczne
usuwanie starych wydań nie jest zaimplementowane. Nie usuwaj katalogów danych,
cache modeli ani checkpointów przy sprzątaniu wydań.
