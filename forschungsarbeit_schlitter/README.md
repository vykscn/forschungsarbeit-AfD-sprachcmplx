# Doch nicht bürgernah? — Sprachkomplexität der AfD im Deutschen Bundestag

Forschungsarbeit zur Frage, ob Abgeordnete der AfD im Bundestag **einfachere** Sprache
nutzen als andere Fraktionen. Sprachkomplexität wird informationstheoretisch über die
mit einem deutschen GPT‑2‑Modell berechnete mittlere **Surprisal** einer Rede gemessen
(2017–März 2025, GERMAPARL2‑Korpus) und mit klassischen Lesbarkeitsindizes trianguliert.

Diese Datei beschreibt **Schritt für Schritt**, wie sich die Ergebnisse von Grund auf
reproduzieren lassen.

---

## Pipeline im Überblick

```
GERMAPARL2-Korpus (.tar.gz)
        │   (R: speech_and_surprisal_generation.R, je Fraktion)
        │   Reden extrahieren  +  Surprisal je Rede via GPT-2 (reticulate → Python)
        ▼
data_projektarbeit/<partei>_speeches_surprisal.csv          (6 Dateien)
        │   (Python: enrich_politicians.py / Notebook-Zelle 0)
        │   Wikidata-Metadaten (Geschlecht, Geburtsjahr, Bildung …) anreichern
        ▼
data_projektarbeit/<partei>_speeches_surprisal_enriched.csv (6 Dateien)
        │   (Python: Notebook-Zelle 1)  zusammenführen + Partei-Indikator
        ▼
data_projektarbeit/all_speeches_enriched.csv                (finaler Datensatz)
        │   (Notebook: analysis.ipynb, Blöcke A–G)
        ▼
analysis/output/results.json · tables/*.csv · figures/*.png
        │   (LaTeX: thesis/assemble.py + latexmk)
        ▼
thesis/thesis.pdf
```

**Wichtig:** Die Skripte und das Notebook nutzen **absolute Pfade** unter
`/Users/nickschlitter/Documents/forschungsarbeit/`. Wer das Repository woanders
ablegt, muss diese Pfade (v. a. `DATA_DIR` / `DATA_FILE`) anpassen.

---

## Voraussetzungen

| Komponente | Zweck |
|---|---|
| **Python 3.12** (conda-Env unter `.conda/`) | Enrichment + Analyse |
| **R** (≥ 4.x) mit `polmineR`, `cwbtools`, `reticulate`, `dplyr`, `stringr`, `purrr`, `tibble`, `arrow` | Korpuszugriff + Reden-Extraktion |
| **`r-tf-env`** (Python-virtualenv für R/reticulate) mit `torch`, `transformers`, `protobuf` | GPT‑2‑Surprisal aus R heraus |
| **GERMAPARL2** Korpus-Tarball (`germaparl_v2.3.0-rc1.tar.gz`) | Datengrundlage |

Das deutsche GPT‑2‑Modell (`dbmdz/german-gpt2`) wird beim ersten Lauf automatisch von
Hugging Face geladen (Internetzugang nötig). Auch das Wikidata-Enrichment braucht Internet.

---

## Schritt 0 — Umgebung einrichten

**Python (Analyse-Seite):**

```bash
cd /Users/nickschlitter/Documents/forschungsarbeit
conda activate ./.conda            # oder: conda create -p ./.conda python=3.12
pip install -r requirements.txt
```

**R-Seite (Surprisal-Generierung):** In R die benötigten Pakete installieren und das
`r-tf-env`-virtualenv mit den ML-Bibliotheken aufsetzen:

```r
install.packages(c("polmineR", "reticulate", "dplyr", "stringr", "purrr", "tibble", "arrow"))
# devtools::install_github("PolMine/cwbtools")
reticulate::virtualenv_create("r-tf-env")
reticulate::virtualenv_install("r-tf-env", c("torch", "transformers", "protobuf"))
```

---

## Schritt 1 — GERMAPARL2-Korpus installieren (einmalig, R)

In `speech_and_surprisal_generation.R` sind die Installationszeilen auskommentiert.
Einmalig den Pfad zum Tarball setzen und installieren:

```r
library(cwbtools)
corpus_install(tarball = "/pfad/zu/germaparl_v2.3.0-rc1.tar.gz")

library(polmineR)
corpus("GERMAPARL2")   # prüft, ob das Korpus verfügbar ist
```

---

## Schritt 2 — Reden extrahieren + Surprisal berechnen (R, je Fraktion)

Datei: **`speech_and_surprisal_generation.R`**. Das Skript

1. lädt das `r-tf-env`-virtualenv und sourct `gpt2_surprisal_optimized.py`
   (`use_virtualenv(...)`, `source_python("gpt2_surprisal_optimized.py")`),
2. filtert das Korpus auf `p_type == "speech"`, eine Fraktion und `protocol_year >= 2017`,
3. fügt Reden je `speaker_name` zusammen und baut ein `tibble` (`name`, `date`, `text`),
4. berechnet je Rede die mittlere Surprisal via `py$compute_surprisal(text)`,
5. schreibt das Ergebnis als CSV.

Da die Korpus-Beta die Partei-Schleife noch nicht zulässt, wird **pro Fraktion einzeln**
gelaufen. Vor jedem Lauf zwei Stellen anpassen:

- **Zeile 36:** `subset(speaker_party == "AfD")` → jeweilige Fraktion
- **Zeile 69:** `write.csv(final_speeches, "afd_speeches_surprisal.csv")` → Dateiname

So entstehen die sechs Dateien:

```
afd_speeches_surprisal.csv      cdu_speeches_surprisal.csv
spd_speeches_surprisal.csv      gruene_speeches_surprisal.csv
fdp_speeches_surprisal.csv      dielinke_speeches_surprisal.csv
```

> ⏳ Dieser Schritt ist der rechenintensivste: GPT‑2 wird über alle Reden ausgeführt.
> Die berechneten CSVs liegen dem Repository bereits bei, sodass die Generierung bei
> Bedarf übersprungen werden kann.

`gpt2_surprisal_optimized.py` lädt `dbmdz/german-gpt2` und berechnet die mittlere
Surprisal als Mittel von −ln P(Token | Kontext) über alle Tokens; bei Texten über
1024 Tokens greift das Kontextfenster des Modells (Truncation, im Methodenteil dokumentiert).

---

## Schritt 3 — Politiker-Metadaten anreichern (Python, je Fraktion)

Datei: **`enrich_politicians.py`** (Wikidata-Anreicherung um Geschlecht, Geburtsjahr,
Geburtsort, Partei, Bildung, Wahlkreis — via SPARQL + Such-Fallback).

Zwei gleichwertige Wege:

**(a) Über das Notebook** (Zelle 0, dort auskommentiert): einmalig auskommentieren und
ausführen — sie reichert jede `*_speeches_surprisal.csv` an und schreibt
`*_speeches_surprisal_enriched.csv`.

**(b) Per Skript, je Datei:**

```bash
python enrich_politicians.py data_projektarbeit/afd_speeches_surprisal.csv
# → data_projektarbeit/afd_speeches_surprisal_enriched.csv   (für jede Fraktion)
```

> Das Enrichment ruft die Wikidata-API auf (rate-limitiert, daher langsam). Die
> angereicherten CSVs liegen ebenfalls bei; der Schritt muss nur bei Datenneubau laufen.

---

## Schritt 4 — Finalen Datensatz bauen (Stacking)

Notebook-**Zelle 1** (dort auskommentiert) führt die sechs `*_enriched.csv` zusammen,
leitet aus dem Dateinamen (Text vor dem ersten `_`) die Spalte `party` ab und schreibt
den finalen Datensatz:

```
data_projektarbeit/all_speeches_enriched.csv
```

Dazu Zelle 1 einmalig auskommentieren und ausführen. Danach Zellen 0 und 1 wieder
auskommentieren (sonst laufen Enrichment/Stacking bei jeder Analyse erneut).

---

## Schritt 5 — Analyse ausführen (Notebook)

Datei: **`analysis.ipynb`** (Blöcke A–G). Liest
`all_speeches_enriched.csv`, bereinigt (prozedurale Beiträge + Reden < 50 Tokens raus →
51.682 Reden / 1.177 Redner:innen), schätzt die Mixed linear models (M0–M2b) samt
Robustheits- und Triangulationsanalysen und schreibt die Outputs.

```bash
conda activate ./.conda
jupyter notebook analysis.ipynb     # alle Zellen ausführen (Run All)
```

Erzeugte Outputs in `analysis/output/`:

- `results.json` — zentrale skalare Kennzahlen (Cohen's *d*, ICC, KIs, Pseudo‑R² …)
- `tables/*.csv` (+ `.md`) — alle Ergebnistabellen
- `figures/*.png` — alle Abbildungen

> Determinismus: Der Lauf setzt `SEED = 42`; `results.json` ist bei unveränderten
> Eingangsdaten reproduzierbar (byte-identisch). Reine Analyse (ohne Surprisal-/
> Enrichment-Schritte) dauert ca. 1–2 Minuten.

---

## Schritt 6 (optional) — Thesis kompilieren

Die schriftliche Arbeit liegt in `thesis/` (LaTeX, biblatex/biber). `thesis.tex` wird aus
den Teil-Dateien (`abstract`, `einleitung`, `theorie`, `methoden_ergebnisse`, `diskussion`,
`fazit`, `anhang`) generiert; Abbildungen werden via `\graphicspath` aus
`analysis/output/figures/` geladen.

```bash
cd thesis
python3 assemble.py                 # erzeugt thesis.tex aus den Teil-Dateien
latexmk -pdf thesis.tex             # bzw. pdflatex → biber → pdflatex → pdflatex
```

Voraussetzung: eine TeX-Distribution mit `biber` und den Paketen `biblatex`, `csquotes`,
`siunitx`, `caption`, `babel-german` u. a.

---

## Repository-Struktur (Auszug)

```
speech_and_surprisal_generation.R   Schritt 1–2: Korpus → Reden + Surprisal (R)
gpt2_surprisal_optimized.py         GPT-2-Surprisal (von R via reticulate gesourct)
enrich_politicians.py               Schritt 3: Wikidata-Anreicherung
analysis.ipynb                      Schritt 4–5: Stacking + Analyse (Blöcke A–G)
requirements.txt / pyproject.toml   Python-Abhängigkeiten
data_projektarbeit/                 Zwischen- und Enddatensätze (*.csv)
analysis/output/                    results.json, tables/, figures/
thesis/                             LaTeX-Quellen + thesis.pdf
```

---

## Kurzfassung (wenn die CSVs schon vorliegen)

Liegt `data_projektarbeit/all_speeches_enriched.csv` bereits vor (im Repo enthalten),
genügt zur Reproduktion der Ergebnisse:

```bash
conda activate ./.conda
pip install -r requirements.txt
jupyter notebook analysis.ipynb     # Run All  → analysis/output/{results.json,tables,figures}
```
