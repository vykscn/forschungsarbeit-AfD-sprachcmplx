"""
enrich_politicians.py
=====================
Reichert einen Datensatz mit Parlamentsreden um Metadaten von Wikidata an.

Öffentliche API
---------------
fetch_metadata(names)       — Schnelle Batch-Abfrage via SPARQL (Schritt 1)
fetch_metadata_full(names)  — SPARQL + Such-Fallback für nicht gefundene Namen (Schritt 1+2)
enrich(df)                  — Führt fetch_metadata_full aus und merged das Ergebnis in df ein

Verwendung im Notebook
----------------------
    from enrich_politicians import enrich, fetch_metadata_full

    # Komplette Pipeline in einem Aufruf:
    enriched = enrich(df)

    # Oder einzeln, um meta separat weiterzuverwenden:
    meta = fetch_metadata_full(df["name"].dropna().unique().tolist())
    enriched = df.merge(meta, on="name", how="left")
"""

import sys
import time

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------

SPARQL     = "https://query.wikidata.org/sparql"
SEARCH_API = "https://www.wikidata.org/w/api.php"
HEADERS    = {"User-Agent": "BundestagSurprisalResearch/1.0"}

POLITICIAN_KEYWORDS = (
    "politiker", "politikerin", "abgeordnet", "bundestag",
    "politician", "statesperson", "member of parliament",
)

# ---------------------------------------------------------------------------
# Schritt 1: SPARQL-Batch-Abfrage
# ---------------------------------------------------------------------------

def _query_chunk(names: list[str]) -> list[dict]:
    """Fragt Wikidata per SPARQL für bis zu ~30 Namen auf einmal ab."""
    values = " ".join(f'"{n}"@de' for n in names)
    query = f"""
SELECT DISTINCT ?name ?sexLabel ?birthdate ?birthplaceLabel ?partyLabel ?educationLabel ?constituencyLabel
WHERE {{
  VALUES ?name {{ {values} }}
  ?person rdfs:label ?name ;
          wdt:P31 wd:Q5 .
  OPTIONAL {{ ?person wdt:P21  ?sex . }}
  OPTIONAL {{ ?person wdt:P569 ?birthdate . }}
  OPTIONAL {{ ?person wdt:P19  ?birthplace . }}
  OPTIONAL {{ ?person wdt:P102 ?party . }}
  OPTIONAL {{ ?person wdt:P69  ?education . }}
  OPTIONAL {{ ?person wdt:P768 ?constituency . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "de,en" . }}
}}
LIMIT 500
"""
    r = requests.get(SPARQL, params={"query": query, "format": "json"},
                     headers=HEADERS, timeout=30)
    r.raise_for_status()

    rows = []
    for b in r.json()["results"]["bindings"]:
        rows.append({
            "name":         b["name"]["value"],
            "sex":          b.get("sexLabel",         {}).get("value"),
            "birthdate":    b.get("birthdate",         {}).get("value"),
            "birthplace":   b.get("birthplaceLabel",   {}).get("value"),
            "party_wiki":   b.get("partyLabel",        {}).get("value"),
            "education":    b.get("educationLabel",    {}).get("value"),
            "constituency": b.get("constituencyLabel", {}).get("value"),
        })
    return rows


def fetch_metadata(names: list[str], chunk_size: int = 30,
                   verbose: bool = True) -> pd.DataFrame:
    """
    Schritt 1: Batch-Abfrage der Metadaten via SPARQL.

    Sucht nach exakter Übereinstimmung des Namens als deutsches Wikidata-Label.
    Gibt einen DataFrame mit einer Zeile pro Person zurück (mehrfache
    Bildungseinträge werden zu einem pipe-separierten String zusammengefasst).
    """
    all_rows = []
    for i in range(0, len(names), chunk_size):
        chunk = names[i : i + chunk_size]
        try:
            all_rows.extend(_query_chunk(chunk))
        except Exception as e:
            print(f"  SPARQL chunk {i // chunk_size} failed: {e}", file=sys.stderr)
        time.sleep(1)

    if not all_rows:
        return pd.DataFrame()

    meta = pd.DataFrame(all_rows)
    meta["birthdate"]  = pd.to_datetime(meta["birthdate"], errors="coerce")
    meta["birth_year"] = meta["birthdate"].dt.year.astype("Int64")

    meta = (
        meta.groupby("name", sort=False)
        .agg({
            "sex":          "first",
            "birth_year":   "first",
            "birthplace":   "first",
            "party_wiki":   "first",
            "education":    lambda x: " | ".join(x.dropna().unique()) or None,
            "constituency": "first",
        })
        .reset_index()
    )

    if verbose:
        print(f"SPARQL: {len(meta)}/{len(set(names))} Namen gefunden")
    return meta


# ---------------------------------------------------------------------------
# Schritt 2: Such-Fallback über die Wikidata-Such-API
# ---------------------------------------------------------------------------

def _get_json(url: str, params: dict, retries: int = 3, backoff: int = 5) -> dict:
    """HTTP-GET mit automatischem Retry bei Rate-Limiting oder leerer Antwort."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=20)
            if r.status_code == 429:
                wait = backoff * (attempt + 1)
                print(f"  Rate limited — warte {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(backoff)
            else:
                raise e
    return {}


def _fetch_props(qid: str) -> dict:
    """Lädt Eigenschaften (Geschlecht, Geburtsdatum, Partei, …) für eine Wikidata-ID."""
    pid_to_col = {
        "P21":  "sex",
        "P569": "birthdate",
        "P19":  "birthplace",
        "P102": "party_wiki",
        "P69":  "education",
        "P768": "constituency",
    }
    data   = _get_json(SEARCH_API, {
        "action": "wbgetentities", "ids": qid,
        "props": "claims|labels", "languages": "de|en", "format": "json",
    })
    entity = data.get("entities", {}).get(qid, {})
    claims = entity.get("claims", {})

    def first_value(prop):
        for snak in claims.get(prop, []):
            dv = snak.get("mainsnak", {}).get("datavalue", {})
            v  = dv.get("value")
            if isinstance(v, dict):
                return v.get("id") or v.get("time", "")[:11].lstrip("+")
            return v
        return None

    row  = {col: first_value(pid) for pid, col in pid_to_col.items()}

    # QIDs in lesbare Labels übersetzen (z.B. Q6581097 → "männlich")
    qids = [row[c] for c in ("sex", "birthplace", "party_wiki", "constituency")
            if row.get(c) and str(row.get(c, "")).startswith("Q")]
    if qids:
        lr = _get_json(SEARCH_API, {
            "action": "wbgetentities", "ids": "|".join(qids),
            "props": "labels", "languages": "de|en", "format": "json",
        }).get("entities", {})

        def label(q):
            e = lr.get(q, {}).get("labels", {})
            return (e.get("de") or e.get("en") or {}).get("value")

        for col in ("sex", "birthplace", "party_wiki", "constituency"):
            if row.get(col) and str(row.get(col, "")).startswith("Q"):
                row[col] = label(row[col])
    return row


def _search_candidates(name: str) -> list[str]:
    """
    Erstellt eine Liste von Suchbegriffen für einen Namen, von spezifisch zu allgemein.

    Beispiele:
      "Roman Johannes Reusch"  → ["Roman Reusch"]          (Mittelnamen weglassen)
      "Armin-Paulus Hampel"    → ["Armin-Paulus Hampel",   (erst exakt versuchen,
                                   "Armin Hampel"]          dann Bindestrich-Suffix kürzen)
    """
    parts        = name.split()
    first, last  = parts[0], parts[-1]
    base         = f"{first} {last}" if len(parts) > 2 else name
    candidates   = [base]
    if "-" in first:
        candidates.append(f"{first.split('-')[0]} {last}")
    return candidates


def _search_fallback(name: str) -> dict | None:
    """Sucht einen Politiker über die Wikidata-Such-API und gibt seine Eigenschaften zurück."""
    for candidate in _search_candidates(name):
        data = _get_json(SEARCH_API, {
            "action": "wbsearchentities", "search": candidate,
            "language": "de", "type": "item", "limit": 5, "format": "json",
        })
        for hit in data.get("search", []):
            desc = hit.get("description", "").lower()
            if any(kw in desc for kw in POLITICIAN_KEYWORDS):
                row        = _fetch_props(hit["id"])
                row["name"] = name
                return row
    return None


def fetch_metadata_fallback(names: list[str], verbose: bool = True) -> pd.DataFrame:
    """
    Schritt 2: Such-Fallback für Namen, die SPARQL nicht gefunden hat.

    Probiert für jeden Namen die Wikidata-Such-API mit mehreren Schreibweisen.
    Gibt einen DataFrame mit denselben Spalten wie fetch_metadata zurück.
    """
    rows = []
    for name in names:
        row = _search_fallback(name)
        if row:
            rows.append(row)
            if verbose:
                print(f"  ✓ {name}")
        else:
            if verbose:
                print(f"  ✗ {name} (nicht gefunden)")
        time.sleep(2)

    if not rows:
        return pd.DataFrame()

    fb = pd.DataFrame(rows)
    fb["birthdate"]  = pd.to_datetime(fb.get("birthdate"), errors="coerce")
    fb["birth_year"] = fb["birthdate"].dt.year.astype("Int64")
    return fb.drop(columns="birthdate", errors="ignore")


# ---------------------------------------------------------------------------
# Kombinierte Pipeline
# ---------------------------------------------------------------------------

def fetch_metadata_full(names: list[str], chunk_size: int = 30,
                        verbose: bool = True) -> pd.DataFrame:
    """
    Führt beide Schritte nacheinander aus und gibt einen vollständigen
    Metadaten-DataFrame zurück.

    Parameter:
        names      : Liste eindeutiger Politikernamen
        chunk_size : Wie viele Namen pro SPARQL-Anfrage (Standard: 30)
        verbose    : Fortschrittsmeldungen ausgeben (Standard: True)

    Rückgabe:
        DataFrame mit Spalten: name, sex, birth_year, birthplace,
                                party_wiki, education, constituency
    """
    unique = list(dict.fromkeys(names))  # Reihenfolge erhalten, Duplikate entfernen

    # Schritt 1: SPARQL
    meta = fetch_metadata(unique, chunk_size=chunk_size, verbose=verbose)

    # Schritt 2: Fallback für nicht gefundene Namen
    matched   = set(meta["name"]) if not meta.empty else set()
    unmatched = [n for n in unique if n not in matched]

    if unmatched:
        if verbose:
            print(f"Fallback-Suche für {len(unmatched)} nicht gefundene Namen...")
        fb = fetch_metadata_fallback(unmatched, verbose=verbose)
        if not fb.empty:
            meta = pd.concat([meta, fb], ignore_index=True)

    if verbose:
        print(f"Gesamt: {meta['name'].nunique()}/{len(unique)} Namen gefunden")
    return meta


def enrich(df: pd.DataFrame, name_col: str = "name",
           chunk_size: int = 30, verbose: bool = True) -> pd.DataFrame:
    """
    Bequemer Einstiegspunkt: ruft fetch_metadata_full auf und merged
    das Ergebnis direkt in den übergebenen DataFrame ein.

    Parameter:
        df         : DataFrame mit einer Spalte, die Politikernamen enthält
        name_col   : Name der Spalten mit den Politikernamen (Standard: "name")
        chunk_size : Wie viele Namen pro SPARQL-Anfrage (Standard: 30)
        verbose    : Fortschrittsmeldungen ausgeben (Standard: True)

    Rückgabe:
        Angereicherter DataFrame (Original + Metadaten-Spalten)
    """
    unique_names = df[name_col].dropna().unique().tolist()
    meta         = fetch_metadata_full(unique_names, chunk_size=chunk_size, verbose=verbose)
    return df.merge(meta, on=name_col, how="left")


# ---------------------------------------------------------------------------
# Direktaufruf (python enrich_politicians.py <csv-datei>)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    if csv_path is None:
        sys.exit("Verwendung: python enrich_politicians.py <pfad/zur/datei.csv>")

    df       = pd.read_csv(csv_path, index_col=0)
    enriched = enrich(df)

    out = csv_path.replace(".csv", "_enriched.csv")
    enriched.to_csv(out, index=False)
    print(f"Gespeichert → {out}")
    print(enriched.head())
