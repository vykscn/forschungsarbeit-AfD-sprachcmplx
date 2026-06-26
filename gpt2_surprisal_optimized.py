# =============================================================================
# gpt2_surprisal_optimized.py
#
# Dieses Skript berechnet sogenannte "Surprisal-Werte" für deutsche Texte.
#
# Was ist Surprisal?
# ------------------
# Surprisal misst, wie überraschend ein Wort im Kontext seiner Vorgängerwörter
# ist. Hoher Surprisal = das Wort kam unerwartet. Niedriger Surprisal = das
# Wort war vorhersehbar. Die Messung basiert auf einem Sprachmodell, das auf
# Basis aller vorherigen Wörter vorhersagt, wie wahrscheinlich das nächste
# Wort ist.
#
# Beispiel: Im Satz "Die Sonne scheint ___"
#   - "hell"  → niedriger Surprisal (vorhersehbar)
#   - "laut"  → hoher Surprisal (unerwartete Wortkombination)
#
# Welches Modell wird verwendet?
# ------------------------------
# "dbmdz/german-gpt2" ist eine Version von GPT-2, die speziell auf deutschen
# Texten trainiert wurde. Das Modell hat gelernt, welche Wörter im Deutschen
# typischerweise aufeinander folgen.
# =============================================================================

import torch                              # PyTorch: das Framework für neuronale Netze
import torch.nn.functional as F          # Mathematische Hilfsfunktionen von PyTorch
from transformers import AutoTokenizer, AutoModelForCausalLM  # HuggingFace-Bibliothek zum Laden vortrainierter Sprachmodelle


# --- Modellname definieren ----------------------------------------------------
# Der Name des Modells auf der HuggingFace-Plattform. Beim ersten Aufruf wird
# es automatisch heruntergeladen und danach lokal zwischengespeichert.
model_name = "dbmdz/german-gpt2"


# --- Hardware erkennen --------------------------------------------------------
# Neuronale Netze rechnen viel schneller auf einer Grafikkarte (GPU) als auf
# dem normalen Prozessor (CPU). Wir prüfen automatisch, welche Hardware
# vorhanden ist und nutzen die schnellste verfügbare.
#
# CUDA  = NVIDIA-Grafikkarten (typisch in Desktop-PCs / Servern)
# MPS   = Apple Silicon (M1/M2/M3/M4/M5-Chips in modernen Macs)
# CPU   = normaler Prozessor (langsam, aber immer verfügbar)
#
# float16 ist ein kompakteres Zahlenformat (halbe Präzision) das auf GPUs
# doppelt so schnell ist wie das Standard-float32, ohne die Ergebnisse
# nennenswert zu verändern.
if torch.cuda.is_available():
    device, dtype = "cuda", torch.float16
elif torch.backends.mps.is_available():
    device, dtype = "mps", torch.float16
else:
    device, dtype = "cpu", torch.float32


# --- Tokenizer laden ----------------------------------------------------------
# Bevor das Modell einen Text verarbeiten kann, muss der Text in Zahlen
# umgewandelt werden. Das übernimmt der Tokenizer:
#
# Text → Tokens → Zahlen
# "Guten Morgen" → ["Gut", "en", "Mor", "gen"] → [1234, 567, 890, 321]
#
# GPT-2 kennt kein spezielles "Auffüll-Symbol" (padding token), das benötigt
# wird, wenn mehrere Texte unterschiedlicher Länge gleichzeitig verarbeitet
# werden. Wir verwenden stattdessen das "End-of-Sequence"-Symbol (eos_token)
# als Platzhalter, da es ohnehin keine inhaltliche Bedeutung hat.
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # eos-Token als Platzhalter verwenden


# --- Modell laden -------------------------------------------------------------
# Das eigentliche Sprachmodell wird geladen und auf die verfügbare Hardware
# verschoben. "eval()" schaltet den Trainingsmodus aus — das Modell ist fertig
# trainiert und muss nur noch Vorhersagen machen, kein Lernen mehr.
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
model.to(device).eval()


# --- Hauptfunktion: Surprisal für mehrere Texte berechnen --------------------
def compute_surprisal_batch(texts: list[str], batch_size: int = 128) -> list[float]:
    """
    Berechnet den mittleren Surprisal-Wert für eine Liste von Texten.

    Parameter:
        texts      : Liste von Texten (in diesem Fall Parlamentsreden)
        batch_size : Wie viele Texte gleichzeitig verarbeitet werden.
                     Mehr = schneller, aber mehr Arbeitsspeicher nötig.

    Rückgabe:
        Liste von Dezimalzahlen — ein mittlerer Surprisal-Wert pro Text.
    """
    all_surprisals = []

    # Texte werden in Gruppen ("Batches") aufgeteilt, weil der Arbeitsspeicher
    # der Grafikkarte nicht unbegrenzt groß ist. Statt alle Texte auf einmal
    # zu verarbeiten, verarbeiten wir sie in Häppchen.
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        # Tokenisierung: Texte → Zahlenfolgen
        # padding=True:    kürzere Texte werden auf die Länge des längsten
        #                  Textes im Batch mit Platzhaltern aufgefüllt
        # truncation=True: Texte die länger als das Modell verarbeiten kann
        #                  (max. 1024 Tokens) werden abgeschnitten
        enc = tokenizer(
            batch,
            return_tensors="pt",   # Ausgabe als PyTorch-Tensoren (mehrdimensionale Zahlenmatrizen)
            padding=True,
            truncation=True,
            max_length=model.config.n_positions,  # GPT-2 Limit: 1024 Tokens
        ).to(device)               # Daten auf die Grafikkarte verschieben

        # Modell-Inferenz: Das Sprachmodell berechnet für jede Position im Text,
        # wie wahrscheinlich jedes Wort im Vokabular als nächstes wäre.
        with torch.inference_mode():
            logits = model(**enc).logits  # Rohe Modellausgaben: [Texte × Tokens × Vokabular]

        B, T, V = logits.shape
        # B = Anzahl der Texte im Batch
        # T = maximale Tokenanzahl
        # V = Vokabulargröße (~50.000 deutsche Wörter/Wortteile)

        # Auffüllungs-Maske: Markiert welche Tokens echte Wörter sind (1)
        # und welche nur Platzhalter (0), damit Platzhalter nicht in die
        # Surprisal-Berechnung einfließen. Wir starten bei Position 1, weil
        # das erste Token keinen Vorgänger hat und daher keinen Surprisal-Wert bekommt.
        mask = enc["attention_mask"][:, 1:].float()  # [Texte × (Tokens-1)]

        # Surprisal-Berechnung mit cross_entropy:
        # Für jede Token-Position berechnen wir, wie überraschend das tatsächlich
        # vorkommende Wort war, gegeben alle vorherigen Wörter.
        #
        # Technischer Hintergrund: Der Logit von Position i-1 gibt an, welche
        # Wörter das Modell nach Position i-1 erwartet. Wir prüfen, wie
        # unwahrscheinlich das Wort an Position i tatsächlich war.
        # cross_entropy berechnet dies effizient ohne den gesamten [B×T×V]-
        # Wahrscheinlichkeitstensor im Speicher ablegen zu müssen (bei langen
        # Texten wären das mehrere hundert Megabyte).
        per_token = F.cross_entropy(
            logits[:, :-1].float().reshape(B * (T - 1), V),  # Vorhersagen für alle Positionen
            enc["input_ids"][:, 1:].reshape(B * (T - 1)),    # Tatsächlich aufgetretene Tokens
            reduction="none",                                  # Einzelwert pro Token, kein Mittelwert
        ).reshape(B, T - 1)

        # Mittlerer Surprisal pro Text: Platzhalter-Tokens werden mit 0
        # multipliziert (ausgeblendet) und der Durchschnitt über alle echten
        # Tokens gebildet.
        surprisals = (per_token * mask).sum(1) / mask.sum(1)
        all_surprisals.extend(surprisals.tolist())

    return all_surprisals


# --- Hilfsfunktion für einen einzelnen Text ----------------------------------
# Wrapper damit man auch einzelne Texte bequem übergeben kann,
# ohne selbst eine Liste erstellen zu müssen.
def compute_surprisal(text: str) -> float:
    return compute_surprisal_batch([text])[0]


# --- Testlauf (nur wenn das Skript direkt ausgeführt wird) -------------------
# Dieser Block läuft nur, wenn man das Skript direkt startet (z.B. über das
# Terminal). Wird das Skript von R über reticulate importiert, wird dieser
# Block übersprungen.
if __name__ == "__main__":
    texts = [
        "Das ist ein iah.",
        "Ich liebe es, in Berlin zu spazieren.",
        "Künstliche Intelligenz ist faszinierend.",
    ]
    results = compute_surprisal_batch(texts)
    for text, s in zip(texts, results):
        print(f"Text: {text}\nMean surprisal: {s:.4f}\n")
