# BPP-Übersicht: Alle Codecs & Datasets

**Letzte Aktualisierung:** 2025-12-16

---

## Vollständige BPP-Tabelle (GEMESSENE WERTE)

### EuroSAT (64×64 px, Original: 24.0 BPP)

| Codec     | q1    | q2    | q3    | q4    | q5    | q6    | Typ |
|-----------|-------|-------|-------|-------|-------|-------|-----|
| **Original** | 24.0 | 24.0 | 24.0 | 24.0 | 24.0 | 24.0 | - |
| JPEG      | 0.45  | 0.59  | 0.77  | 0.89  | 1.20  | 1.73  | Quality-controlled |
| JPEG2000  | 0.10  | 0.25  | 0.40  | 0.57  | 0.77  | 0.97  | Ratio-controlled |
| Cheng2020 | 0.13  | 0.18  | 0.26  | 0.39  | 0.53  | 0.72  | Lambda-index |
| MS-ILLM   | 0.04  | 0.07  | 0.14  | 0.28  | 0.45  | 0.90  | Quality-index |
| JPEG-AI   | 0.13  | 0.25  | 0.50  | 0.75  | 1.00  | 1.50  | **Rate-controlled** |

### RESISC45 (256×256 px, Original: 24.0 BPP) - GEMESSENE WERTE

| Codec     | q1    | q2    | q3    | q4    | q5    | q6    | Typ |
|-----------|-------|-------|-------|-------|-------|-------|-----|
| **Original** | 24.0 | 24.0 | 24.0 | 24.0 | 24.0 | 24.0 | - |
| JPEG      | 0.302 | 0.658 | 1.201 | 1.549 | 1.900 | 2.230 | Quality-controlled |
| JPEG2000  | 0.235 | 0.351 | 0.592 | 0.950 | 1.702 | 2.386 | Ratio-controlled |
| Cheng2020 | -     | -     | 0.304 | 0.494 | 0.709 | 0.982 | Lambda-index |
| MS-ILLM   | *     | *     | *     | *     | *     | *     | Quality-index |
| JPEG-AI   | 0.12  | 0.25  | 0.50  | 0.75  | 1.00  | 1.50  | **Rate-controlled** |

**Legende:**
- `-` = Nicht komprimiert für dieses Dataset
- `*` = BPP aus Kompression nicht gespeichert (MS-ILLM auf Cloud-GPU komprimiert)
- **Rate-controlled** = JPEG-AI garantiert exakte BPP durch Bitrate-Targeting

---

## Kompressionsraten (bei 24 BPP Original)

| BPP Target | Kompressionsrate |
|------------|------------------|
| 0.12       | **200:1**        |
| 0.25       | **96:1**         |
| 0.50       | **48:1**         |
| 0.75       | **32:1**         |
| 1.00       | **24:1**         |
| 1.50       | **16:1**         |

---

## Quality Level Konfiguration

### Codec-spezifische Parameter

| Codec     | q1    | q2    | q3    | q4    | q5    | q6    | Library/Notes |
|-----------|-------|-------|-------|-------|-------|-------|---------------|
| JPEG      | 10    | 25    | 50    | 75    | 90    | 95    | Quality 1-100 (Pillow) |
| JPEG2000  | 5     | 10    | 20    | 40    | 80    | 160   | Compression Ratio (OpenJPEG) |
| Cheng2020 | 1     | 2     | 3     | 4     | 5     | 6     | Lambda Index (CompressAI) |
| MS-ILLM   | 1     | 2     | 3     | 4     | 5     | 6     | Quality Index |
| JPEG-AI   | 1     | 2     | 3     | 4     | 5     | 6     | BPP Target (RefSW 6.0) |

---

## Begründung der Quality Level Wahl

### In der Thesis dokumentiert? ✅ JA

**Fundstellen:**
- [chap04.tex](../docs/thesis_fhkufstein/chapters/chap04.tex#L216): Section "Quality Level Configuration"
- [app02.tex](../docs/thesis_fhkufstein/appendix/app02.tex#L51): Appendix "Compression Parameters"

### Rationale:

1. **Abdeckung des praktisch relevanten BPP-Bereichs:**
   - Ultra-niedrig (0.04-0.13 BPP): Extreme Kompression, Test der Grenzen
   - Niedrig (0.14-0.30 BPP): Typisch für Bandbreiten-kritische Anwendungen
   - Mittel (0.40-0.60 BPP): Balance zwischen Qualität und Kompression
   - Hoch (0.75-1.00 BPP): Hochqualitative Archivierung
   - Sehr hoch (1.00-1.50 BPP): Nahezu verlustfrei

2. **Fairer Cross-Codec Vergleich:**
   - Jeder Codec wird mit 6 Levels evaluiert
   - Unterschiedliche interne Parameter → unterschiedliche BPP
   - Vergleich erfolgt über Rate-Accuracy-Kurven (nicht Level-zu-Level)

3. **Konsistenz mit Literatur:**
   - 0.1-1.5 BPP entspricht Standard-Testbereich in Kompressionspapern
   - Ermöglicht BD-Rate Berechnung und Vergleich mit State-of-the-Art

### Wichtiger Hinweis aus Thesis:

> "Quality levels are not directly comparable across codecs. Each codec's internal quality parameter produces different bitrates, necessitating bitrate-normalized comparison for fair evaluation."

---

## Aktueller Evaluationsstatus ✅ VOLLSTÄNDIG

| Dataset   | JPEG | JPEG2000 | Cheng2020 | MS-ILLM | JPEG-AI |
|-----------|------|----------|-----------|---------|---------|
| EuroSAT   | ✅   | ✅       | ✅        | ✅      | ✅      |
| RESISC45  | ✅   | ✅       | ✅        | ✅      | ✅      |

**JPEG-AI RESISC45 - VOLLSTÄNDIG (2025-12-16):**

| Quality | BPP  | ResNet-18 | ViT-S/16 |
|---------|------|-----------|----------|
| Q1      | 0.12 | 91.19%    | 94.25%   |
| Q2      | 0.25 | 94.84%    | 98.45%   |
| Q3      | 0.50 | 97.42%    | 99.29%   |
| Q4      | 0.75 | 97.94%    | 99.42%   |
| Q5      | 1.00 | 98.27%    | 99.45%   |
| Q6      | 1.50 | 98.48%    | 99.49%   |
