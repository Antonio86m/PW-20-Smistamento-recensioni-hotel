# 🏨 Sistema di Classificazione Recensioni Alberghiere

Sistema ML end-to-end per classificazione automatica di recensioni alberghiere con assegnazione reparto e analisi sentiment.

## 📋 Descrizione

Questo prototipo implementa una pipeline completa di Machine Learning per:

1. **Classificazione Reparto** - Assegna automaticamente recensioni a:
   - 🛏️ **Housekeeping** (pulizia, manutenzione camere)
   - 🎫 **Reception** (check-in/out, servizio clienti)
   - 🍽️ **F&B** (Food & Beverage: colazione, ristorante, bar)

2. **Analisi Sentiment** - Classifica recensioni come:
   - 😊 **Positive**
   - 😞 **Negative**

3. **Dashboard Interattiva** - Interfaccia web per:
   - Predizione singola recensione
   - Batch processing con upload/download CSV
   - Visualizzazione probabilità e confidenza

## 🚀 Quick Start

### Prerequisiti

- Python 3.11+
- pip o uv (package manager)

### Installazione

Le dipendenze sono già configurate nel progetto. Se esegui localmente:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit
```

### Esecuzione Completa

#### Step 1: Genera Dataset Sintetico

```bash
python generate_dataset.py
```

**Output:**
- `dataset_recensioni.csv` - 500 recensioni italiane con etichette

**Tempo stimato:** ~2 secondi

---

#### Step 2: Training Modelli ML

```bash
python train_model.py
```

**Output:**
- `models/vectorizer.pkl` - TF-IDF vectorizer
- `models/department_model.pkl` - Classificatore reparto
- `models/sentiment_model.pkl` - Classificatore sentiment
- `results/confusion_matrix_department.png` - Matrice confusione reparto
- `results/confusion_matrix_sentiment.png` - Matrice confusione sentiment
- `results/metrics.pkl` - Metriche performance

**Tempo stimato:** ~5-10 secondi

**Metriche attese:**
- Department Accuracy: >85%
- Sentiment Accuracy: >85%

---

#### Step 3: Esegui Dashboard

```bash
streamlit run app.py --server.port 5000
```

**Funzionalità:**

1. **Predizione Singola:**
   - Inserisci titolo e testo recensione
   - Ottieni reparto, sentiment e probabilità
   - Visualizza testo preprocessato

2. **Batch Processing:**
   - Carica CSV con colonne `title` e `body`
   - Ottieni predizioni per tutte le recensioni
   - Scarica CSV risultati con timestamp

**Accesso:** Browser su `http://localhost:5000`

---

#### Step 4: Genera Report

```bash
python generate_report.py
```

**Output:**
- `REPORT.md` - Report completo (~15 pagine) con:
  - Flusso operativo e gestione reclami
  - Scelte tecniche e algoritmi
  - Risultati e metriche
  - Limiti e miglioramenti proposti
- `results/metrics_summary.png` - Grafico riepilogativo metriche

**Tempo stimato:** ~1 secondo

---

## 📁 Struttura Progetto

```
.
├── generate_dataset.py          # Generatore dataset sintetico
├── train_model.py                # Pipeline training ML
├── utils.py                      # Utility preprocessing e predizione
├── app.py                        # Dashboard Streamlit
├── generate_report.py            # Generatore report automatico
├── README.md                     # Questo file
│
├── dataset_recensioni.csv        # Dataset generato (500 recensioni)
│
├── models/                       # Modelli addestrati
│   ├── vectorizer.pkl
│   ├── department_model.pkl
│   └── sentiment_model.pkl
│
└── results/                      # Risultati e visualizzazioni
    ├── confusion_matrix_department.png
    ├── confusion_matrix_sentiment.png
    ├── metrics_summary.png
    └── metrics.pkl
```

---

## 🔧 Dettagli Tecnici

### Algoritmi Utilizzati

1. **Preprocessing:**
   - Lowercasing
   - Rimozione punteggiatura
   - Tokenizzazione base

2. **Feature Extraction:**
   - **TF-IDF Vectorizer**
     - `max_features=1000`
     - `ngram_range=(1,2)` - unigrammi e bigrammi
     - `min_df=2`, `max_df=0.8`

3. **Classificatori:**
   - **Logistic Regression** (per entrambi i task)
     - `C=1.0` - regolarizzazione L2
     - `class_weight='balanced'` - gestisce sbilanciamento
     - `max_iter=1000`

### Perché Logistic Regression?

- ✅ Semplice e interpretabile
- ✅ Veloce (predizione <1ms)
- ✅ Ottime performance su testo con TF-IDF
- ✅ Probabilità calibrate
- ✅ Ideale per prototipi e baseline

---

## 📊 Formato Dati

### Dataset Input (CSV)

Il file `dataset_recensioni.csv` contiene:

| Colonna | Tipo | Descrizione |
|---------|------|-------------|
| `id` | int | Identificativo univoco |
| `title` | string | Titolo recensione |
| `body` | string | Testo recensione |
| `department` | string | Reparto (Housekeeping/Reception/F&B) |
| `sentiment` | string | Sentiment (positive/negative) |

**Esempio:**

```csv
id,title,body,department,sentiment
1,"Camera impeccabile","La camera era pulitissima, letto comodo",Housekeeping,positive
2,"Check-in lento","Attesa di 30 minuti per il check-in",Reception,negative
```

### Predizioni Output (CSV)

Il batch processing genera CSV con:

| Colonna | Tipo | Descrizione |
|---------|------|-------------|
| `title` | string | Titolo originale |
| `body` | string | Testo originale |
| `predicted_department` | string | Reparto predetto |
| `department_confidence` | float | Confidenza reparto (0-1) |
| `predicted_sentiment` | string | Sentiment predetto |
| `sentiment_confidence` | float | Confidenza sentiment (0-1) |
| `positive_probability` | float | Probabilità positivo (0-1) |

**Filename formato:** `predizioni_recensioni_YYYYMMDD_HHMMSS.csv`

---

## 🎯 Casi d'Uso

### 1. Routing Automatico Recensioni

```python
from utils import load_models, predict_single

# Carica modelli
vectorizer, dept_model, sent_model = load_models()

# Nuova recensione
result = predict_single(
    title="Camera pulita",
    body="Stanza impeccabile e letto comodo",
    vectorizer=vectorizer,
    dept_model=dept_model,
    sent_model=sent_model
)

print(f"Reparto: {result['department']}")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidenza: {result['department_confidence']:.2%}")
```

### 2. Analisi Batch Storica

```python
import pandas as pd
from utils import load_models, predict_batch

# Carica recensioni storiche
df = pd.read_csv('recensioni_gennaio.csv')

# Predici
vectorizer, dept_model, sent_model = load_models()
df_results = predict_batch(df, vectorizer, dept_model, sent_model)

# Statistiche
print(df_results['predicted_department'].value_counts())
print(f"Sentiment positivi: {(df_results['predicted_sentiment']=='positive').sum()}")
```

### 3. Dashboard Real-Time

Esegui dashboard per interfaccia utente:
```bash
streamlit run app.py --server.port 5000
```

---

## 📈 Performance Attese

Con dataset sintetico (500 recensioni, split 80/20):

| Metrica | Department | Sentiment |
|---------|------------|-----------|
| **Accuracy** | ~88-92% | ~90-95% |
| **F1-Macro** | ~87-91% | ~90-94% |
| **F1-Weighted** | ~88-92% | ~90-95% |

**Note:** Performance su dati reali potrebbero variare del ±5-10%.

---

## ⚠️ Limitazioni

### Dataset Sintetico
- Recensioni generate artificialmente
- Lessico limitato e pattern predefiniti
- Nessuna variabilità linguistica reale

### Preprocessing Basilare
- Nessuna rimozione stopwords italiane
- Nessuna lemmatizzazione
- Nessuna gestione negazioni

### Modello Semplice
- Sentiment binario (manca "neutro")
- Nessuna soglia confidenza per casi incerti
- Solo lingua italiana

---

## 🔮 Miglioramenti Futuri

### Quick Wins
1. ✅ Aggiungere stopwords italiane (spaCy)
2. ✅ Implementare lemmatizzazione
3. ✅ Soglie confidenza per human review

### Medio Termine
1. 📊 Raccogliere dati reali (1000+ recensioni)
2. 🧠 Testare ensemble models
3. 🎯 Sentiment tripartito (pos/neu/neg)

### Lungo Termine
1. 🤖 Fine-tuning BERT italiano (UmBERTO)
2. 🏷️ Classificazione multi-label
3. 📉 Analisi trend temporali
4. 🔄 Feedback loop automatico

---

## 🛠️ Troubleshooting

### Errore: "Models not found"
**Soluzione:** Esegui prima `python train_model.py`

### Errore: "Dataset not found"
**Soluzione:** Esegui prima `python generate_dataset.py`

### Dashboard non si apre
**Soluzione:** 
1. Verifica modelli esistano in `models/`
2. Controlla porta 5000 sia libera
3. Usa `streamlit run app.py --server.port 5001`

### Performance basse
**Possibili cause:**
- Dataset troppo piccolo → Genera più recensioni
- Sbilanciamento classi → Verifica distribuzione
- Overfitting → Riduci `max_features` in TF-IDF

---

## 📚 Risorse

### Documentazione
- [Scikit-learn](https://scikit-learn.org/)
- [Streamlit](https://docs.streamlit.io/)
- [Pandas](https://pandas.pydata.org/)

### Paper Rilevanti
- Logistic Regression for Text Classification
- TF-IDF vs Word Embeddings
- Sentiment Analysis in Hospitality

---

## 📝 Licenza

Questo è un prototipo educativo/dimostrativo.

---

## 👥 Autori

Sistema sviluppato come prototipo ML per classificazione recensioni alberghiere.

**Data:** Ottobre 2025

---

## 🆘 Supporto

Per problemi o domande:
1. Verifica di aver eseguito tutti gli step in ordine
2. Controlla i file di log in `results/`
3. Leggi il REPORT.md per dettagli tecnici

---

## 🎉 Esempi Uso Dashboard

### Predizione Singola

1. Apri dashboard
2. Tab "Predizione Singola"
3. Inserisci:
   - **Titolo:** "Colazione eccellente"
   - **Testo:** "Buffet molto ricco con prodotti freschi e di qualità"
4. Click "Analizza Recensione"
5. Vedi risultati:
   - Reparto: F&B
   - Sentiment: Positive
   - Confidenza: 95%

### Batch Processing

1. Tab "Batch Processing"
2. Scarica CSV esempio
3. Modifica con tue recensioni
4. Upload file
5. Click "Esegui Predizioni Batch"
6. Download risultati con timestamp

---

**Buon lavoro! 🚀**
