"""
Utility Functions per Preprocessing e Predizione
"""

import re
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Lazy load spaCy per evitare import al boot
_nlp = None

def get_spacy_model():
    """Carica modello spaCy in modo lazy"""
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load('it_core_news_sm')
        except OSError:
            print("⚠️ Modello spaCy non trovato. Usa preprocessing basilare.")
            _nlp = False
    return _nlp


def preprocess_text(text, use_advanced=False):
    """
    Preprocessing del testo con opzione avanzata (spaCy)
    
    Args:
        text: stringa di testo da preprocessare
        use_advanced: se True usa lemmatizzazione e stopword removal con spaCy
    
    Returns:
        testo preprocessato
    """
    if not isinstance(text, str):
        return ""
    
    # Preprocessing avanzato con spaCy
    if use_advanced:
        nlp = get_spacy_model()
        if nlp and nlp is not False:
            doc = nlp(text.lower())
            # Lemmatizzazione + rimozione stopwords + punteggiatura
            tokens = [token.lemma_ for token in doc 
                     if not token.is_stop and not token.is_punct and token.text.strip()]
            return ' '.join(tokens)
    
    # Preprocessing basilare (fallback o default)
    # Lowercasing
    text = text.lower()
    
    # Rimozione punteggiatura
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Rimozione spazi multipli
    text = re.sub(r'\s+', ' ', text)
    
    # Strip spazi iniziali e finali
    text = text.strip()
    
    return text


def combine_title_body(title, body):
    """
    Combina titolo e corpo della recensione
    
    Args:
        title: titolo della recensione
        body: corpo della recensione
    
    Returns:
        testo combinato
    """
    return f"{title} {body}"


def load_models(dept_model_path='models/department_model.pkl', 
                sent_model_path='models/sentiment_model.pkl',
                vectorizer_path='models/vectorizer.pkl'):
    """
    Carica i modelli salvati
    
    Args:
        dept_model_path: percorso modello department
        sent_model_path: percorso modello sentiment
        vectorizer_path: percorso vectorizer
    
    Returns:
        tuple (vectorizer, dept_model, sent_model)
    """
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open(dept_model_path, 'rb') as f:
        dept_model = pickle.load(f)
    
    with open(sent_model_path, 'rb') as f:
        sent_model = pickle.load(f)
    
    return vectorizer, dept_model, sent_model


def predict_single(title, body, vectorizer, dept_model, sent_model, use_advanced=False, confidence_threshold=0.7):
    """
    Predice reparto e sentiment per una singola recensione
    
    Args:
        title: titolo recensione
        body: corpo recensione
        vectorizer: TfidfVectorizer fitted
        dept_model: modello department
        sent_model: modello sentiment
        use_advanced: usa preprocessing avanzato (spaCy)
        confidence_threshold: soglia per marcare predizioni incerte
    
    Returns:
        dict con predizioni e probabilità
    """
    # Combina e preprocessa
    text = combine_title_body(title, body)
    text_clean = preprocess_text(text, use_advanced=use_advanced)
    
    # Vectorizza
    X = vectorizer.transform([text_clean])
    
    # Predici department
    dept_pred = dept_model.predict(X)[0]
    dept_proba = dept_model.predict_proba(X)[0]
    dept_confidence = max(dept_proba)
    dept_uncertain = dept_confidence < confidence_threshold
    
    # Predici sentiment
    sent_pred = sent_model.predict(X)[0]
    sent_proba = sent_model.predict_proba(X)[0]
    sent_confidence = max(sent_proba)
    sent_uncertain = sent_confidence < confidence_threshold
    
    # Mappa probabilità per sentiment
    sent_classes = sent_model.classes_
    if sent_pred == 'positive':
        positive_prob = sent_proba[list(sent_classes).index('positive')]
    else:
        positive_prob = sent_proba[list(sent_classes).index('positive')]
    
    return {
        'department': dept_pred,
        'department_confidence': dept_confidence,
        'department_uncertain': dept_uncertain,
        'sentiment': sent_pred,
        'sentiment_confidence': sent_confidence,
        'sentiment_uncertain': sent_uncertain,
        'positive_probability': positive_prob,
        'text_original': text,
        'text_processed': text_clean
    }


def predict_batch(df, vectorizer, dept_model, sent_model, use_advanced=False, confidence_threshold=0.7):
    """
    Predice reparto e sentiment per un batch di recensioni
    
    Args:
        df: DataFrame con colonne 'title' e 'body'
        vectorizer: TfidfVectorizer fitted
        dept_model: modello department
        sent_model: modello sentiment
        use_advanced: usa preprocessing avanzato (spaCy)
        confidence_threshold: soglia per marcare predizioni incerte
    
    Returns:
        DataFrame con predizioni aggiunte
    """
    # Combina e preprocessa
    df['text'] = df.apply(lambda row: combine_title_body(row['title'], row['body']), axis=1)
    df['text_clean'] = df['text'].apply(lambda x: preprocess_text(x, use_advanced=use_advanced))
    
    # Vectorizza
    X = vectorizer.transform(df['text_clean'])
    
    # Predici department
    df['predicted_department'] = dept_model.predict(X)
    dept_proba = dept_model.predict_proba(X)
    df['department_confidence'] = dept_proba.max(axis=1)
    df['department_uncertain'] = df['department_confidence'] < confidence_threshold
    
    # Predici sentiment
    df['predicted_sentiment'] = sent_model.predict(X)
    sent_proba = sent_model.predict_proba(X)
    df['sentiment_confidence'] = sent_proba.max(axis=1)
    df['sentiment_uncertain'] = df['sentiment_confidence'] < confidence_threshold
    
    # Probabilità positive
    sent_classes = sent_model.classes_
    positive_idx = list(sent_classes).index('positive')
    df['positive_probability'] = sent_proba[:, positive_idx]
    
    return df
