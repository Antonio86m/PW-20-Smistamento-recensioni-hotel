"""
Pipeline ML per Classificazione Recensioni Alberghiere
Training, Valutazione e Salvataggio Modelli
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from utils import preprocess_text, combine_title_body


def load_and_prepare_data(filepath='dataset_recensioni.csv'):
    """
    Carica e prepara i dati per il training
    
    Args:
        filepath: percorso del file CSV
    
    Returns:
        DataFrame preparato
    """
    print("📂 Caricamento dataset...")
    df = pd.read_csv(filepath)
    
    print(f"✓ Caricati {len(df)} recensioni")
    print(f"  - Reparti: {df['department'].nunique()}")
    print(f"  - Sentiment: {df['sentiment'].nunique()}")
    
    # Combina titolo e corpo
    df['text'] = df.apply(lambda row: combine_title_body(row['title'], row['body']), axis=1)
    
    # Preprocessing
    print("\n🔧 Preprocessing testo...")
    df['text_clean'] = df['text'].apply(preprocess_text)
    
    return df


def train_vectorizer(X_train):
    """
    Crea e addestra TF-IDF Vectorizer
    
    Args:
        X_train: lista di testi di training
    
    Returns:
        vectorizer fitted
    """
    print("\n📊 Creazione TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    
    vectorizer.fit(X_train)
    print(f"✓ Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    return vectorizer


def train_department_classifier(X_train, y_train, X_test, y_test):
    """
    Addestra classificatore per reparto
    
    Args:
        X_train, y_train: dati training
        X_test, y_test: dati test
    
    Returns:
        modello addestrato e metriche
    """
    print("\n🎯 Training Department Classifier...")
    
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        C=1.0
    )
    
    model.fit(X_train, y_train)
    
    # Predizioni
    y_pred = model.predict(X_test)
    
    # Metriche
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"✓ Accuracy: {accuracy:.4f}")
    print(f"✓ F1-Score (macro): {f1_macro:.4f}")
    print(f"✓ F1-Score (weighted): {f1_weighted:.4f}")
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'y_test': y_test,
        'y_pred': y_pred
    }
    
    return model, metrics


def train_sentiment_classifier(X_train, y_train, X_test, y_test):
    """
    Addestra classificatore per sentiment
    
    Args:
        X_train, y_train: dati training
        X_test, y_test: dati test
    
    Returns:
        modello addestrato e metriche
    """
    print("\n😊 Training Sentiment Classifier...")
    
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        C=1.0
    )
    
    model.fit(X_train, y_train)
    
    # Predizioni
    y_pred = model.predict(X_test)
    
    # Metriche
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"✓ Accuracy: {accuracy:.4f}")
    print(f"✓ F1-Score (macro): {f1_macro:.4f}")
    print(f"✓ F1-Score (weighted): {f1_weighted:.4f}")
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'y_test': y_test,
        'y_pred': y_pred
    }
    
    return model, metrics


def plot_confusion_matrix(cm, labels, title, filename):
    """
    Crea e salva confusion matrix
    
    Args:
        cm: confusion matrix
        labels: etichette classi
        title: titolo grafico
        filename: nome file output
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Salvato: {filename}")


def analyze_errors(df_test, y_test, y_pred, task_name):
    """
    Analizza errori di classificazione
    
    Args:
        df_test: DataFrame di test
        y_test: etichette vere
        y_pred: predizioni
        task_name: nome del task ('department' o 'sentiment')
    
    Returns:
        lista di esempi di errori
    """
    print(f"\n🔍 Analisi Errori - {task_name.upper()}")
    
    # Trova errori
    errors_idx = np.where(y_test != y_pred)[0]
    
    if len(errors_idx) == 0:
        print("✓ Nessun errore trovato!")
        return []
    
    print(f"Errori totali: {len(errors_idx)} su {len(y_test)} ({len(errors_idx)/len(y_test)*100:.1f}%)")
    
    # Mostra alcuni esempi
    error_examples = []
    for idx in errors_idx[:5]:
        example = {
            'text': df_test.iloc[idx]['text'][:100] + '...',
            'true_label': y_test.iloc[idx],
            'predicted_label': y_pred[idx]
        }
        error_examples.append(example)
        print(f"\n  Testo: {example['text']}")
        print(f"  Vero: {example['true_label']} | Predetto: {example['predicted_label']}")
    
    return error_examples


def save_models(vectorizer, dept_model, sent_model, output_dir='models'):
    """
    Salva i modelli addestrati
    
    Args:
        vectorizer: TF-IDF vectorizer
        dept_model: modello department
        sent_model: modello sentiment
        output_dir: directory output
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n💾 Salvataggio modelli in {output_dir}/...")
    
    with open(f'{output_dir}/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    with open(f'{output_dir}/department_model.pkl', 'wb') as f:
        pickle.dump(dept_model, f)
    
    with open(f'{output_dir}/sentiment_model.pkl', 'wb') as f:
        pickle.dump(sent_model, f)
    
    print("✓ Modelli salvati con successo!")


def main():
    """
    Pipeline principale di training
    """
    print("=" * 60)
    print("  PIPELINE ML - CLASSIFICAZIONE RECENSIONI ALBERGHIERE")
    print("=" * 60)
    
    # 1. Carica dati
    df = load_and_prepare_data('dataset_recensioni.csv')
    
    # 2. Split train/test
    print("\n📊 Split Train/Test (80/20)...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['department'])
    print(f"  Training set: {len(train_df)} recensioni")
    print(f"  Test set: {len(test_df)} recensioni")
    
    # 3. Vectorizer
    vectorizer = train_vectorizer(train_df['text_clean'].values)
    
    # Trasforma dati
    X_train = vectorizer.transform(train_df['text_clean'].values)
    X_test = vectorizer.transform(test_df['text_clean'].values)
    
    # 4. Train Department Classifier
    dept_model, dept_metrics = train_department_classifier(
        X_train, train_df['department'].values,
        X_test, test_df['department'].values
    )
    
    # Confusion Matrix Department
    os.makedirs('results', exist_ok=True)
    plot_confusion_matrix(
        dept_metrics['confusion_matrix'],
        sorted(df['department'].unique()),
        'Confusion Matrix - Department Classification',
        'results/confusion_matrix_department.png'
    )
    
    # Analisi errori department
    dept_errors = analyze_errors(
        test_df, 
        test_df['department'],
        dept_metrics['y_pred'],
        'department'
    )
    
    # 5. Train Sentiment Classifier
    sent_model, sent_metrics = train_sentiment_classifier(
        X_train, train_df['sentiment'].values,
        X_test, test_df['sentiment'].values
    )
    
    # Confusion Matrix Sentiment
    plot_confusion_matrix(
        sent_metrics['confusion_matrix'],
        sorted(df['sentiment'].unique()),
        'Confusion Matrix - Sentiment Classification',
        'results/confusion_matrix_sentiment.png'
    )
    
    # Analisi errori sentiment
    sent_errors = analyze_errors(
        test_df,
        test_df['sentiment'],
        sent_metrics['y_pred'],
        'sentiment'
    )
    
    # 6. Salva modelli
    save_models(vectorizer, dept_model, sent_model)
    
    # 7. Salva metriche
    metrics_summary = {
        'department': {
            'accuracy': float(dept_metrics['accuracy']),
            'f1_macro': float(dept_metrics['f1_macro']),
            'f1_weighted': float(dept_metrics['f1_weighted']),
            'error_examples': dept_errors
        },
        'sentiment': {
            'accuracy': float(sent_metrics['accuracy']),
            'f1_macro': float(sent_metrics['f1_macro']),
            'f1_weighted': float(sent_metrics['f1_weighted']),
            'error_examples': sent_errors
        }
    }
    
    with open('results/metrics.pkl', 'wb') as f:
        pickle.dump(metrics_summary, f)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETATO CON SUCCESSO!")
    print("=" * 60)
    print(f"\nModelli salvati in: models/")
    print(f"Risultati salvati in: results/")
    print(f"\nDepartment - Accuracy: {dept_metrics['accuracy']:.4f} | F1: {dept_metrics['f1_macro']:.4f}")
    print(f"Sentiment - Accuracy: {sent_metrics['accuracy']:.4f} | F1: {sent_metrics['f1_macro']:.4f}")
    
    return dept_model, sent_model, vectorizer, metrics_summary


if __name__ == '__main__':
    dept_model, sent_model, vectorizer, metrics = main()
