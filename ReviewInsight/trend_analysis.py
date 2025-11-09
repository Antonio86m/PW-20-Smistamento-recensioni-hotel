"""
Analisi Trend Temporali per Recensioni Alberghiere
Visualizza pattern e tendenze nei reclami/sentiment nel tempo
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
import numpy as np

random.seed(42)
np.random.seed(42)

def generate_time_series_data(df, start_date='2024-01-01', days=90):
    """
    Aggiunge timestamp casuali alle recensioni per simulare distribuzione temporale
    
    Args:
        df: DataFrame con recensioni
        start_date: data inizio periodo
        days: numero giorni da simulare
    
    Returns:
        DataFrame con colonna 'date' aggiunta
    """
    
    start = pd.to_datetime(start_date)
    date_range = pd.date_range(start=start, periods=days, freq='D')
    
    # Assegna date casuali
    df_copy = df.copy()
    df_copy['date'] = [random.choice(date_range) for _ in range(len(df))]
    df_copy = df_copy.sort_values('date')
    
    return df_copy


def create_trend_visualizations(df, output_dir='results'):
    """
    Crea visualizzazioni per analisi trend temporali
    
    Args:
        df: DataFrame con recensioni e timestamp
        output_dir: directory output per salvare grafici
    """
    
    # Assicura che esista la colonna date
    if 'date' not in df.columns:
        df = generate_time_series_data(df)
    
    # Setup stile
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # 1. Trend Sentiment nel Tempo
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Grafico 1: Volume recensioni per sentiment nel tempo
    sentiment_time = df.groupby([pd.Grouper(key='date', freq='W'), 'sentiment']).size().unstack(fill_value=0)
    sentiment_time.plot(ax=axes[0, 0], marker='o', linewidth=2)
    axes[0, 0].set_title('Volume Recensioni per Sentiment (Settimanale)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Data', fontsize=11)
    axes[0, 0].set_ylabel('Numero Recensioni', fontsize=11)
    axes[0, 0].legend(title='Sentiment')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Grafico 2: Percentuale sentiment nel tempo
    sentiment_pct = sentiment_time.div(sentiment_time.sum(axis=1), axis=0) * 100
    sentiment_pct.plot(ax=axes[0, 1], kind='area', stacked=True, alpha=0.7)
    axes[0, 1].set_title('Distribuzione % Sentiment nel Tempo', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Data', fontsize=11)
    axes[0, 1].set_ylabel('Percentuale (%)', fontsize=11)
    axes[0, 1].legend(title='Sentiment', loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Grafico 3: Trend reclami per reparto
    dept_time = df[df['sentiment'] == 'negative'].groupby([pd.Grouper(key='date', freq='W'), 'department']).size().unstack(fill_value=0)
    dept_time.plot(ax=axes[1, 0], marker='s', linewidth=2)
    axes[1, 0].set_title('Reclami (Negative) per Reparto nel Tempo', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Data', fontsize=11)
    axes[1, 0].set_ylabel('Numero Reclami', fontsize=11)
    axes[1, 0].legend(title='Reparto')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Grafico 4: Heatmap sentiment per reparto e settimana
    df['week'] = df['date'].dt.isocalendar().week
    heatmap_data = df.pivot_table(
        values='sentiment',
        index='department',
        columns='week',
        aggfunc=lambda x: (x == 'negative').sum()
    ).fillna(0)
    
    sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[1, 1], cbar_kws={'label': 'Reclami'})
    axes[1, 1].set_title('Heatmap Reclami per Reparto e Settimana', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Settimana dell\'Anno', fontsize=11)
    axes[1, 1].set_ylabel('Reparto', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/trend_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Salvato: {output_dir}/trend_analysis.png")
    plt.close()
    
    # 2. Analisi Dettagliata Reclami
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Grafico 5: Top problemi per reparto
    negative_reviews = df[df['sentiment'] == 'negative']
    dept_counts = negative_reviews['department'].value_counts()
    
    colors = ['#ff6b6b', '#ee5a6f', '#c44569']
    dept_counts.plot(ax=axes[0], kind='barh', color=colors)
    axes[0].set_title('Distribuzione Reclami per Reparto', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Numero Reclami', fontsize=11)
    axes[0].set_ylabel('Reparto', fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Aggiungi valori sulle barre
    for i, v in enumerate(dept_counts):
        axes[0].text(v + 0.1, i, str(int(v)), va='center')
    
    # Grafico 6: Trend positività per reparto
    positive_reviews = df[df['sentiment'] == 'positive']
    dept_positive = positive_reviews['department'].value_counts()
    
    colors_pos = ['#4ecdc4', '#45b7d1', '#5f27cd']
    dept_positive.plot(ax=axes[1], kind='barh', color=colors_pos)
    axes[1].set_title('Distribuzione Recensioni Positive per Reparto', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Numero Recensioni Positive', fontsize=11)
    axes[1].set_ylabel('Reparto', fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # Aggiungi valori sulle barre
    for i, v in enumerate(dept_positive):
        axes[1].text(v + 0.1, i, str(int(v)), va='center')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/complaint_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Salvato: {output_dir}/complaint_analysis.png")
    plt.close()
    
    # 3. Statistiche Riepilogative
    print("\n" + "="*60)
    print("  ANALISI TREND TEMPORALI - STATISTICHE")
    print("="*60)
    
    print(f"\n📅 Periodo Analizzato: {df['date'].min().date()} - {df['date'].max().date()}")
    print(f"📊 Totale Recensioni: {len(df)}")
    
    print("\n--- Distribuzione Sentiment ---")
    sentiment_dist = df['sentiment'].value_counts()
    for sent, count in sentiment_dist.items():
        pct = (count / len(df)) * 100
        print(f"{sent.capitalize()}: {count} ({pct:.1f}%)")
    
    print("\n--- Reclami per Reparto (Negative) ---")
    negative_dept = df[df['sentiment'] == 'negative']['department'].value_counts()
    for dept, count in negative_dept.items():
        total_dept = len(df[df['department'] == dept])
        pct = (count / total_dept) * 100
        print(f"{dept}: {count} reclami ({pct:.1f}% delle recensioni del reparto)")
    
    print("\n--- Media Recensioni per Settimana ---")
    weekly_avg = df.groupby(pd.Grouper(key='date', freq='W')).size().mean()
    print(f"Media: {weekly_avg:.1f} recensioni/settimana")
    
    # Calcola trend
    df_sorted = df.sort_values('date')
    first_half = df_sorted.iloc[:len(df_sorted)//2]
    second_half = df_sorted.iloc[len(df_sorted)//2:]
    
    first_neg_pct = (first_half['sentiment'] == 'negative').sum() / len(first_half) * 100
    second_neg_pct = (second_half['sentiment'] == 'negative').sum() / len(second_half) * 100
    
    print(f"\n--- Trend Reclami ---")
    print(f"Prima metà periodo: {first_neg_pct:.1f}% negative")
    print(f"Seconda metà periodo: {second_neg_pct:.1f}% negative")
    
    if second_neg_pct > first_neg_pct:
        diff = second_neg_pct - first_neg_pct
        print(f"⚠️ TREND IN PEGGIORAMENTO: +{diff:.1f}% reclami")
    elif second_neg_pct < first_neg_pct:
        diff = first_neg_pct - second_neg_pct
        print(f"✅ TREND IN MIGLIORAMENTO: -{diff:.1f}% reclami")
    else:
        print(f"➡️ TREND STABILE")
    
    print("="*60)
    
    return df


if __name__ == "__main__":
    # Carica dataset
    try:
        df = pd.read_csv('dataset_recensioni.csv')
        print("✓ Caricato dataset sintetico")
    except:
        print("❌ Errore: dataset_recensioni.csv non trovato")
        exit(1)
    
    # Genera time series
    df_ts = generate_time_series_data(df, start_date='2024-01-01', days=90)
    print(f"✓ Generati timestamp per {len(df_ts)} recensioni (90 giorni)")
    
    # Crea visualizzazioni
    df_result = create_trend_visualizations(df_ts)
    
    # Salva dataset con timestamp
    df_result.to_csv('dataset_with_timestamps.csv', index=False)
    print("\n✓ Salvato: dataset_with_timestamps.csv")
    print("\n✅ Analisi trend completata!")
