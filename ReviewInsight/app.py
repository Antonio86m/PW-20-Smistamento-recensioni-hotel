"""
Dashboard Streamlit per Classificazione Recensioni Alberghiere
Predizione singola e batch processing con export CSV
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime
from utils import load_models, predict_single, predict_batch

# Configurazione pagina
st.set_page_config(
    page_title="Hotel Review Classifier",
    page_icon="🏨",
    layout="wide"
)

# CSS personalizzato
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .positive {
        color: #28a745;
        font-weight: bold;
    }
    .negative {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_models():
    """Carica modelli (cached)"""
    if not os.path.exists('models/vectorizer.pkl'):
        return None, None, None
    
    try:
        vectorizer, dept_model, sent_model = load_models()
        return vectorizer, dept_model, sent_model
    except Exception as e:
        st.error(f"Errore caricamento modelli: {e}")
        return None, None, None


def main():
    """Applicazione principale"""
    
    st.markdown('<div class="main-header">🏨 Sistema di Classificazione Recensioni Alberghiere</div>', 
                unsafe_allow_html=True)
    
    # Sidebar per navigazione
    st.sidebar.title("📊 Navigazione")
    page = st.sidebar.radio(
        "Seleziona funzionalità:",
        ["🔍 Classificazione Recensioni", "📈 Analisi Trend Temporali"]
    )
    
    if page == "📈 Analisi Trend Temporali":
        show_trend_analysis()
        return
    
    # Pagina principale - Classificazione
    st.markdown("""
    Questo sistema classifica automaticamente le recensioni degli ospiti assegnandole al **reparto competente** 
    e rilevando il **sentiment** (positivo/neutro/negativo).
    
    **Reparti:**
    - 🛏️ **Housekeeping** - Pulizia camere, manutenzione, comfort
    - 🎫 **Reception** - Check-in/out, staff, servizio clienti
    - 🍽️ **F&B** - Colazione, ristorante, bar, servizio ristorazione
    
    **Sentiment:**
    - 😊 **Positive** - Esperienza molto positiva
    - 😐 **Neutral** - Esperienza nella media, nulla di particolare
    - 😞 **Negative** - Esperienza negativa, problemi rilevati
    """)
    
    # Carica modelli
    vectorizer, dept_model, sent_model = load_trained_models()
    
    if vectorizer is None:
        st.error("⚠️ Modelli non trovati! Esegui prima `python train_model.py`")
        st.stop()
    
    st.success("✅ Modelli caricati con successo!")
    
    # Configurazione globale sidebar
    with st.sidebar:
        st.header("⚙️ Configurazione")
        
        use_advanced = st.checkbox(
            "Usa preprocessing avanzato (spaCy)",
            value=False,
            help="Attiva lemmatizzazione e rimozione stopwords italiane"
        )
        
        confidence_threshold = st.slider(
            "Soglia Confidenza",
            min_value=0.5,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Predizioni sotto questa soglia sono marcate come incerte"
        )
        
        st.markdown("---")
        st.markdown("**Impostazioni Attive:**")
        st.write(f"- Preprocessing: {'Avanzato (spaCy)' if use_advanced else 'Basilare'}")
        st.write(f"- Soglia: {confidence_threshold:.0%}")
    
    # Tabs per le diverse funzionalità
    tab1, tab2 = st.tabs(["🔍 Predizione Singola", "📊 Batch Processing"])
    
    # ========== TAB 1: PREDIZIONE SINGOLA ==========
    with tab1:
        st.header("Predizione Singola")
        st.markdown("Inserisci titolo e testo di una recensione per ottenere classificazione e sentiment.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            title_input = st.text_input(
                "Titolo Recensione",
                placeholder="es. Camera pulita e confortevole"
            )
        
        with col2:
            body_input = st.text_area(
                "Testo Recensione",
                placeholder="es. La camera era pulitissima, letto comodo e tutto perfettamente in ordine...",
                height=100
            )
        
        if st.button("🔍 Analizza Recensione", type="primary"):
            if not title_input.strip() or not body_input.strip():
                st.warning("⚠️ Inserisci sia titolo che testo della recensione")
            else:
                with st.spinner("Analisi in corso..."):
                    result = predict_single(
                        title_input, 
                        body_input, 
                        vectorizer, 
                        dept_model, 
                        sent_model,
                        use_advanced=use_advanced,
                        confidence_threshold=confidence_threshold
                    )
                
                st.success("✅ Analisi completata!")
                
                # Avviso se incerto
                if result['department_uncertain'] or result['sentiment_uncertain']:
                    st.warning("⚠️ Attenzione: Confidenza bassa rilevata. Si consiglia revisione manuale.")
                
                # Risultati
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 🎯 Reparto Assegnato")
                    dept_emoji = {
                        'Housekeeping': '🛏️',
                        'Reception': '🎫',
                        'F&B': '🍽️'
                    }
                    st.markdown(f"## {dept_emoji.get(result['department'], '📌')} {result['department']}")
                    st.metric("Confidenza", f"{result['department_confidence']*100:.1f}%")
                    if result['department_uncertain']:
                        st.error("🚨 Predizione incerta!")
                
                with col2:
                    st.markdown("### 😊 Sentiment")
                    # Emoji per sentiment
                    sent_emoji_map = {'positive': '😊', 'neutral': '😐', 'negative': '😞'}
                    sent_emoji = sent_emoji_map.get(result['sentiment'], '❓')
                    
                    # Colore per sentiment  
                    sent_class = 'positive' if result['sentiment'] == 'positive' else 'negative' if result['sentiment'] == 'negative' else ''
                    st.markdown(f"## {sent_emoji} <span class='{sent_class}'>{result['sentiment'].upper()}</span>", 
                               unsafe_allow_html=True)
                    st.metric("Confidenza", f"{result['sentiment_confidence']*100:.1f}%")
                    if result['sentiment_uncertain']:
                        st.error("🚨 Predizione incerta!")
                
                with col3:
                    st.markdown("### 📊 Probabilità Positivo")
                    st.markdown(f"## {result['positive_probability']*100:.1f}%")
                    st.progress(result['positive_probability'])
                
                # Mostra testo preprocessato
                with st.expander("🔧 Dettagli Preprocessing"):
                    st.markdown("**Testo Originale:**")
                    st.text(result['text_original'])
                    st.markdown("**Testo Preprocessato:**")
                    st.code(result['text_processed'])
    
    # ========== TAB 2: BATCH PROCESSING ==========
    with tab2:
        st.header("Batch Processing - Carica CSV")
        st.markdown("""
        Carica un file CSV con colonne **title** e **body** per ottenere predizioni in batch.
        Il sistema genererà un file CSV con tutte le predizioni e un timestamp.
        """)
        
        # Esempio formato
        with st.expander("📋 Formato CSV Richiesto"):
            example_df = pd.DataFrame({
                'title': ['Camera pulita', 'Check-in lento', 'Colazione ottima'],
                'body': [
                    'La camera era pulitissima e comoda',
                    'Attesa di 40 minuti per il check-in',
                    'Buffet ricco e prodotti freschi'
                ]
            })
            st.dataframe(example_df)
            
            # Download esempio
            csv_example = example_df.to_csv(index=False, encoding='utf-8')
            st.download_button(
                label="⬇️ Scarica CSV Esempio",
                data=csv_example,
                file_name="esempio_recensioni.csv",
                mime="text/csv"
            )
        
        # Upload file
        uploaded_file = st.file_uploader(
            "Carica file CSV",
            type=['csv'],
            help="File CSV con colonne 'title' e 'body'"
        )
        
        if uploaded_file is not None:
            try:
                # Leggi CSV
                df_upload = pd.read_csv(uploaded_file)
                
                # Verifica colonne
                if 'title' not in df_upload.columns or 'body' not in df_upload.columns:
                    st.error("⚠️ Il CSV deve contenere le colonne 'title' e 'body'")
                else:
                    st.success(f"✅ File caricato: {len(df_upload)} recensioni")
                    
                    # Mostra preview
                    with st.expander("👀 Preview Dati"):
                        st.dataframe(df_upload.head(10))
                    
                    # Bottone predizione
                    if st.button("🚀 Esegui Predizioni Batch", type="primary"):
                        with st.spinner(f"Elaborazione di {len(df_upload)} recensioni..."):
                            # Predici
                            df_result = predict_batch(
                                df_upload.copy(),
                                vectorizer,
                                dept_model,
                                sent_model,
                                use_advanced=use_advanced,
                                confidence_threshold=confidence_threshold
                            )
                            
                            # Seleziona colonne output
                            output_cols = [
                                'title', 'body',
                                'predicted_department', 'department_confidence', 'department_uncertain',
                                'predicted_sentiment', 'sentiment_confidence', 'sentiment_uncertain',
                                'positive_probability'
                            ]
                            
                            df_output = df_result[output_cols].copy()
                            
                            # Arrotonda valori
                            df_output['department_confidence'] = df_output['department_confidence'].round(4)
                            df_output['sentiment_confidence'] = df_output['sentiment_confidence'].round(4)
                            df_output['positive_probability'] = df_output['positive_probability'].round(4)
                        
                        st.success("✅ Predizioni completate!")
                        
                        # Avviso predizioni incerte
                        uncertain_count = df_result['department_uncertain'].sum() + df_result['sentiment_uncertain'].sum()
                        if uncertain_count > 0:
                            st.warning(f"⚠️ {uncertain_count} predizioni incerte rilevate. Controlla colonne *_uncertain.")
                        
                        # Statistiche
                        st.markdown("### 📊 Statistiche Risultati")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**Distribuzione Reparti:**")
                            dept_counts = df_output['predicted_department'].value_counts()
                            for dept, count in dept_counts.items():
                                st.write(f"- {dept}: {count} ({count/len(df_output)*100:.1f}%)")
                        
                        with col2:
                            st.markdown("**Distribuzione Sentiment:**")
                            sent_counts = df_output['predicted_sentiment'].value_counts()
                            for sent, count in sent_counts.items():
                                st.write(f"- {sent}: {count} ({count/len(df_output)*100:.1f}%)")
                        
                        with col3:
                            st.markdown("**Predizioni Incerte:**")
                            dept_unc = df_result['department_uncertain'].sum()
                            sent_unc = df_result['sentiment_uncertain'].sum()
                            st.write(f"- Reparto: {dept_unc} ({dept_unc/len(df_output)*100:.1f}%)")
                            st.write(f"- Sentiment: {sent_unc} ({sent_unc/len(df_output)*100:.1f}%)")
                        
                        # Mostra risultati
                        st.markdown("### 📋 Risultati Predizioni")
                        st.dataframe(df_output, height=400)
                        
                        # Export CSV con timestamp
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        csv_output = df_output.to_csv(index=False, encoding='utf-8')
                        
                        st.download_button(
                            label="⬇️ Scarica Risultati CSV",
                            data=csv_output,
                            file_name=f"predizioni_recensioni_{timestamp}.csv",
                            mime="text/csv",
                            type="primary"
                        )
                        
                        st.info(f"💾 File pronto: predizioni_recensioni_{timestamp}.csv")
            
            except Exception as e:
                st.error(f"❌ Errore elaborazione file: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Sistema di Classificazione Recensioni Alberghiere | ML Pipeline con Logistic Regression
    </div>
    """, unsafe_allow_html=True)


def show_trend_analysis():
    """Mostra pagina analisi trend temporali"""
    
    st.markdown("## 📈 Analisi Trend Temporali")
    st.markdown("""
    Analizza l'andamento dei reclami e del sentiment nel tempo per identificare pattern e tendenze.
    """)
    
    # Verifica se esistono le visualizzazioni
    trend_img = 'results/trend_analysis.png'
    complaint_img = 'results/complaint_analysis.png'
    
    if not os.path.exists(trend_img) or not os.path.exists(complaint_img):
        st.warning("⚠️ Le visualizzazioni dei trend non sono ancora state generate.")
        st.info("Esegui `python trend_analysis.py` per generare le analisi temporali.")
        
        if st.button("🔄 Genera Analisi Trend"):
            with st.spinner("Generazione analisi in corso..."):
                import subprocess
                result = subprocess.run(['python', 'trend_analysis.py'], capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("✅ Analisi trend generate con successo!")
                    st.rerun()
                else:
                    st.error(f"❌ Errore durante la generazione: {result.stderr}")
        return
    
    # Mostra statistiche riepilogative
    st.markdown("### 📊 Statistiche Periodo")
    
    # Carica dataset con timestamp
    if os.path.exists('dataset_with_timestamps.csv'):
        df = pd.read_csv('dataset_with_timestamps.csv')
        df['date'] = pd.to_datetime(df['date'])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Totale Recensioni", len(df))
        
        with col2:
            negative_pct = (df['sentiment'] == 'negative').sum() / len(df) * 100
            st.metric("% Negative", f"{negative_pct:.1f}%", 
                     delta=f"{negative_pct - 33.3:.1f}%" if negative_pct != 33.3 else None,
                     delta_color="inverse")
        
        with col3:
            positive_pct = (df['sentiment'] == 'positive').sum() / len(df) * 100
            st.metric("% Positive", f"{positive_pct:.1f}%",
                     delta=f"{positive_pct - 33.3:.1f}%" if positive_pct != 33.3 else None)
        
        with col4:
            neutral_pct = (df['sentiment'] == 'neutral').sum() / len(df) * 100
            st.metric("% Neutral", f"{neutral_pct:.1f}%")
        
        st.markdown(f"**Periodo:** {df['date'].min().date()} - {df['date'].max().date()}")
    
    # Mostra grafici trend
    st.markdown("### 📉 Trend Sentiment e Reclami")
    st.image(trend_img, use_container_width=True)
    
    st.markdown("### 📊 Analisi Dettagliata Reclami")
    st.image(complaint_img, use_container_width=True)
    
    # Insights e raccomandazioni
    st.markdown("### 💡 Insights e Raccomandazioni")
    
    if os.path.exists('dataset_with_timestamps.csv'):
        df_sorted = df.sort_values('date')
        first_half = df_sorted.iloc[:len(df_sorted)//2]
        second_half = df_sorted.iloc[len(df_sorted)//2:]
        
        first_neg_pct = (first_half['sentiment'] == 'negative').sum() / len(first_half) * 100
        second_neg_pct = (second_half['sentiment'] == 'negative').sum() / len(second_half) * 100
        
        if second_neg_pct > first_neg_pct + 2:
            st.error(f"""
            ⚠️ **ATTENZIONE**: Trend reclami in peggioramento!
            - Prima metà periodo: {first_neg_pct:.1f}% negative
            - Seconda metà periodo: {second_neg_pct:.1f}% negative
            - Variazione: +{second_neg_pct - first_neg_pct:.1f}%
            
            **Azioni consigliate:**
            - Analizzare le cause principali dei reclami recenti
            - Verificare eventuali cambiamenti operativi nel periodo
            - Pianificare interventi correttivi urgenti
            """)
        elif second_neg_pct < first_neg_pct - 2:
            st.success(f"""
            ✅ **OTTIMO**: Trend reclami in miglioramento!
            - Prima metà periodo: {first_neg_pct:.1f}% negative
            - Seconda metà periodo: {second_neg_pct:.1f}% negative
            - Variazione: {second_neg_pct - first_neg_pct:.1f}%
            
            **Azioni consigliate:**
            - Identificare le best practice implementate
            - Mantenere gli standard qualitativi raggiunti
            - Condividere strategie di successo con tutti i reparti
            """)
        else:
            st.info(f"""
            ➡️ **Trend stabile**: Reclami costanti nel tempo
            - Prima metà periodo: {first_neg_pct:.1f}% negative
            - Seconda metà periodo: {second_neg_pct:.1f}% negative
            
            **Azioni consigliate:**
            - Monitorare costantemente la qualità del servizio
            - Identificare aree di miglioramento potenziale
            - Implementare strategie proattive per ridurre i reclami
            """)
        
        # Analisi per reparto
        st.markdown("### 🎯 Focus per Reparto")
        
        negative_by_dept = df[df['sentiment'] == 'negative']['department'].value_counts()
        
        for dept in ['Housekeeping', 'Reception', 'F&B']:
            if dept in negative_by_dept.index:
                dept_total = len(df[df['department'] == dept])
                dept_neg = negative_by_dept[dept]
                dept_neg_pct = (dept_neg / dept_total) * 100
                
                if dept_neg_pct > 35:
                    st.warning(f"⚠️ **{dept}**: {dept_neg} reclami ({dept_neg_pct:.1f}%) - Richiede attenzione prioritaria")
                elif dept_neg_pct > 25:
                    st.info(f"ℹ️ **{dept}**: {dept_neg} reclami ({dept_neg_pct:.1f}%) - Monitorare attentamente")
                else:
                    st.success(f"✅ **{dept}**: {dept_neg} reclami ({dept_neg_pct:.1f}%) - Performance soddisfacente")


if __name__ == '__main__':
    main()
