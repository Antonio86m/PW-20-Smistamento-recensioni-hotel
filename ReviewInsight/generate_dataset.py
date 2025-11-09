"""
Generatore di Dataset Sintetico per Recensioni Alberghiere
Crea recensioni italiane con classificazione per reparto e sentiment
"""

import pandas as pd
import random
import csv

random.seed(42)

# Template di recensioni per ciascun reparto
HOUSEKEEPING_KEYWORDS = {
    'positive': [
        ("Camera impeccabile", "La camera era pulitissima, letto comodo e tutto perfettamente in ordine."),
        ("Pulizia eccellente", "Stanza sempre pulita e ordinata, cambio asciugamani ogni giorno."),
        ("Camera perfetta", "Camera spaziosa, pulita e ben organizzata. Complimenti al personale."),
        ("Letto comodo", "Il letto era comodissimo e la camera profumava di pulito."),
        ("Ottima pulizia", "Personale di pulizia molto efficiente, camera sempre impeccabile."),
        ("Stanza ben tenuta", "Camera pulita e ben curata, tutto funzionante e in ordine."),
        ("Pulizia giornaliera", "Apprezzo molto la pulizia quotidiana della camera, tutto perfetto."),
        ("Camera luminosa", "Stanza luminosa, pulita e accogliente. Ottimo servizio di pulizia."),
    ],
    'neutral': [
        ("Camera standard", "Camera normale, pulita quanto basta, niente di particolare."),
        ("Soggiorno nella norma", "Stanza accettabile, pulizia sufficiente, nulla da segnalare."),
        ("Camera ok", "Camera pulita ma un po' vecchia, nel complesso accettabile."),
        ("Nella media", "Stanza nella media, pulizia discreta, esperienza ok."),
        ("Niente di speciale", "Camera standard, pulizia sufficiente, prezzo nella norma."),
    ],
    'negative': [
        ("Camera sporca", "La camera non era pulita all'arrivo, polvere ovunque e bagno sporco."),
        ("Pulizia scarsa", "Scarsa pulizia della stanza, asciugamani non cambiati per giorni."),
        ("Bagno sporco", "Il bagno era sporco, presenza di capelli e macchie evidenti."),
        ("Camera disordinata", "Camera disordinata e sporca, lenzuola con macchie."),
        ("Mancanza pulizia", "Non hanno pulito la camera per tutto il soggiorno, inaccettabile."),
        ("Stanza trascurata", "Camera molto trascurata, polvere e cattivi odori."),
        ("Letto scomodo", "Letto scomodissimo e lenzuola sporche, pessima esperienza."),
        ("Camera degradata", "Camera in pessime condizioni, sporcizia e odore sgradevole."),
    ]
}

RECEPTION_KEYWORDS = {
    'positive': [
        ("Check-in veloce", "Check-in rapido ed efficiente, personale cordiale e disponibile."),
        ("Staff eccellente", "Personale della reception molto professionale e gentile."),
        ("Accoglienza perfetta", "Ottima accoglienza, ci hanno fatto sentire subito a nostro agio."),
        ("Check-out rapido", "Check-out velocissimo senza problemi, personale efficiente."),
        ("Personale gentile", "Reception sempre disponibile e cortese per qualsiasi richiesta."),
        ("Servizio impeccabile", "Servizio clienti impeccabile, personale molto preparato."),
        ("Ottima organizzazione", "Reception ben organizzata, nessun problema durante il soggiorno."),
        ("Staff disponibile", "Personale sempre pronto ad aiutare con il sorriso."),
    ],
    'neutral': [
        ("Servizio accettabile", "Personale della reception cortese, check-in normale."),
        ("Reception standard", "Servizio nella norma, niente di particolare da segnalare."),
        ("Check-in normale", "Check-in senza problemi, tempi di attesa accettabili."),
        ("Staff nella media", "Personale educato, servizio sufficiente."),
        ("Esperienza standard", "Reception funzionale, nessun problema particolare."),
    ],
    'negative': [
        ("Check-in lento", "Check-in lunghissimo, attesa di oltre 30 minuti alla reception."),
        ("Personale scortese", "Personale della reception poco cortese e disponibile."),
        ("Servizio pessimo", "Pessimo servizio, personale incompetente e maleducato."),
        ("Accoglienza fredda", "Accoglienza molto fredda, nessuna assistenza dal personale."),
        ("Attesa infinita", "Tempi di attesa assurdi per il check-in, personale lento."),
        ("Reception disorganizzata", "Reception completamente disorganizzata, caos totale."),
        ("Staff inefficiente", "Personale inefficiente, nessuna risposta alle nostre richieste."),
        ("Problemi prenotazione", "Problemi con la prenotazione, reception non ha saputo risolvere."),
    ]
}

FB_KEYWORDS = {
    'positive': [
        ("Colazione fantastica", "Colazione abbondante e varia, prodotti freschi e di qualità."),
        ("Ristorante ottimo", "Ristorante dell'hotel eccellente, piatti buonissimi e ben presentati."),
        ("Ottimo cibo", "Cibo delizioso e servizio al tavolo impeccabile."),
        ("Colazione ricca", "Buffet colazione molto ricco, c'è davvero di tutto."),
        ("Cucina eccellente", "Cucina di ottimo livello, piatti tipici preparati benissimo."),
        ("Bar fornitissimo", "Bar ben fornito, personale cordiale e prezzi giusti."),
        ("Pranzo buonissimo", "Pranzato al ristorante, tutto ottimo e porzioni generose."),
        ("Servizio ristorante", "Servizio al ristorante veloce e professionale, cibo eccellente."),
    ],
    'neutral': [
        ("Colazione sufficiente", "Colazione standard, scelta discreta, prodotti nella media."),
        ("Ristorante normale", "Cibo accettabile, niente di speciale ma nemmeno male."),
        ("Cibo nella media", "Ristorante con cibo ok, servizio sufficiente."),
        ("Colazione ok", "Buffet colazione nella norma, varietà accettabile."),
        ("Esperienza media", "Cibo e servizio al ristorante nella media, prezzi ok."),
    ],
    'negative': [
        ("Colazione scarsa", "Colazione molto scarsa, poca scelta e prodotti di bassa qualità."),
        ("Ristorante deludente", "Ristorante molto deludente, cibo freddo e poco gustoso."),
        ("Cibo scadente", "Cibo di pessima qualità, colazione immangiabile."),
        ("Colazione fredda", "Colazione sempre fredda, prodotti vecchi e poco freschi."),
        ("Servizio lento", "Servizio al ristorante lentissimo, cibo arrivato freddo."),
        ("Bar caro", "Prezzi del bar esagerati, qualità molto bassa."),
        ("Cucina mediocre", "Cucina mediocre, piatti senza sapore e mal presentati."),
        ("Colazione monotona", "Colazione monotona, sempre le stesse cose per giorni."),
    ]
}

# Recensioni ambigue (miste tra reparti)
AMBIGUOUS_REVIEWS = [
    ("Esperienza mista", "Camera pulita ma personale poco disponibile, colazione nella media.", "Reception", "negative"),
    ("Bene ma non benissimo", "Check-in veloce, però la stanza aveva dei problemi di pulizia.", "Housekeeping", "negative"),
    ("Soggiorno discreto", "Colazione buona ma camera piccola e rumorosa, staff ok.", "F&B", "neutral"),
    ("Alcune criticità", "Reception cortese ma tempi lunghi, camera pulita, cibo discreto.", "Reception", "neutral"),
    ("Nel complesso ok", "Buona pulizia camera, colazione sufficiente, personale a volte scortese.", "Housekeeping", "neutral"),
    ("Valutazione media", "Ristorante buono, però check-in caotico e camera non perfetta.", "F&B", "neutral"),
]


def generate_synthetic_dataset(n_samples=500, output_file='dataset_recensioni.csv'):
    """
    Genera dataset sintetico di recensioni alberghiere
    
    Args:
        n_samples: numero totale di recensioni da generare
        output_file: nome file CSV di output
    
    Returns:
        DataFrame con le recensioni generate
    """
    
    reviews = []
    review_id = 1
    
    # Calcola distribuzione per ottenere esattamente n_samples
    # Vogliamo circa 10% recensioni ambigue, ma arrotondiamo per ottenere numero esatto
    ambiguous_count = n_samples // 10
    regular_count = n_samples - ambiguous_count
    samples_per_dept = regular_count // 3
    
    # Housekeeping prende il remainder per garantire numero esatto
    housekeeping_count = samples_per_dept + (regular_count % 3)
    reception_count = samples_per_dept
    fb_count = samples_per_dept
    
    # Genera recensioni per Housekeeping
    for _ in range(housekeeping_count):
        # Distribuzione: 40% positive, 30% neutral, 30% negative
        rand_val = random.random()
        if rand_val < 0.4:
            sentiment = 'positive'
        elif rand_val < 0.7:
            sentiment = 'neutral'
        else:
            sentiment = 'negative'
        title, body = random.choice(HOUSEKEEPING_KEYWORDS[sentiment])
        
        # Aggiungi variazione al testo
        if random.random() < 0.3:
            body += " " + random.choice([
                "Consiglio vivamente questo hotel.",
                "Torneremo sicuramente.",
                "Non tornerò mai più.",
                "Esperienza da dimenticare.",
                "Rapporto qualità-prezzo ottimo.",
                "Prezzo troppo alto per quello che offre."
            ])
        
        reviews.append({
            'id': review_id,
            'title': title,
            'body': body,
            'department': 'Housekeeping',
            'sentiment': sentiment
        })
        review_id += 1
    
    # Genera recensioni per Reception
    for _ in range(reception_count):
        # Distribuzione: 40% positive, 30% neutral, 30% negative
        rand_val = random.random()
        if rand_val < 0.4:
            sentiment = 'positive'
        elif rand_val < 0.7:
            sentiment = 'neutral'
        else:
            sentiment = 'negative'
        title, body = random.choice(RECEPTION_KEYWORDS[sentiment])
        
        if random.random() < 0.3:
            body += " " + random.choice([
                "Personale molto professionale.",
                "Servizio clienti eccezionale.",
                "Mai più in questo hotel.",
                "Servizio decisamente migliorabile.",
                "Tutto perfetto dall'inizio alla fine.",
                "Disorganizzazione totale."
            ])
        
        reviews.append({
            'id': review_id,
            'title': title,
            'body': body,
            'department': 'Reception',
            'sentiment': sentiment
        })
        review_id += 1
    
    # Genera recensioni per F&B
    for _ in range(fb_count):
        # Distribuzione: 40% positive, 30% neutral, 30% negative
        rand_val = random.random()
        if rand_val < 0.4:
            sentiment = 'positive'
        elif rand_val < 0.7:
            sentiment = 'neutral'
        else:
            sentiment = 'negative'
        title, body = random.choice(FB_KEYWORDS[sentiment])
        
        if random.random() < 0.3:
            body += " " + random.choice([
                "Complimenti allo chef.",
                "Qualità eccellente dei prodotti.",
                "Cibo immangiabile.",
                "Servizio da rivedere completamente.",
                "Ottimo rapporto qualità prezzo.",
                "Prezzi eccessivi per la qualità offerta."
            ])
        
        reviews.append({
            'id': review_id,
            'title': title,
            'body': body,
            'department': 'F&B',
            'sentiment': sentiment
        })
        review_id += 1
    
    # Aggiungi recensioni ambigue (ciclando se necessario)
    for i in range(ambiguous_count):
        title, body, dept, sent = AMBIGUOUS_REVIEWS[i % len(AMBIGUOUS_REVIEWS)]
        reviews.append({
            'id': review_id,
            'title': title,
            'body': body,
            'department': dept,
            'sentiment': sent
        })
        review_id += 1
    
    # Crea DataFrame e mescola le righe
    df = pd.DataFrame(reviews)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Salva CSV
    df.to_csv(output_file, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
    
    print(f"✓ Dataset generato: {len(df)} recensioni")
    print(f"✓ File salvato: {output_file}")
    print("\n--- Distribuzione Reparti ---")
    print(df['department'].value_counts())
    print("\n--- Distribuzione Sentiment ---")
    print(df['sentiment'].value_counts())
    print("\n--- Esempi di Recensioni ---")
    print(df[['title', 'department', 'sentiment']].head(10))
    
    return df


if __name__ == '__main__':
    # Genera dataset con 500 recensioni
    df = generate_synthetic_dataset(n_samples=500, output_file='dataset_recensioni.csv')
    print(f"\n✓ Dataset pronto per il training!")
