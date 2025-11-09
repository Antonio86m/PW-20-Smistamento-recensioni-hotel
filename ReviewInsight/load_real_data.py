"""
Simulatore di Dataset Reale di Recensioni Alberghiere
Crea recensioni realistiche simulate per testing del sistema
"""

import pandas as pd
import random
import csv

random.seed(42)

# Recensioni realistiche simulate (basate su pattern di recensioni vere)
REALISTIC_REVIEWS = [
    # Housekeeping - Positive
    ("Pulizia impeccabile", "La camera era davvero pulita, ogni dettaglio curato. Il letto era comodo e le lenzuola profumate. Il personale delle pulizie ha fatto un lavoro eccellente.", "Housekeeping",          "positive"),
    ("Camera sempre in ordine", "Durante il nostro soggiorno di 5 giorni, la camera è stata pulita quotidianamente con grande professionalità. Asciugamani sempre freschi.", "Housekeeping", "positive"),
    ("Ottimo comfort", "Letto king size comodissimo, camera spaziosa e luminosa. Tutto funzionante e pulito. Ci siamo trovati benissimo.", "Housekeeping", "positive"),
    
    # Housekeeping - Neutral
    ("Camera standard", "Camera nella media per la categoria. Pulizia accettabile ma niente di eccezionale. Va bene per un soggiorno breve.", "Housekeeping", "neutral"),
    ("Tutto ok", "Camera senza particolari pregi o difetti. Pulita quanto basta, arredamento un po' datato ma funzionale.", "Housekeeping", "neutral"),
    ("Nella norma", "Stanza conforme alle aspettative per il prezzo pagato. Pulizia discreta, comfort sufficiente.", "Housekeeping", "neutral"),
    
    # Housekeeping - Negative
    ("Delusione totale", "Camera sporca all'arrivo, polvere sui mobili e capelli nel bagno. Ho dovuto chiedere di ripulire.", "Housekeeping", "negative"),
    ("Manutenzione carente", "Camera vecchia con evidenti problemi di manutenzione. Macchie sul tappeto, rubinetti che perdono, aria condizionata rumorosa.", "Housekeeping", "negative"),
    ("Pessima pulizia", "Lenzuola con macchie, bagno sporco. Durante il soggiorno di 3 giorni non hanno mai pulito nonostante le richieste.", "Housekeeping", "negative"),
    
    # Reception - Positive
    ("Staff fantastico", "Il personale della reception è stato eccezionale! Ci hanno aiutato con consigli sui ristoranti e prenotazioni. Check-in rapidissimo.", "Reception", "positive"),
    ("Accoglienza calorosa", "Siamo stati accolti con grande professionalità e gentilezza. Hanno risolto subito un problema con la prenotazione.", "Reception", "positive"),
    ("Servizio di alto livello", "Reception sempre disponibile 24/7. Personale multilingue, competente e sorridente. Consigliatissimo!", "Reception", "positive"),
    
    # Reception - Neutral
    ("Servizio standard", "Check-in senza problemi ma niente di memorabile. Personale educato ma un po' distaccato.", "Reception", "neutral"),
    ("Reception funzionale", "Tutto ok alla reception, tempi di attesa accettabili. Nulla da segnalare in particolare.", "Reception", "neutral"),
    ("Nella media", "Staff cortese ma non particolarmente disponibile. Check-in veloce, check-out normale.", "Reception", "neutral"),
    
    # Reception - Negative
    ("Disorganizzazione totale", "Attesa di 45 minuti per il check-in con personale confuso e scortese. Hanno perso la prenotazione.", "Reception", "negative"),
    ("Pessima esperienza", "Staff maleducato e incompetente. Hanno dato informazioni sbagliate su orari e servizi. Mai più!", "Reception", "negative"),
    ("Servizio inesistente", "Reception chiusa in orari indicati come aperti. Nessuna assistenza durante il soggiorno. Deludente.", "Reception", "negative"),
    
    # F&B - Positive
    ("Colazione da 10", "Buffet colazione incredibile! Vasta scelta di prodotti freschi, dolci e salati. Personale attento e cordiale.", "F&B", "positive"),
    ("Ristorante top", "Cenato al ristorante dell'hotel e devo dire che è stato fantastico. Piatti gourmet, porzioni giuste, vini eccellenti.", "F&B", "positive"),
    ("Cibo eccellente", "La qualità del cibo è davvero alta. Colazione con prodotti tipici locali, tutto fresco e buonissimo.", "F&B", "positive"),
    
    # F&B - Neutral
    ("Colazione sufficiente", "Buffet nella media, scelta limitata ma prodotti discreti. Va bene per iniziare la giornata.", "F&B", "neutral"),
    ("Ristorante ok", "Cibo accettabile, niente di speciale ma nemmeno male. Prezzi in linea con la zona.", "F&B", "neutral"),
    ("Nella media", "Colazione standard da hotel, nulla di particolare. Prodotti industriali ma freschi.", "F&B", "neutral"),
    
    # F&B - Negative
    ("Colazione scarsissima", "Colazione molto deludente, poca scelta e prodotti di bassa qualità. Caffè imbevibile.", "F&B", "negative"),
    ("Ristorante pessimo", "Cibo freddo, servizio lentissimo e porzioni minuscole a prezzi esagerati. Meglio andare fuori.", "F&B", "negative"),
    ("Qualità scadente", "Prodotti scadenti a colazione, pane duro, frutta marcia. Il bar ha prezzi assurdi per la qualità offerta.", "F&B", "negative"),
    
    # Recensioni miste/complesse
    ("Esperienza contrastante", "Ottima posizione e camera pulita, ma personale poco disponibile. Colazione discreta ma troppo affollata.", "Reception", "neutral"),
    ("Bene e male", "Camera spaziosa e pulita ma rumorosa. Staff gentile ma check-in molto lento. Colazione buona.", "Housekeeping", "neutral"),
    ("Aspettative non soddisfatte", "Per il prezzo pagato mi aspettavo di più. Camera ok ma servizio reception disorganizzato. Colazione nella media.", "Reception", "negative"),
    ("Ottimo rapporto qualità-prezzo", "Non è un 5 stelle ma per il prezzo è perfetto. Camera pulita, personale gentile, colazione abbondante.", "Housekeeping", "positive"),
    ("Soggiorno piacevole", "Nel complesso una bella esperienza. Qualche piccola pecca ma nulla di grave. Lo consiglierei.", "Reception", "positive"),
    ("Migliorabile", "L'hotel ha potenziale ma servizio colazione da rivedere. Camera ok, staff gentile ma poco formato.", "F&B", "neutral"),
    
    # Recensioni specifiche e dettagliate
    ("Attenzione ai dettagli", "Le cameriere sono molto attente. Hanno notato che preferivo acqua frizzante e me l'hanno lasciata in camera ogni giorno. Piccoli gesti che fanno la differenza!",                   "Housekeeping", "positive"),
    ("Problema risolto velocemente", "Il primo giorno l'aria condizionata non funzionava ma hanno mandato subito un tecnico. Camera pulitissima e letto comodissimo.", "Housekeeping", "positive"),
    ("Colazione con vista", "La sala colazione ha una vista fantastica! Cibo buono, caffè eccellente, servizio al tavolo impeccabile.", "F&B", "positive"),
    ("Reception professionale", "Check-in alle 2 di notte senza problemi. Il ragazzo della reception ci ha dato una mappa e consigli utilissimi per visitare la città.", "Reception", "positive"),
    
    # Recensioni negative specifiche
    ("Rumore insopportabile", "Camera affacciata su strada trafficata, impossibile dormire. Nessun isolamento acustico. Inoltre scarsa pulizia.", "Housekeeping", "negative"),
    ("Promesse non mantenute", "Sul sito promettevano servizi che non esistono. La reception ha dato informazioni false sui trasporti.", "Reception", "negative"),
    ("Cibo di pessima qualità", "A colazione prodotti confezionati scadenti. Il ristorante serve piatti surgelati riscaldati male. Prezzi altissimi.", "F&B", "negative"),
    
    # Recensioni con confronti
    ("Meglio di quanto pensassi", "Ero scettico leggendo alcune recensioni negative, ma mi sono ricreduto. Camera pulita, staff cortese, ottima colazione.", "Housekeeping", "positive"),
    ("Aspettative troppo alte", "Dopo aver letto ottime recensioni sono rimasto deluso. Camera nella media, servizio lento.", "Reception", "neutral"),
]


def load_realistic_dataset(output_file='dataset_realistic.csv'):
    """
    Crea un dataset realistico simulato di recensioni
    
    Args:
        output_file: nome file CSV di output
    
    Returns:
        DataFrame con le recensioni
    """
    
    reviews_data = []
    
    for idx, (title, body, department, sentiment) in enumerate(REALISTIC_REVIEWS, 1):
        reviews_data.append({
            'id': idx,
            'title': title,
            'body': body,
            'department': department,
            'sentiment': sentiment
        })
    
    df = pd.DataFrame(reviews_data)
    
    # Salva CSV
    df.to_csv(output_file, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
    
    print(f"✓ Dataset realistico creato: {len(df)} recensioni")
    print(f"✓ File salvato: {output_file}")
    print()
    print("--- Distribuzione Reparti ---")
    print(df['department'].value_counts())
    print()
    print("--- Distribuzione Sentiment ---")
    print(df['sentiment'].value_counts())
    print()
    print("--- Prime 5 recensioni ---")
    print(df[['title', 'department', 'sentiment']].head())
    
    return df


if __name__ == "__main__":
    df = load_realistic_dataset()
    print("\n✓ Dataset realistico pronto per il testing!")
