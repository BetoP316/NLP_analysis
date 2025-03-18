from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd 
from functools import partial
import re
import spacy
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import sys
sys.path.append('/Users/beto/Desktop/')
#from sentiment_utils import compute_sentiment
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import zscore
from scipy.stats import shapiro, kstest
from scipy.stats import shapiro




#base_path = "C:/Users/FINANZAS/Desktop/DATOS/DATAFRAMES/"
base_path = "/Users/beto/Desktop/Research_paper/DATA/DATA/"
#base_path = "C:/Users/hpc/Desktop/DATA/"

for year in range(2001, 2025):
    file_path = f"{base_path}df_news_{year}.csv"
    df_name = f"df_news_{year}" 
    globals()[df_name] = pd.read_csv(file_path)
    print(f"{df_name} loaded successfully")
    
    
'''    
for year in range(2001, 2025):
    df_name = f"df_news_{year}"  # Name of the dataframe
    if df_name in globals():
        unique_sections = globals()[df_name]['sec'].unique()  # Get unique values in 'sec'
        print(f"Unique values in 'sec' for {df_name}:")
        print(unique_sections)
        print("\n")
    else:
        print(f"{df_name} does not exist.")
'''

#Removing Duplicates
for year in range(2001, 2025):
    df_name = f"df_news_{year}" 
    if df_name in globals():
        df = globals()[df_name]
        
        before_count = len(df)
        df.drop_duplicates(inplace=True)
        after_count = len(df)
        
        globals()[df_name] = df
        
        print(f"Processed {df_name}: Removed {before_count - after_count} duplicate rows.")
    else:
        print(f"{df_name} does not exist.")

'''
#Keep selected columns
for year in range(2001, 2025):
    df_name = f'df_news_{year}'
    if df_name in globals():
        df = globals()[df_name]
        
        df = df[['id', 'fecha', 'Section', 'texto', 'fuera_articulo']].rename(columns={'sec': 'Section'})
        
        globals()[df_name] = df
        
        # Print confirmation
        print(f"Updated columns for {df_name}. Only specified columns retained.")
    else:
        print(f"{df_name} does not exist in the current environment.")

'''
'''
keywords = {
    'ECONOMIA': [
        'economia', 'comercio', 'negocios', 'empresarial', 'laboral', 'sector productivo', 'empresa',
        'mercado', 'recursos humanos', 'innovacion', 'expectativas', 'hidrocarburos', 'acuerdo comercial', 
        'bolsa', 'tendencias de mercado', 'ecuador potencia camaronera', 'sector estrategico', 'exportaciones',
        'trabajo', 'superavit', 'deficit', 'inversion', 'banco', 'credito', 'financiero', 'fondo', 'mercantil',
        'capital', 'economico', 'deuda', 'ganancias', 'perdidas', 'inversor', 'importacion', 'exportacion',
        'inmobiliario', 'moneda', 'divisas', 'remesas', 'tarifa', 'arancel', 'fiscal', 'microcredito', 'dinero',
        'presupuesto', 'subsidio', 'recesion', 'inflacion', 'pib', 'impuesto', 'renta', 'finanzas', 'costos',
        'cierre economico', 'industrial', 'fondos', 'ganancia', 'banca', 'competitividad', 'subvencion',
        'economía circular', 'financiación', 'inversión extranjera', 'banca central', 'fondos de inversión'
    ],
    'SOCIEDAD': [
        'sociedad', 'seguridad ciudadana', 'comunidad', 'coyuntura', 'cultura', 'interculturalidad', 'educacion',
        'asamblea nacional', 'institucion del estado', 'problemática', 'ciencia', 'actualidad', 'noticias', 
        'hechos', 'opinion', 'nacional', 'panorama global', 'paises', 'bomberos', 'servicio',
        'comunitario', 'familia', 'ciudadania', 'vecinos', 'residencial', 'vecindario', 'evento', 'social',
        'iglesia', 'ong', 'voluntario', 'municipio', 'festival', 'campaña', 'fundacion', 'programa social',
        'proyecto comunitario', 'desarrollo social', 'juventud', 'discapacidad', 'inclusion', 'asistencia',
        'ciudad', 'civil', 'organizacion', 'vecino', 'ambiente', 'vida cotidiana', 'cultural', 'historia',
        'solidaridad', 'fiesta', 'familias', 'comunidades', 'festejo', 'union', 'inmigración', 'integración',
        'convivencia', 'igualdad de género', 'derechos humanos', 'inclusión social', 'colectivo'
    ],
    'SALUD': [
        'salud', 'pandemia', 'coronavirus', 'situacion sanitaria', 'efectos de la pandemia', 'crisis sanitaria',
        'lucha global contra pandemia', 'salud publica', 'enfermedades', 'farmacos', 'medicamentos', 'iess',
        'vacunacion', 'hospital', 'clinica', 'tratamiento', 'paciente', 'epidemia', 'doctor', 'enfermera',
        'enfermedad', 'contagio', 'sintomas', 'medicina', 'cirugia', 'consulta', 'cancer', 'prevencion', 'sanidad',
        'urgencias', 'sistema de salud', 'nutricion', 'terapia', 'psicologia', 'cardiologia', 'bienestar',
        'higiene', 'cuidados', 'rehabilitacion', 'farmacia', 'cuidado', 'salud mental', 'hospitalización',
        'diagnóstico', 'servicios médicos', 'emergencias médicas', 'bioseguridad'
    ],
    'SEGURIDAD': [
        'seguridad', 'inseguridad', 'delincuencia', 'violencia', 'crisis penitenciaria', 'peligro en calles',
        'asesinatos', 'crimen organizado', 'bandas delictivas', 'emergencia', 'defensa', 'civil', 'policia',
        'milicia', 'asesinato', 'robos', 'secuestro', 'criminal', 'fuerzas armadas', 'guardia', 'justicia',
        'corte', 'fiscalia', 'prision', 'pena', 'carcel', 'terrorismo', 'armado', 'conflicto', 'seguro', 
        'incendio', 'accidente', 'vigilancia', 'investigacion', 'peritaje', 'patrullaje', 'sospechoso', 'delito',
        'operativo', 'escolta', 'seguridad nacional', 'autodefensa', 'vigilante', 'patrulla',
        'arma', 'confiscacion', 'persecucion', 'jurisdiccion', 'protección civil', 'crimen cibernético', 
        'tráfico de armas', 'seguridad pública', 'operativos de seguridad', 'fraude'
    ],
    'POLITICA': [
        'politica', 'coyuntura politica', 'cambio de gobierno', 'elecciones', 'resultados electorales', 
        'actividad politica', 'presidenciables', 'anos', 'cabildo', 'presidenciales', 'gobierno', 'consenso',
        'partidos', 'politicos', 'paquetazo', 'sri', 'congreso', 'intendencia', 'defensoria', 'estado',
        'ministro', 'presidente', 'diputado', 'alcalde', 'regimen', 'legislativo', 'reforma', 'oposicion',
        'parlamento', 'ministerio', 'legislacion', 'constitucion', 'democracia', 'proyecto de ley',
        'junta', 'administracion', 'consejo', 'vicepresidente', 'campaña', 'alianza', 'diputados', 'senador',
        'gobernacion', 'candidato', 'representante', 'eleccion', 'municipio', 'jurisdiccion', 'senado', 'mandato',
        'alianzas políticas', 'coalición', 'diplomacia', 'geopolítica', 'políticas públicas', 'derechos civiles',
        'sistema electoral'
    ]
}

# Function to assign identifier based on keywords
def assign_identifier(text):
    if pd.isnull(text) or text.strip() == "":
        return 'OTHER'
    
    text = text.lower()  
    match_counts = {title: 0 for title in keywords}  # Initialize match counts for each category
    
    for title, words in keywords.items():
        match_counts[title] = sum(word in text for word in words)
    
    max_category = max(match_counts, key=match_counts.get)
    
    return max_category if match_counts[max_category] > 0 else 'OTHER'

# Apply function to the 'text' column and create a new 'titulo' column
for year in range(2001, 2025):
    df_name = f'df_news_{year}'  # Generate the dataframe name
    if df_name in globals():  # Check if the dataframe exists in the global scope
        globals()[df_name]['sec'] = globals()[df_name]['texto'].apply(assign_identifier)

'''
'''
frequency_tables = {}

for year in range(2001, 2025):
    try:
        freq_table = globals()[f'df_news_{year}']['sec'].value_counts().reset_index()
        freq_table.columns = ['sec', 'count']
        
        # Calculate percentage and add it as a new column
        total_count = freq_table['count'].sum()
        freq_table['percentage'] = (freq_table['count'] / total_count) * 100
        
        # Store the frequency table in the dictionary with the year as the key
        frequency_tables[year] = freq_table
        
        # Print the frequency table for each year
        print(f"\nFrequency Table for {year}:\n", freq_table)
        
    except KeyError:
        print(f"DataFrame for year {year} not found in the global scope.")
'''


# Remove intro spaces, line breaks, and consolidate into a single line
for year in range(2001, 2025):
    df_name = f'df_news_{year}'
    if df_name in globals():
        df = globals()[df_name]
        
        df['texto'] = df['texto'].str.replace(r'\s+', ' ', regex=True).str.strip()
        
        globals()[df_name] = df
        
        print(f"Cleaned extra spaces in 'texto' column for {df_name}.")
    else:
        print(f"{df_name} does not exist in the current environment.")



#Remove unecessary rows
for year in range(2002, 2003):
    df_name = f'df_news_{year}'
    if df_name in globals():
        df = globals()[df_name]
        
        initial_rows = len(df)
        
        df = df[df['texto'].notna() & df['texto'].str.strip().astype(bool)]
        
        rows_removed = initial_rows - len(df)
        
        globals()[df_name] = df
        
        # Print the number of rows removed
        print(f"Removed {rows_removed} empty 'texto' rows from {df_name}.")
    else:
        print(f"{df_name} does not exist in the current environment.")



###########################################################################################################################################################
##### FUNCTIONS #####
###########################################################################################################################################################


##### Cleaning using SPACY #####

nlp = spacy.load("es_core_news_lg")

def clean_spanish_phrases(text, progress_counter, total_count, df_name):
    # Tokenize text into words
    if not isinstance(text, str):
        return None
    
    try:
        words = text.split()
        cleaned_words = []
        
        for word in words:
            if len(word) > 1 and re.search(r'[aeiouáéíóú]', word, re.IGNORECASE):
                if not re.match(r'^[a-zA-ZáéíóúÁÉÍÓÚ]{1,2}$', word) and len(set(word)) > 2:
                    cleaned_words.append(word)

        cleaned_text = " ".join(cleaned_words)
        doc = nlp(cleaned_text)
        
        final_text = " ".join([token.text for token in doc if token.is_alpha and not token.is_stop])
        
        progress_counter[0] += 1
        completion_percentage = (progress_counter[0] / total_count) * 100
        print(f"Processing {df_name}: {completion_percentage:.2f}% completed.", end='\r')
        
        return final_text




##### SENTIMENT analyzer #####

target_sections = {'ECONOMIA', 'SOCIEDAD', 'SALUD', 'SEGURIDAD', 'POLITICA'}

def remove_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series[(series >= lower_bound) & (series <= upper_bound)]



'''

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

# Set the timeout signal handler
signal.signal(signal.SIGALRM, timeout_handler)

def compute_combined_sentiment(text, progress_counter, total_count, df_name, max_time=2): 
    if len(text) > 5000:
        # Split text into sentences if character count exceeds 5000
        sentences = [sentence.strip() for sentence in text.split('.') if sentence.strip()]
    else:
        sentences = [text]

    textblob_scores = []

    for sentence in sentences:
        try:
            # Start a timeout for processing this sentence
            signal.alarm(max_time)
            
            # Calculate TextBlob sentiment score
            textblob_score = TextBlob(sentence).sentiment.polarity * TextBlob(sentence).sentiment.subjectivity
            textblob_scores.append(textblob_score)
            
            # Disable the alarm after successful processing
            signal.alarm(0)

        except (TimeoutException, Exception):
            # Skip this observation if it times out or encounters an error
            print(f"\nSkipping sentence in {df_name} due to processing timeout or error.")
            signal.alarm(0)  # Disable alarm

    # Calculate average score if any scores were successfully calculated
    if textblob_scores:
        avg_textblob_score = sum(textblob_scores) / len(textblob_scores)
    else:
        avg_textblob_score = None  # Indicate skipped processing if no valid scores

    # Update and print progress
    progress_counter[0] += 1
    completion_percentage = (progress_counter[0] / total_count) * 100
    print(f"Processing {df_name}: {completion_percentage:.2f}% completed.", end='\r')

    return avg_textblob_score
'''



###########################################################################################################################################################
##### SENTIMENT ANALYSIS #####
###########################################################################################################################################################

start_year = 2002
end_year = 2024

target_sections = {'ECONOMIA', 'SOCIEDAD', 'SALUD', 'SEGURIDAD', 'POLITICA'}
num_per_section = 2000

for year in range(start_year, end_year + 1):
    df_name = f'df_news_{year}'
    if df_name in globals():  # Check if dataframe exists in globals
        df = globals()[df_name]  # Retrieve the dataframe
        
        
            df = df[df['Section'].isin(target_sections)]
        
        # Dictionary to store the sampled data for each section
        sampled_dataframes = {}
        total_samples = 0

        for section in target_sections:
            section_df = df[df['Section'] == section]
            
            if len(section_df) < num_per_section:
                sampled_dataframes[section] = section_df
                total_samples += len(section_df)
            else:
                sampled_dataframes[section] = section_df.sample(n=num_per_section, random_state=1)
                total_samples += num_per_section

        shortfall = num_per_section * len(target_sections) - total_samples
        remaining_sections = [section for section in target_sections if len(sampled_dataframes[section]) == num_per_section]
        
        if shortfall > 0 and remaining_sections:
            additional_per_section = shortfall // len(remaining_sections)
            extra_samples = []

            for section in remaining_sections:
                extra_section_df = df[df['Section'] == section].drop(sampled_dataframes[section].index)
                extra_samples.append(extra_section_df.sample(n=additional_per_section, random_state=1, replace=True))
            
            
            df = pd.concat(list(sampled_dataframes.values()) + extra_samples, ignore_index=True)
        else:
            df = pd.concat(sampled_dataframes.values(), ignore_index=True)


        globals()[df_name] = df
        print(f"Data for {df_name} updated with balanced sampling and {len(df)} observations.")
    else:
        print(f"{df_name} does not exist in the current environment.")


# Loop 1: Cleaning text exceprts
for year in range(start_year, end_year + 1):
    df_name = f'df_news_{year}'
    
    if df_name in globals():
        df = globals()[df_name]
        
        progress_counter = [0]
        total_count = len(df)
        
        cleaning_function = partial(clean_spanish_phrases, 
                                    progress_counter=progress_counter, 
                                    total_count=total_count, 
                                    df_name=df_name)
        
        df['texto'] = df['texto'].apply(cleaning_function)
        
        globals()[df_name] = df
        
        print(f"\nText cleaned for {df_name}. {total_count} rows processed.")
    else:
        print(f"{df_name} does not exist in the current environment.")
        
        

# Loop 2: Calculate Sentiment Scores 

for year in range(start_year, end_year + 1):
    df_name = f'df_news_{year}'
    if df_name in globals():
        df = globals()[df_name] 

        df = compute_sentiment_with_progress(df, 'texto')  

        globals()[df_name] = df  # Update the global variable with the modified dataframe
        print(f"Sentiment scores calculated for the entire dataset in {df_name}.")
    else:
        print(f"{df_name} does not exist in the current environment.")
        

# Sentiment datatframe concat

dataframes_list = []

for year in range(2001, 2025):
    df_name = f'df_news_{year}'

    if df_name in globals():
        df = globals()[df_name]

        selected_columns = df[['id', 'fecha', 'Section', 'Senti_Score']]

        dataframes_list.append(selected_columns)
    else:
        print(f"{df_name} does not exist in the current environment.")

sentiment = pd.concat(dataframes_list, ignore_index=True)
print("Data extracted and combined into 'sentiment' dataset.")
Data extracted and combined into 'sentiment' dataset.

#sentiment.to_csv("/Users/beto/Desktop/Research_paper/DATA/sentiment_raw.csv")


# Loop 3: Outliers removal Values and '0's imputation
for section in target_sections:
    section_mask = (sentiment['Section'] == section)
    zero_mask = (sentiment['Senti_Score'] == 0)

    sentiment.loc[section_mask, 'Senti_Score'] = remove_outliers(sentiment.loc[section_mask, 'Senti_Score'])

    sentiment.loc[section_mask & zero_mask, 'Senti_Score'] = (
        sentiment.loc[section_mask, 'Senti_Score']
        .replace(0, pd.NA)  # Temporarily replace 0 with NaN for rolling mean calculation
        .fillna(sentiment.loc[section_mask, 'Senti_Score'].rolling(window=12, min_periods=1).mean())
    )

print("Outliers removed and zero values imputed for 'sentiment' dataset.")
        



# Normality check
normality_results = {}

for section in target_sections:
    section_data = sentiment[sentiment['Section'] == section]['Senti_Score'].dropna()

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(section_data, kde=True)
    plt.title(f"Histogram of Senti_Score for {section}")
    plt.xlabel('Senti_Score')
    
    plt.subplot(1, 2, 2)
    stats.probplot(section_data, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot for {section}")
    
    plt.tight_layout()
    plt.show()

    stat, p_value = stats.shapiro(section_data)
    normality_results[section] = {
        'Shapiro-Wilk Statistic': stat,
        'p-value': p_value,
        'Normal Distribution': p_value > 0.05  # True if p-value > 0.05 (normality assumed)
    }

# Display normality test results
for section, result in normality_results.items():
    print(f"\nSection: {section}")
    print(f"Shapiro-Wilk Statistic: {result['Shapiro-Wilk Statistic']:.4f}")
    print(f"p-value: {result['p-value']:.4f}")
    if result['Normal Distribution']:
        print("Conclusion: Data follows a normal distribution.")
    else:
        print("Conclusion: Data does not follow a normal distribution.")



#Treating
sentiment = sentiment.reset_index() if 'fecha' not in sentiment.columns else sentiment

sentiment['fecha'] = pd.to_datetime(sentiment['fecha'], format='%Y-%m-%d')

sentiment['Fecha'] = sentiment['fecha'].dt.to_period('M')

sentiment_monthly = sentiment.groupby(['Fecha', 'Section'])['Senti_Score'].mean().reset_index()

# Step 4: Ensure each month-section combination is present, filling missing with NA
all_months = pd.date_range(sentiment['fecha'].min(), sentiment['fecha'].max(), freq='M').to_period('M')
all_sections = sentiment['Section'].unique()
all_combinations = pd.MultiIndex.from_product([all_months, all_sections], names=['Fecha', 'Section'])

sentiment_monthly = sentiment_monthly.set_index(['Fecha', 'Section']).reindex(all_combinations).reset_index()

# Step 5: Convert 'Fecha' to datetime format for compatibility with matplotlib
sentiment_monthly['Fecha'] = sentiment_monthly['Fecha'].dt.to_timestamp()

# Step 6: Check for seasonal patterns, de-seasonalize if necessary
for section in all_sections:
    section_data = sentiment_monthly[sentiment_monthly['Section'] == section].copy()
    section_data.set_index('Fecha', inplace=True)

    section_data['Senti_Score'] = section_data['Senti_Score'].interpolate()

    decomposition = seasonal_decompose(section_data['Senti_Score'], model='additive', period=12, extrapolate_trend='freq')
    section_data['Senti_Score_deseasonalized'] = section_data['Senti_Score'] - decomposition.seasonal
    
    decomposition.plot()
    plt.suptitle(f'Seasonal Decomposition for {section}', y=1.02)
    plt.show()
    
    sentiment_monthly.loc[sentiment_monthly['Section'] == section, 'Senti_Score_deseasonalized'] = section_data['Senti_Score_deseasonalized'].values

# Step 7: Check for stationarity and apply differencing if needed
def adf_test(series):
    result = adfuller(series.dropna())
    return result[1]  # Return p-value

for section in all_sections:
    series = sentiment_monthly[sentiment_monthly['Section'] == section]['Senti_Score_deseasonalized']
    if adf_test(series) > 0.05:  # p-value > 0.05 implies non-stationarity
        sentiment_monthly.loc[sentiment_monthly['Section'] == section, 'Senti_Score_stationary'] = series.diff()
    else:
        sentiment_monthly.loc[sentiment_monthly['Section'] == section, 'Senti_Score_stationary'] = series

# Step 8: Apply Z-score normalization to the stationary series
sentiment_monthly['Senti_Score_normalized'] = sentiment_monthly.groupby('Section')['Senti_Score_stationary'].transform(lambda x: zscore(x, nan_policy='omit'))

# Step 9: Clean up the DataFrame to keep only the necessary columns
sentiment_monthly = sentiment_monthly[['Fecha', 'Section', 'Senti_Score_normalized']]

#print(sentiment_monthly.head())





# Treating NA values
na_count = sentiment_monthly['Senti_Score_normalized'].isna().sum()
print(f"Number of NA values in 'Senti_Score_normalized': {na_count}")

if na_count > 0:
    sentiment_monthly['Senti_Score_normalized'] = (
        sentiment_monthly.groupby('Section')['Senti_Score_normalized']
        .apply(lambda x: x.interpolate(method='linear'))
        .reset_index(level=0, drop=True)
    )

    remaining_na_count = sentiment_monthly['Senti_Score_normalized'].isna().sum()
    print(f"Number of remaining NA values after interpolation: {remaining_na_count}")
    
    if remaining_na_count > 0:
        sentiment_monthly['Senti_Score_normalized'] = (
            sentiment_monthly.groupby('Section')['Senti_Score_normalized']
            .apply(lambda x: x.ffill().bfill())
            .reset_index(level=0, drop=True)
        )

#print(sentiment_monthly.head())




sentiment_monthly = pd.read_csv('/Users/beto/Desktop/Research_paper/DATA/sentiment_st.csv')
sentiment_monthly['Fecha'] = pd.to_datetime(sentiment_monthly['Fecha'])
sentiment_monthly.set_index('Fecha', inplace=True)

sections = {
    'SALUD': 'HEALTH',
    'ECONOMIA': 'ECONOMY',
    'SOCIEDAD': 'SOCIETY',
    'SEGURIDAD': 'SECURITY',
    'POLITICA': 'POLITICS'
}

# Loop through the dictionary and replace values in 'Section'
for old_name, new_name in sections.items():
    sentiment_monthly['Section'] = sentiment_monthly['Section'].replace(old_name, new_name)
  




# Plotting
palette = ["#4a235a", "#e74c3c", "#58d68d", "#f1c40f", "#7F00FF"]

# Set up the plotting style and figure size
sns.set(style="whitegrid")
plt.figure(figsize=(25, 12))

sns.lineplot(
    data=sentiment_monthly,
    x='Fecha', y='Senti_Score', 
    hue='Section', 
    palette=palette,
    marker='o'
)

# Customize the legend to appear in the lower left corner
plt.legend(title="Section", loc="lower left", fontsize=15, title_fontsize=15)
plt.xlabel("", fontsize=18)
plt.ylabel("Sentiment Score", fontsize=18)
plt.title("Sentiment Analysis Over Time by Section", fontsize=35)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()




# Define color palette, label font size, and sections
palette = ["#9F2B68", "#BF40BF", "#800020", "#CF9FFF", "#7F00FF"]
label_fontsize = 16
sections = ['HEALTH', 'ECONOMY', 'SOCIETY', 'SECURITY', 'POLITICS']

sns.set(style="whitegrid")
num_rows, num_cols = 3, 2
fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 15), sharex=True, sharey=True)
axes = axes.flatten()

# Plot each section
for i, section in enumerate(sections):
    ax = axes[i]
    sns.lineplot(
        data=sentiment_monthly[sentiment_monthly['Section'] == section],
        x='Fecha', y='Senti_Score', marker='o', ax=ax, color=palette[i]
    )
    ax.set_title(f"{section}", fontsize=25)
    ax.set_xlabel("", fontsize=label_fontsize)
    ax.set_ylabel("Sentiment Score", fontsize=label_fontsize)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_xticklabels([])  # Remove x-axis labels

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.suptitle("Time Series for Sentiment Analysis by Section", fontsize=35)

plt.show()







# Variation 1
palette = ["#9F2B68", "#BF40BF", "#800020", "#CF9FFF", "#7F00FF"]
label_fontsize = 20

plt.figure(figsize=(18, 15))  
sns.set(style="whitegrid")
num_rows = 3
num_cols = 2
fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 15), sharex=True, sharey=True)
axes = axes.flatten() 

highlight_periods = [
    ("2015-01-01", "2016-12-01"),
    ("2016-04-01", "2016-12-01"),
    ("2019-09-01", "2019-10-01"),
    ("2020-03-01", "2020-12-01"),
    ("2021-01-01", "2021-12-01"),
]

for i, section in enumerate(sections):
    ax = axes[i]
    sns.lineplot(
        data=sentiment_monthly[sentiment_monthly['Section'] == section],
        x='Fecha', y='Senti_Score', marker='o', ax=ax, color=palette[i]
    )
    ax.set_title(f" {section}", fontsize=25)
    ax.set_xlabel("", fontsize=label_fontsize)
    ax.set_ylabel("Sentiment Score", fontsize=label_fontsize)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    
    for start_date, end_date in highlight_periods:
        ax.axvspan(start_date, end_date, color="grey", alpha=0.3)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
    plt.suptitle("Time Series for Sentiment Analysis by Section", fontsize=35)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.show()


#Variation 2
palette = ["#9F2B68", "#BF40BF", "#800020", "#CF9FFF", "#7F00FF"]
highlight_periods = [
    ("2015-01-01", "2016-12-01"),
    ("2016-04-01", "2016-12-01"),
    ("2019-09-01", "2019-10-01"),
    ("2020-03-01", "2020-12-01"),
    ("2021-01-01", "2021-12-01"),
]
label_fontsize = 30

sns.set(style="whitegrid")

for i, section in enumerate(sections):
    plt.figure(figsize=(30, 12))  # Tamaño de cada gráfico individual
    sns.lineplot(
        data=sentiment_monthly[sentiment_monthly['Section'] == section],
        x='Fecha', y='Senti_Score', marker='o', color=palette[i]
    )
    
    plt.title(f"{section}", fontsize=45)
    plt.xlabel(" ", fontsize=label_fontsize)
    plt.ylabel("Sentiment Score", fontsize=label_fontsize)
    plt.xticks(rotation=45, fontsize=18)
    plt.yticks(fontsize=18)
    
    for start_date, end_date in highlight_periods:
        plt.axvspan(start_date, end_date, color="grey", alpha=0.3)  # Franja gris con transparencia
    
    plt.tight_layout()
    plt.show()



#Correlation heatmaps 
plt.figure(figsize=(10, 8))
all_years_corr = sentiment_monthly.pivot_table(index=sentiment_monthly.index, columns='Section', values='Senti_Score').corr()

mask = np.triu(np.ones_like(all_years_corr, dtype=bool))

sns.heatmap(all_years_corr, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation of Different Sections", fontsize=18)
plt.xlabel('')  # Remove x-axis label
plt.ylabel('')  # Remove y-axis label
plt.show()

highlight_periods = {
    "Recession of 2015-2016": ("2015-01-01", "2016-12-01"),
    "Earthquake 2016": ("2016-04-01", "2016-12-01"),
    "Fuel Subsidies lift-up 2019": ("2019-10-01", "2019-11-01"),
    "COVID-19 Pandemic": ("2020-03-01", "2020-12-01"),
    "Post-Pandemic Recession": ("2021-01-01", "2021-12-01"),
}

# Loop through each highlight period to generate a heatmap for each
for period_name, (start_date, end_date) in highlight_periods.items():
    period_data = sentiment_monthly.loc[start_date:end_date]
    
    period_corr = period_data.pivot_table(index=period_data.index, columns='Section', values='Senti_Score').corr()
    
    mask = np.triu(np.ones_like(period_corr, dtype=bool))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(period_corr, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f"{period_name}", fontsize=18)
    plt.xlabel('')  
    plt.ylabel('')  
    plt.show()








#KDE
plt.figure(figsize=(18, 12))

sections = ['HEALTH', 'ECONOMY', 'SOCIETY', 'SECURITY', 'POLITICS']
palette = ["#9F2B68", "#BF40BF", "#800020", "#CF9FFF", "#7F00FF"]

# Set up the subplot grid
num_rows = len(sections) // 2 + len(sections) % 2  # Two columns
fig, axes = plt.subplots(num_rows, 2, figsize=(18, 12), sharex=True, sharey=True)
axes = axes.flatten()  # Flatten for easy iteration

# Plot KDE for each section
for i, section in enumerate(sections):
    ax = axes[i]
    
    section_data = sentiment_monthly[sentiment_monthly['Section'] == section]
    
    sns.kdeplot(
        data=section_data,
        x='Senti_Score',
        ax=ax,
        color=palette[i % len(palette)],
        fill=True
    )
    
    mean_value = section_data['Senti_Score'].mean()
    ax.axvline(mean_value, color=palette[i % len(palette)], linestyle='--', linewidth=1.5)
    
    ax.set_title(f"{section}", fontsize=20)
    ax.set_xlabel("", fontsize=14)  
    ax.set_ylabel("", fontsize=16)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.suptitle("KDE for Sentiment Score by Section", fontsize=30)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.show()
    
    
    
    
    

recessionary_periods = {
    "Recession of 2015-2016": ("2015-01-01", "2016-12-01"),
    "Earthquake 2016": ("2016-04-01", "2016-12-01"),
    "Fuel Subsidies lift-up 2019": ("2019-10-01", "2019-11-01"),
    "COVID-19 Pandemic": ("2020-03-01", "2020-12-01"),
    "Post-Pandemic Recession": ("2021-01-01", "2021-12-01"),
}

for period_name, (start_date, end_date) in recessionary_periods.items():
    num_rows = len(sections) // 2 + len(sections) % 2  # Two columns
    fig, axes = plt.subplots(num_rows, 2, figsize=(18, 12), sharex=True, sharey=True)
    axes = axes.flatten()  # Flatten for easy iteration

    for i, section in enumerate(sections):
        ax = axes[i]

        period_data = sentiment_monthly[(sentiment_monthly['Section'] == section) & 
                                        (sentiment_monthly.index >= start_date) & 
                                        (sentiment_monthly.index <= end_date)]
        
        if not period_data.empty:
            sns.kdeplot(
                data=period_data,
                x='Senti_Score',
                ax=ax,
                color=palette[i % len(palette)],
                fill=True
            )
            
            mean_value = period_data['Senti_Score'].mean()
            ax.axvline(mean_value, color=palette[i % len(palette)], linestyle='--', linewidth=1.5)
        
        # Set title and labels
        ax.set_title(f"{section}", fontsize=20)
        ax.set_xlabel("")  # Remove x-axis label
        ax.set_ylabel("", fontsize=16)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.suptitle(f"{period_name}", fontsize=30)

    # Remove any unused subplots if there are fewer sections than grid spaces
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.show()



recession_periods = [
    ("2015-01-01", "2016-12-01"),
    ("2016-04-01", "2016-12-01"),
    ("2019-09-01", "2019-10-01"),
    ("2020-03-01", "2020-12-01"),
    ("2021-01-01", "2021-12-01"),
]

# Label each period as 'Expansion' or 'Recession'
sentiment_monthly['Period'] = 'Expansion'
for start_date, end_date in recession_periods:
    mask = (sentiment_monthly.index >= start_date) & (sentiment_monthly.index <= end_date)
    sentiment_monthly.loc[mask, 'Period'] = 'Recession'

# Set up the palette and style
sns.set(style="whitegrid")
palette = {"Expansion": "red", "Recession": "blue"}

sections = sentiment_monthly['Section'].unique()
num_sections = len(sections)

fig, axes = plt.subplots(3, 2, figsize=(18, 12), sharex=True, sharey=True)
axes = axes.flatten()  

# Plot for each section
for i, section in enumerate(sections):
    ax = axes[i]
    section_data = sentiment_monthly[sentiment_monthly['Section'] == section]
    
    # Plot KDE for expansion
    sns.kdeplot(
        data=section_data[section_data['Period'] == 'Expansion'],
        x='Senti_Score',
        ax=ax,
        color=palette['Expansion'],
        linestyle="-",
        label="Expansion" if i == 0 else ""  # Label only on the first plot for legend
    )
    
    # Plot KDE for recession
    sns.kdeplot(
        data=section_data[section_data['Period'] == 'Recession'],
        x='Senti_Score',
        ax=ax,
        color=palette['Recession'],
        linestyle="--",
        label="Recession" if i == 0 else ""  # Label only on the first plot for legend
    )

    # Calculate the median of Senti_Score for the middle line
    median_value = section_data['Senti_Score'].median()
    
    ax.axvline(median_value, color="red", linestyle="--", linewidth=1.5)

    ax.set_title(f"{section}", fontsize=16)
    ax.set_xlabel(" ", fontsize=16)
    ax.set_ylabel("Density", fontsize=16)

# Add the legend to the last empty slot if there are fewer than 6 sections
if num_sections < len(axes):
    legend_ax = axes[num_sections]  
    legend_ax.axis("off")  

    handles, labels = axes[0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="center", fontsize=20, title="Period", title_fontsize=22)

for j in range(num_sections + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("KDE for Sentiment Score by Section over Expansionary and Recessionary periods", fontsize=30)
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.show()
