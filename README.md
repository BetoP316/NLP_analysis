# Effectiveness of Sentiment Analysis in Predicting Ecuador’s GDP Annual Growth Rate Using Newspaper Textual Data (2001-2024)
# 📌 Project Overview
Ecuador’s economic volatility demands real-time insights. This study enhances GDP forecasting by integrating sentiment analysis from 240,000+ newspaper articles (2001–2024). Using NLP models (TextBlob, VADER, SpaCy) and LASSO-ARDL econometrics, we outperform traditional survey-based benchmarks for short-term predictions.

🔹 **Key Takeaways:**  
✅ Sentiment analysis improves **1-3 month** GDP forecasts 📉  
✅ LASSO-ARDL reduces **forecast errors (RMSE)** by ~**30%** 🧠  
✅ Real-time text analysis captures **public & market perception** better than monthly surveys  
✅ Beyond 6 months, traditional macroeconomic models remain **more reliable** 

## 📊 Methodology & Approach  

### **1️⃣ Data Collection**  
I analyzed **240,000+ news articles** from Ecuadorian newspapers between **2000-2024** to extract sentiment indicators, graciously provided by USFQ-DataHUB

📌 **Sources:**  

| Data Source | Timeframe | Data Type |
|------------|-----------|------------|
| Central Bank of Ecuador | 2001–2024 | GDP Growth Rate |
| National Statistics Institute (INEC) | 2001–2024 | Business & Consumer Sentiment Surveys |
| USFQ-DataHUB News Articles | 2001–2024 | Sentiment Analysis Indicators |

---

### **2️⃣ NLP Sentiment Analysis**  
I processed textual data using **Natural Language Processing (NLP)** tools:

🔹 **VADER** – Valence Aware Sentiment Scoring 🤖  
🔹 **TextBlob** – Lexicon-based Sentiment Estimation 📝  
🔹 **SpaCy** – Named Entity Recognition & Text Cleaning 🏷️  

📌 **Sentiment Categories Identified:**  
✅ **Economy** 🏦  
✅ **Politics** 🏛️  
✅ **Health** 🏥  
✅ **Security** 🚓  
✅ **Society** 👥  

### **3️⃣ Forecasting Model: LASSO-ARDL**  
The **Autoregressive Distributed Lag (ARDL) model** combined with **LASSO regularization** helps refine the GDP forecast by eliminating irrelevant predictors.

📌 **Why LASSO-ARDL?**  
✅ Captures both **short-term dynamics & long-term relationships**  
✅ Filters **noise & redundant predictors** using LASSO regularization  
✅ Incorporates **real-time sentiment indicators**  

📈 **Model Equation:**  
```math
GDP_t = \beta_0 + \sum_{i=1}^{n} \beta_i X_{t-i} + \sum_{j=1}^{m} \gamma_j S_{t-j} + \epsilon_t
```

* Where $$ \( S_{t-j} \) $$ represents sentiment scores at $$ lag \( j \)$$ *


📊 **Performance Metrics:**  

| Model | RMSE (1-month) | RMSE (3-month) | RMSE (6-month) |
|------------|-----------|-----------|-----------|
| **Baseline ARDL** | 1.38 | 1.09 | 0.82 |
| **LASSO-ARDL** ✅ | 0.91 📉 | 0.82 📉 | 0.64 📉 |

📌 **Key Takeaway:** **Short-term predictions (1-3 months) improve significantly, but beyond 6 months, accuracy stabilizes.**  

---

## 🎯 Key Results & Insights  

📌 **Main Findings:**  
✅ **Sentiment-enhanced models significantly outperform survey-based forecasts for short-term GDP predictions.**  
✅ **Economic downturns (COVID-19, 2015 Recession) create visible sentiment shifts affecting GDP trends.**  
✅ **LASSO-ARDL provides more stable forecasts by reducing overfitting and noise from textual data.**  
✅ **While real-time sentiment analysis is effective for short horizons, structural surveys remain crucial for long-term projections.**

✅ (+) **Granger Causality Tests confirm that economic sentiment influences GDP growth, reinforcing the predictive power of news-based sentiment indicators.**

---
