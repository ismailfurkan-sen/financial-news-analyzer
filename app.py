import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from serpapi import GoogleSearch

app = FastAPI(
    title="Financial News Sentiment Analyzer API",
    description="Yerel LLM (Qwen) ve SerpAPI destekli finansal haber duygu analiz motoru",
    version="1.1"
)

# Geliştirilmiş Pydantic Şeması
class FinancialNewsAnalysis(BaseModel):
    news_score: Literal["olumlu", "nötr", "olumsuz"] = Field(
        description="Haberin borsa ve piyasa için duygu durumu sınıflandırması"
    )
    news_status: Literal["var", "yok"] = Field(
        description="Haberin içinde sayısal, oran veya istatistiki bir veri bulunup bulunmadığı"
    )
    company_name: str = Field(
        description="Haberin odaklandığı şirket, varlık, borsa veya emtia adı (Örn: Borsa İstanbul, Bitcoin, Apple vb.)"
    )
    summary: str = Field(
        description="Haberin 1-2 cümlelik net Türkçe özeti"
    )

def fetch_live_financial_news(api_key: str, query: str = "Borsa İstanbul ekonomi"):
    params = {
        "engine": "google_news",
        "q": query,
        "api_key": api_key,
        "hl": "tr",
        "gl": "tr"
    }
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        news_results = results.get("news_results", [])
        if not news_results:
            return None
        first_news = news_results[0]
        return f"{first_news.get('title', '')} - {first_news.get('snippet', '')}"
    except Exception as e:
        print(f"SerpAPI hatası: {e}")
        return None

def analyze_news_with_ollama(news_text: str):
    url = "http://localhost:11434/api/chat"
    model_name = "qwen2.5:3b" # Kararlı ve hızlı model

    prompt = f"""
    Sen profesyonel bir borsa ve finansal haber analistisin. Aşağıdaki canlı haberi incele.
    KESİNLİKLE sadece şu JSON formatında yanıt ver ve başka hiçbir açıklama yazma:
    {{
      "news_score": "olumlu" veya "nötr" veya "olumsuz",
      "news_status": "var" veya "yok",
      "company_name": "ilgili varlık, şirket veya borsa adı",
      "summary": "haberin 1-2 cümlelik net Türkçe özeti"
    }}

    Haber: "{news_text}"
    """

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "Sen bir finansal analiz asistanısın. Sadece istenen JSON formatında yanıt veren bir servissin."},
            {"role": "user", "content": prompt}
        ],
        "format": "json",
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result_json = response.json()
        raw_content = result_json["message"]["content"]
        parsed_data = FinancialNewsAnalysis.model_validate_json(raw_content)
        return parsed_data
    except Exception as e:
        print(f"Ollama analiz hatası: {e}")
        return None

@app.get("/analyze")
def analyze_endpoint(api_key: str, query: str = "Borsa İstanbul ekonomi"):
    """
    Belirtilen arama sorgusu ile Google Haberler'den veri çeker 
    ve yerel Ollama modeli ile yapılandırılmış analiz üretir.
    """
    live_news = fetch_live_financial_news(api_key, query)
    if not live_news:
        raise HTTPException(status_code=404, detail="Haber bulunamadı veya SerpAPI anahtarı geçersiz.")
    
    analysis = analyze_news_with_ollama(live_news)
    if not analysis:
        raise HTTPException(status_code=500, detail="Yerel LLM (Ollama) analiz sırasında hata oluştu.")
    
    return {
        "status": "success",
        "news_text": live_news,
        "analysis": analysis
    }