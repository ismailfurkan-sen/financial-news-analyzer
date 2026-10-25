import json
import requests
from pydantic import BaseModel, Field
from typing import Literal
from serpapi import GoogleSearch

# Hocanın istediği JSON şemasını Pydantic ile modelliyoruz
class FinancialNewsAnalysis(BaseModel):
    news_score: Literal["olumlu", "nötr", "olumsuz"] = Field(
        description="Haberin borsa ve piyasa için duygu durumu sınıflandırması"
    )
    news_status: Literal["var", "yok"] = Field(
        description="Haberin içinde sayısal, oran veya istatistiki bir veri bulunup bulunmadığı"
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
            print("Arama sonucunda haber bulunamadı.")
            return None
        
        # Gelen ilk güncel haberi başlık ve özetiyle alıyoruz
        first_news = news_results[0]
        title = first_news.get("title", "")
        snippet = first_news.get("snippet", "")
        return f"{title} - {snippet}"
        
    except Exception as e:
        print(f"SerpAPI bağlantı hatası: {e}")
        return None

def analyze_news_with_ollama(news_text: str):
    url = "http://localhost:11434/api/chat"
    model_name = "qwen2.5:7b" 

    prompt = f"""
    Sen profesyonel bir borsa ve finansal haber analistisin. Aşağıdaki canlı haberi incele.
    KESİNLİKLE sadece şu JSON formatında yanıt ver ve başka hiçbir açıklama yazma:
    {{
      "news_score": "olumlu" veya "nötr" veya "olumsuz",
      "news_status": "var" veya "yok"
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

    print(f"'{model_name}' modeli ile canlı haber analiz ediliyor...")
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

if __name__ == "__main__":
    # SerpAPI anahtarını buradaki tırnak içine yapıştır
    SERPAPI_KEY = "ded71dfa981e6683e2be636f721768df65642c94413448337be6acc25143c001"
    
    print("SerpAPI üzerinden Google News'ten canlı haber çekiliyor...")
    live_news = fetch_live_financial_news(SERPAPI_KEY)
    
    if live_news:
        print(f"\n[ÇEKİLEN CANLI HABER]: {live_news}\n" + "-"*50)
        
        analysis_result = analyze_news_with_ollama(live_news)
        
        if analysis_result:
            print("\n[BAŞARILI] Yapılandırılmış LLM Çıktısı:")
            print(analysis_result.model_dump_json(indent=2))
    else:
        print("Canlı haber çekilemedi, lütfen API anahtarını kontrol et.")