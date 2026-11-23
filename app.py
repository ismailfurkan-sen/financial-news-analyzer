import json
import os
import re
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Financial News Sentiment Analyzer API - Thesis Edition", version="5.4")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "clef-flash")
DB_PATH = "database.db"

# 1. Genişletilmiş Kaynak Güvenilirlik Katsayıları
SOURCE_CREDIBILITY_WEIGHTS: Dict[str, float] = {
    "kap": 1.5,
    "resmi": 1.5,
    "bloomberg": 1.3,
    "reuters": 1.3,
    "foreks": 1.2,
    "global news": 1.2,
    "cnbc": 1.2,
    "midas": 1.1,
    "rota borsa": 1.1,
    "google news tr": 1.0,
    "paratic": 1.0,
    "manşetiz": 1.0,
    "uzmanpara": 1.0,
    "tipranks": 1.0,
    "tradingview": 1.1,
    "mynet": 0.9,
    "hisse.net": 0.8,
    "default": 1.0
}

def get_source_weight(source_name: str) -> float:
    s = source_name.lower()
    for key, weight in SOURCE_CREDIBILITY_WEIGHTS.items():
        if key in s:
            return weight
    return SOURCE_CREDIBILITY_WEIGHTS["default"]

# Kalibre edilmiş karar motoru kriterleri (Karma haber ve marj baskısı hassasiyetli)
SENTIMENT_CRITERIA: Dict[str, str] = {
    "cok_olumlu": "Güncel ve taze: Dev ihracat sözleşmesi, milyarlık anlaşma, yüksek kâr açıklaması, temettü, büyük devlet teşviki",
    "olumlu": "Güncel: Beklentilerden iyi bilanço, maliyet baskısı olmadan net büyüme, yeni proje açılışı, analist hedef fiyat yükseltmesi",
    "notr": "Eski, rutin veya karma: Sıradan günlük bültenler, yönü belirsiz teknik analizler, hem büyüme hem de maliyet/marj baskısı içeren iki yönlü karma haberler",
    "olumsuz": "Güncel: Zayıf bilanço, beklentinin altında kâr, gelecek dönem marj daralması uyarısı, küçük ceza, operasyonel aksaklıklar",
    "cok_olumsuz": "Güncel ve kritik: Şirket iflası, ağır hukuki davalar, SEC/SPK soruşturması, büyük kriz veya felaketler",
}

LABEL_VALUES: Dict[str, float] = {
    "cok_olumlu": 1.0,
    "olumlu": 0.5,
    "notr": 0.0,
    "olumsuz": -0.5,
    "cok_olumsuz": -1.0,
}

LABEL_TR = {
    "cok_olumlu": "çok olumlu",
    "olumlu": "olumlu",
    "notr": "nötr",
    "olumsuz": "olumsuz",
    "cok_olumsuz": "çok olumsuz",
}


# ---------------------------------------------------------------- Veritabanı
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            total_news INTEGER,
            average_score REAL,
            market_sentiment TEXT,
            summary TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


init_db()


# ------------------------------------------------------------ Pydantic Modelleri
class NewsItemAnalysis(BaseModel):
    title: str
    date: str = Field(..., description="Haberin yayın tarihi.")
    score: float = Field(..., description="-1.0 ile +1.0 arası skor.")
    reason: str


class FinancialAnalysisResponse(BaseModel):
    status: str
    total_news_analyzed: int
    average_sentiment_score: float
    market_sentiment: str = Field(..., description="Güçlü Pozitif, Pozitif, Nötr, Negatif, Güçlü Negatif")
    analyzed_items: List[NewsItemAnalysis]
    overall_summary: str
class FinancialAnalysisResponse(BaseModel):
    status: str
    total_news_analyzed: int
    average_sentiment_score: float
    market_sentiment: str = Field(..., description="Güçlü Pozitif, Pozitif, Nötr, Negatif, Güçlü Negatif")
    risk_alert: Optional[str] = Field(None, description="Otomatik risk eşik yönetimi uyarı mesajı")
    analyzed_items: List[NewsItemAnalysis]
    overall_summary: str


def check_risk_thresholds(avg_score: float, negative_count: int) -> Optional[str]:
    """Otomatik Alarm ve Risk Eşik Yönetimi (Alert Mechanism)"""
    if avg_score <= -0.3 or negative_count >= 4:
        return "KIRMIZI ALARM: Ciddi Negatif Duygu Yığılması ve Risk Eşiği Aşıldı!"
    elif avg_score <= -0.15:
        return "SARI ALARM: Negatif Momentum Baskısı Gözleniyor."
    elif avg_score >= 0.4:
        return "YEŞİL ALARM: Güçlü Boğa (Bullish) Sinyali ve Yüksek Pozitif Momentum."
    return None


# ------------------------------------------------------------- Evrensel Akıllı Varlık (Entity) Filtresi
def is_relevant_news(title: str, snippet: str, query: str) -> bool:
    q_lower = query.lower().strip()
    text = f"{title} {snippet}".lower()
    
    general_terms = ["borsa", "ekonomi", "crypto", "kripto", "piyasa", "stock", "market", "altın", "dolar", "faiz"]
    if any(term in q_lower for term in general_terms):
        return True
        
    query_tokens = [w for w in q_lower.split() if len(w) > 1]
    if not query_tokens:
        return True
        
    match_found = any(token in text for token in query_tokens)

    return match_found


# ------------------------------------------------------------- Son 90 Günün Haberi Filtresi (Genişletilmiş Pencere)
def is_recent_news(date_str: str) -> bool:
    if not date_str or "Tarih Belirtilmemiş" in date_str:
        return True
    
    d_lower = date_str.lower()
    if any(w in d_lower for w in ["ago", "saat", "gün", "dakika", "hour", "day", "minute", "moments", "hafta", "week"]):
        return True

    try:
        cleaned_date = date_str.split(",")[0].strip()
        parsed_date = None
        for fmt in ("%m/%d/%Y", "%d/%m/%Y", "%b %d, %Y", "%Y-%m-%d"):
            try:
                parsed_date = datetime.strptime(cleaned_date, fmt)
                break
            except ValueError:
                continue
                
        if parsed_date:
            threshold_date = datetime.now() - timedelta(days=90)
            if parsed_date < threshold_date:
                return False
        else:
            years = re.findall(r'\b(20\d\d)\b', date_str)
            if years:
                latest_year = int(max(years))
                if latest_year < datetime.now().year:
                    return False
    except Exception:
        pass
        
    return True


# ------------------------------------------------------------- Haber Çekme ve Filtreleme
# ------------------------------------------------------------- Genişletilmiş ve Çoklu Sorgu (Multi-Query) Haber Çekme
def fetch_multiple_financial_news(api_key: str, query: str) -> List[Tuple[str, str, str, str]]:
    url = "https://serpapi.com/search"
    items: List[Tuple[str, str, str, str]] = []
    seen = set()

    q = query.lower()
    is_crypto = any(c in q for c in ["btc", "bitcoin", "eth", "ethereum", "crypto", "kripto", "sol", "ripple"])

    # Hocanın istediği 15-20 haberi yakalayabilmek için arama varyasyonlarını çoğaltıyoruz
    if is_crypto:
        search_queries = [
            f"{query} crypto news",
            f"{query} blockchain update",
            f"{query} fiyat analizi duyuru"
        ]
    else:
        search_queries = [
            f"{query} hisse haber",
            f"{query} bilanço KAP açıklaması",
            f"{query} hedef fiyat analiz",
            f"{query} son dakika gelişmeleri"
        ]

    searches = []
    for sq in search_queries:
        # Türkçe aramalar
        searches.append(({"engine": "google_news", "q": sq, "api_key": api_key, "hl": "tr", "gl": "tr", "tbs": "sbd:1"}, "Google News TR"))
        # Küresel / İngilizce destekli aramalar (Yabancı kaynakları da katmak için)
        searches.append(({"engine": "google_news", "q": f"{query} stock news earnings", "api_key": api_key, "hl": "en", "gl": "us", "tbs": "sbd:1"}, "Global News"))

    for params, default_source in searches:
        # Hedeflenen 20 habere ulaştıysak döngüyü durdur
        if len(items) >= 20:
            break
        try:
            resp = requests.get(url, params=params, timeout=20)
            if resp.status_code != 200:
                continue
            for item in resp.json().get("news_results", []):
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                source = (item.get("source") or {}).get("name", default_source)
                date = item.get("date", "Tarih Belirtilmemiş")
                
                # Rutin filtreler
                if "Günlük Teknik Analiz" in title:
                    continue
                if not is_relevant_news(title, snippet, query):
                    continue
                if not is_recent_news(date):
                    continue
                    
                if title and title not in seen:
                    seen.add(title)
                    items.append((title, snippet, source, date))
                    if len(items) >= 20:
                        break
        except Exception as e:
            print(f"Haber Çekme Hatası: {e}")

    return items

# ------------------------------------------------------------ Model ve Yanıt Çözümleme
def resolve_model_name() -> str:
    r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
    r.raise_for_status()
    names = [m.get("name", "") for m in r.json().get("models", [])]
    if OLLAMA_MODEL in names:
        return OLLAMA_MODEL
    for n in names:
        if n.split(":")[0] == OLLAMA_MODEL.split(":")[0]:
            return n
    raise RuntimeError(f"'{OLLAMA_MODEL}' modeli Ollama'da bulunamadı. Yüklü modeller: {names}")


def _find_probs(obj, labels: List[str]) -> Optional[Dict[str, float]]:
    if isinstance(obj, dict):
        if all(isinstance(obj.get(l), (int, float)) for l in labels):
            return {l: float(obj[l]) for l in labels}
        for v in obj.values():
            found = _find_probs(v, labels)
            if found:
                return found
    elif isinstance(obj, list):
        found_map: Dict[str, float] = {}
        for el in obj:
            if isinstance(el, dict):
                name = next((el[k] for k in ("label", "option", "choice", "name", "answer", "key", "value")
                             if isinstance(el.get(k), str) and el[k] in labels), None)
                prob = next((el[k] for k in ("probability", "prob", "p", "score")
                             if isinstance(el.get(k), (int, float))), None)
                if name is not None and prob is not None:
                    found_map[name] = float(prob)
        if len(found_map) == len(labels):
            return found_map
        for el in obj:
            found = _find_probs(el, labels)
            if found:
                return found
    return None


def _find_label(obj, labels: List[str]) -> Optional[str]:
    if isinstance(obj, str):
        return obj if obj in labels else None
    if isinstance(obj, dict):
        for v in obj.values():
            found = _find_label(v, labels)
            if found:
                return found
    elif isinstance(obj, list):
        for el in obj:
            found = _find_label(el, labels)
            if found:
                return found
    return None


import statistics
from collections import Counter

def _single_inference(state: str, model_name: str) -> Optional[Tuple[float, str, float]]:
    """Modelden tek bir çıkarım (inference) alır."""
    payload = {
        "model": model_name,
        "state": state,
        "options": {"temperature": 0.3},  # Çeşitlilik ve tutarlılık dengesi için düşük sıcaklık
        "questions": {
            "sentiment": {
                "type": "choice",
                "instructions": "Bu güncel finansal haberin yayın tarihini, büyüme verilerini ve olası maliyet/marj baskılarını birlikte değerlendirerek piyasa görünümü üzerindeki net etkisi nedir?",
                "criteria": SENTIMENT_CRITERIA,
            }
        },
    }

    try:
        r = requests.post(f"{OLLAMA_URL}/v1/systemone", json=payload, timeout=60)
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception:
        return None

    labels = list(LABEL_VALUES.keys())
    probs = _find_probs(data, labels)

    if probs:
        total = sum(max(p, 0.0) for p in probs.values()) or 1.0
        probs = {k: max(v, 0.0) / total for k, v in probs.items()}
        score = sum(probs[k] * LABEL_VALUES[k] for k in labels)
        top = max(probs, key=probs.get)
        return (round(score, 3), top, probs[top])

    label = _find_label(data, labels)
    if label:
        return (LABEL_VALUES[label], label, 1.0)

    return None


def classify_news(title: str, snippet: str, date: str, model_name: str, runs: int = 3) -> Tuple[Optional[Tuple[float, str, float]], str]:
    """Self-Consistency (Öz-Tutarlılık): Haberi 'runs' kez değerlendirip medyan skor ve çoğunluk oylaması döner."""
    state = f"Haber Tarihi: {date}\nHaber başlığı: {title}\nÖzet: {snippet}"
    
    valid_runs: List[Tuple[float, str, float]] = []
    for _ in range(runs):
        res = _single_inference(state, model_name)
        if res is not None:
            valid_runs.append(res)
            
    if not valid_runs:
        return None, "Ollama Self-Consistency turlarında geçerli yanıt alınamadı."

    # 1. Skorların medyanını (ortanca değerini) alarak uç sapmaları eliyoruz
    scores = [r[0] for r in valid_runs]
    consistent_score = round(statistics.median(scores), 3)

    # 2. Etiketler arasında çoğunluk oylaması (Majority Voting) yapıyoruz
    labels = [r[1] for r in valid_runs]
    most_common_label = Counter(labels).most_common(1)[0][0]

    # 3. Seçilen etikete ait ortalama güven olasılığını hesaplıyoruz
    matching_probs = [r[2] for r in valid_runs if r[1] == most_common_label]
    avg_prob = round(sum(matching_probs) / len(matching_probs), 3)

    return (consistent_score, most_common_label, avg_prob), ""
def sentiment_label(avg: float) -> str:
    if avg >= 0.4:
        return "Güçlü Pozitif"
    if avg >= 0.15:
        return "Pozitif"
    if avg > -0.15:
        return "Nötr"
    if avg > -0.4:
        return "Negatif"
    return "Güçlü Negatif"


def analyze_news_batch(news_list: List[Tuple[str, str, str, str]], query: str) -> Tuple[Optional[FinancialAnalysisResponse], str]:
    try:
        model_name = resolve_model_name()
    except Exception as e:
        return None, str(e)

    results: List[NewsItemAnalysis] = []
    last_error = ""
    
    total_weighted_score = 0.0
    total_weight = 0.0

    for title, snippet, source, date in news_list:
        out, err = classify_news(title, snippet, date, model_name)
        if out is None:
            last_error = err
            continue
        score, label, prob = out
        
        weight = get_source_weight(source)
        
        total_weighted_score += (score * weight)
        total_weight += weight

        results.append(NewsItemAnalysis(
            title=title,
            date=date,
            score=score,
            reason=f"Kaynak: {source} (Tarih: {date}) [Güvenilirlik Katsayısı: {weight}x]. Etki: {LABEL_TR[label]} (%{prob * 100:.0f} olasılık).",
        ))

    if not results or total_weight == 0:
        return None, last_error or "Hiçbir güncel haber sınıflandırılamadı."

    avg = round(total_weighted_score / total_weight, 4)
    
    pos = sum(1 for i in results if i.score >= 0.15)
    neg = sum(1 for i in results if i.score <= -0.15)
    neu = len(results) - pos - neg
    best = max(results, key=lambda i: i.score)
    worst = min(results, key=lambda i: i.score)

    summary = (f"'{query}' için {len(results)} güncel haber analiz edildi (Varlık, son 90 gün tarih ve kaynak ağırlıklı): "
               f"{pos} olumlu, {neu} nötr, {neg} olumsuz. "
               f"Ağırlıklı ortalama duygu skoru {avg:+.2f}. "
               f"En olumlu haber: \"{best.title}\" ({best.score:+.2f}). "
               f"En olumsuz haber: \"{worst.title}\" ({worst.score:+.2f}).")

    return FinancialAnalysisResponse(
        status="success",
        total_news_analyzed=len(results),
        average_sentiment_score=avg,
        market_sentiment=sentiment_label(avg),
        analyzed_items=results,
        overall_summary=summary,
    ), ""


# ------------------------------------------------------------ Uç Noktalar
@app.get("/analyze", response_model=FinancialAnalysisResponse)
def analyze_endpoint(api_key: str, query: str = "Borsa İstanbul ekonomi"):
    news_list = fetch_multiple_financial_news(api_key, query)

    if not news_list:
        raise HTTPException(status_code=404, detail=f"'{query}' sorgusu için geçerli ve güncel haber bulunamadı.")

    analysis_result, error = analyze_news_batch(news_list, query)

    if analysis_result is None:
        raise HTTPException(status_code=502, detail=f"Clef-Flash analiz hatası: {error}")

    # Risk alarmını güvenli bir şekilde hesapla ve ekle
    try:
        neg_count = sum(1 for i in analysis_result.analyzed_items if i.score <= -0.15)
        risk_alert_msg = check_risk_thresholds(analysis_result.average_sentiment_score, neg_count)
        analysis_result.risk_alert = risk_alert_msg
    except Exception as e:
        print(f"Risk Alarm Hesaplama Hatası: {e}")
        analysis_result.risk_alert = None

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO analyses (query, total_news, average_score, market_sentiment, summary) VALUES (?, ?, ?, ?, ?)",
            (query, analysis_result.total_news_analyzed, analysis_result.average_sentiment_score,
             analysis_result.market_sentiment, analysis_result.overall_summary),
        )
        conn.commit()
        conn.close()
    except Exception as db_err:
        print(f"DB Kayıt Hatası: {db_err}")

    return analysis_result

@app.get("/history")
def get_history():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, query, total_news, average_score, market_sentiment, summary, timestamp "
                   "FROM analyses ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    return [{
        "id": r[0], "query": r[1], "total_news": r[2], "average_score": r[3],
        "market_sentiment": r[4], "summary": r[5], "timestamp": r[6],
    } for r in rows]


# ------------------------------------------------------------ YENİ MODÜL: Zaman Serisi ve Trend Analizi
@app.get("/trend")
def get_sentiment_trend(query: str = "THYAO"):
    """Belirli bir varlığın geçmiş analiz verilerini çekerek sentiment momentum ve trend yönünü hesaplar."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT average_score, market_sentiment, summary, timestamp 
        FROM analyses 
        WHERE query LIKE ? 
        ORDER BY timestamp ASC
    """, (f"%{query}%",))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail=f"'{query}' için veritabanında geçmiş analiz kaydı bulunamadı. Önce /analyze uç noktasını çalıştırın.")

    scores = [r[0] for r in rows]
    history_points = [{"score": r[0], "market_sentiment": r[1], "summary": r[2], "timestamp": r[3]} for r in rows]

    current_score = scores[-1]
    initial_score = scores[0]
    score_change = round(current_score - initial_score, 4)

    # Sentiment Momentum (Kısa vadeli hareketli ortalama farkı)
    if len(scores) >= 2:
        recent_avg = sum(scores[-3:]) / min(len(scores), 3)
        older_avg = sum(scores[:-3]) / max(len(scores) - 3, 1) if len(scores) > 3 else scores[0]
        momentum = round(recent_avg - older_avg, 4)
    else:
        momentum = 0.0

    # Trend Yönü Sınıflandırması
    if current_score >= 0.2:
        trend_direction = "Yükselen Trend (Bullish)"
    elif current_score <= -0.2:
        trend_direction = "Düşüş Trendi (Bearish)"
    else:
        trend_direction = "Yatay / Nötr Trend"

    return {
        "status": "success",
        "query": query,
        "total_recorded_analyses": len(rows),
        "current_sentiment_score": current_score,
        "score_change_since_first_record": score_change,
        "sentiment_momentum": momentum,
        "trend_direction": trend_direction,
        "historical_data_points": history_points
    }
    


# ------------------------------------------------------------ Arayüz Sunumu (FileResponse)
@app.get("/")
def read_root():
    return FileResponse("index.html")