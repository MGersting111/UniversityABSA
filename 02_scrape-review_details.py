import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

INPUT_FILE = "01_review_links/review_links_tu_darmstadt.txt"
OUTPUT_FILE = "02_reviews/tu_darmstadt_reviews.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
}

ASPECT_KEYS = [
    "Studieninhalte", "Dozenten", "Lehrveranstaltungen",
    "Ausstattung", "Organisation", "Literaturzugang", "Digitales Studieren"
]


def extract_review_data(url, rating_id):
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        if res.status_code != 200:
            print(f"❌ Fehler bei {url} – Status {res.status_code}")
            return None

        soup = BeautifulSoup(res.text, "html.parser")
        aspects = {k: None for k in ASPECT_KEYS}

        # Einzelaspekte parsen
        for li in soup.select("div.report-ratings li"):
            text = li.get_text(strip=True)
            value_tag = li.select_one("div.rating-value")
            star_tag = li.select_one("div.rating-stars")

            for asp in aspects:
                if asp in text:
                    value = None
                    if star_tag and star_tag.has_attr("data-rating"):
                        value = star_tag["data-rating"]
                    elif value_tag:
                        value = value_tag.get_text(strip=True).replace(",", ".")

                    try:
                        aspects[asp] = float(value)
                    except:
                        pass

        # Universität
        university_name_tag = soup.select_one("span.header-subtitle a")
        university_name = university_name_tag.text.strip() if university_name_tag else None

        # Gesamtwertung (score)
        score = None
        for li in soup.select("li"):
            label = li.find("strong")
            if label and "Gesamtbewertung" in label.get_text():
                score_tag = li.select_one(".rating-value span, .rating-value")
                if score_tag:
                    try:
                        score = float(score_tag.text.strip().replace(",", "."))
                    except:
                        pass
                break

        # Datum
        date = None
        for li in soup.select("li"):
            if "Veröffentlicht am:" in li.get_text():
                span = li.find("span", class_="value")
                if span:
                    date = span.get_text(strip=True)
                break

        # Text
        text_full_tag = soup.select_one("div.report-text")
        text_full = text_full_tag.get_text(strip=True) if text_full_tag else ""

        return {
            "rating_id": rating_id,
            "university": university_name,
            "rating": score,
            "date": date,
            "text_full": text_full,
            **aspects
        }

    except Exception as e:
        print(f"❌ Fehler bei {url}: {e}")
        return None


def main():
    results = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        links = [line.strip() for line in f if line.strip()]

    for idx, url in enumerate(links, start=1):  # Für Tests nur 5
        print(f"🔍 Review {idx}/{len(links)}: {url}")
        data = extract_review_data(url, rating_id=idx)
        if data:
            results.append(data)
        time.sleep(0.3)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\n✅ Fertig! {len(df)} Bewertungen gespeichert in {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
