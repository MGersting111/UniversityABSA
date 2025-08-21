import requests
from bs4 import BeautifulSoup
import time

BASE_URL = "https://www.studycheck.de/hochschulen/thm/bewertungen"
TITLE = 'thm'
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
}

def collect_review_links_live(max_pages=10):
    review_urls = []

    for page in range(1, max_pages + 1):
        url = f"{BASE_URL}/seite-{page}"
        print(f"Lade Seite {page}: {url}")

        try:
            res = requests.get(url, headers=HEADERS, timeout=10)
            if res.status_code != 200:
                print(f"Fehler beim Abruf: Status {res.status_code}")
                continue

            soup = BeautifulSoup(res.text, "html.parser")
            found_links = 0

            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "bericht-" in href:
                    if href.startswith("/"):
                        href = "https://www.studycheck.de" + href
                    review_urls.append(href)
                    found_links += 1

            print(f"{found_links} Review-Links gefunden")
            time.sleep(0.3)

        except Exception as e:
            print(f"Fehler bei Seite {page}: {e}")
            continue

    print(f"Gesamt: {len(review_urls)} Review-Links")
    return review_urls

if __name__ == "__main__":
    links = collect_review_links_live(max_pages=500)

    with open(f"01_review_links/review_links_{TITLE}.txt", "w", encoding="utf-8") as f:
        seen = set()
        for link in links:
            if link not in seen:
                f.write(link + "\n")
                seen.add(link)

    print(f"Gespeichert in: review_links_{TITLE}.txt")
