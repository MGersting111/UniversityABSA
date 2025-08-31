import csv
import pathlib
import random
import time
from typing import Optional
from bs4 import BeautifulSoup
import requests




def load_html(source: str, timeout: float = 15.0) -> str:
    """Lädt HTML von einer URL oder aus einer lokalen Datei."""
    if source.startswith("http://") or source.startswith("https://"):
        if requests is None:
            raise RuntimeError("Das 'requests'-Paket ist nicht installiert.")
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(source, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    else:
        # Lokale Datei
        p = pathlib.Path(source)
        return p.read_text(encoding="utf-8", errors="ignore")


def extract_studiengang(html: str) -> Optional[str]:
    #Extrahiert den Studiengang aus einer Review-Seite
    soup = BeautifulSoup(html, "html.parser")

    # 1) Normalfall
    node = soup.select_one("header.card-header p a")
    if node and node.get_text(strip=True):
        return node.get_text(strip=True)

    # 2) Alternative Struktur
    node = soup.select_one(".style-h1 a")
    if node and node.get_text(strip=True):
        return node.get_text(strip=True)

    # 3) Breadcrumbs
    names = [n.get_text(strip=True) for n in soup.select('[itemtype*="ListItem"] [itemprop="name"]')]
    if names:
        candidates = [n for n in names if n and n.lower() != "bewertung"]
        if candidates:
            return candidates[-1]

    # 4) Titel-Fallback
    title_tag = soup.find("title")
    if title_tag and title_tag.get_text():
        t = title_tag.get_text()
        if " | " in t:
            return t.split(" | ", 1)[0].strip()

    return None


def main():
    input_file = "01_review_links/review_links_tu_darmstadt.txt"
    output_file = "studiengaenge_tu_darmstadt.csv"

    min_sleep = 0.5
    max_sleep = 1

    in_path = pathlib.Path(input_file)
    if not in_path.exists():
        raise FileNotFoundError(f"Eingabedatei nicht gefunden: {in_path}")

    rows = []
    counter = 1
    for raw in in_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        url = raw.strip()
        if not url or url.startswith("#"):
            continue

        try:
            html = load_html(url)
            studiengang = extract_studiengang(html)
            rows.append({"id": counter, "url": url, "studiengang": studiengang or ""})
            print(f"[{counter}] {url}  ->  {studiengang or '— NICHT GEFUNDEN —'}")
        except Exception as e:
            rows.append({"id": counter, "url": url, "studiengang": ""})
            print(f"[{counter}] FEHLER: {url}  ->  {e}")

        counter += 1

        if url.startswith("http"):
            time.sleep(random.uniform(min_sleep, max_sleep))

    out_path = pathlib.Path(output_file)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "url", "studiengang"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nFertig. Ergebnisse gespeichert in: {out_path.resolve()}")


if __name__ == "__main__":
    main()
