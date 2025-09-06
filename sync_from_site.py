# sync_from_site.py
import csv, re, pathlib, requests
from bs4 import BeautifulSoup

SITE_URL = "https://meff.netlify.app/"
CSV_PATH = pathlib.Path("data/knowledge_base.csv")

def extract_faq_pairs(text: str):
    lines = [l.strip() for l in text.splitlines()]
    pairs = []
    q, buf = None, []
    for ln in lines:
        if not ln:
            continue
        if ln.endswith("?"):
            if q and buf:
                ans = " ".join(buf).strip()
                if len(ans) > 2:
                    pairs.append((q, ans))
            q, buf = ln, []
        else:
            if q:
                buf.append(ln)
    if q and buf:
        pairs.append((q, " ".join(buf).strip()))
    return pairs

def extract_services_pairs(soup: BeautifulSoup):
    pairs = []
    services_header = soup.find(lambda tag: tag.name in ["h2","h3"] and "Услуги" in tag.get_text())
    if not services_header:
        return pairs
    root = services_header.find_parent() or soup
    for h in root.find_all(["h3","h4","h5"]):
        title = h.get_text(" ", strip=True)
        if not title:
            continue
        desc_el = h.find_next_sibling()
        desc = ""
        steps = 0
        while desc_el and steps < 3 and desc_el.name not in ["h2","h3","h4","h5"]:
            t = desc_el.get_text(" ", strip=True)
            if t:
                desc += (" " if desc else "") + t
            desc_el = desc_el.find_next_sibling()
            steps += 1
        if title and desc and len(desc) > 3:
            q = f"Что входит: {title}?"
            a = desc
            pairs.append((q, a))
    return pairs

def main():
    print(f"Fetching {SITE_URL} ...")
    html = requests.get(SITE_URL, timeout=20).text
    soup = BeautifulSoup(html, "html.parser")
    full_text = soup.get_text("\n", strip=True)
    faq_match = re.search(r"(FAQ и Контакты[\s\S]+?)(Обо мне|©|$)", full_text)
    faq_text = faq_match.group(1) if faq_match else ""
    faq_pairs = extract_faq_pairs(faq_text)
    service_pairs = extract_services_pairs(soup)
    rows = [{"question": q, "answer": a, "url": SITE_URL} for q,a in faq_pairs + service_pairs]
    if not rows:
        raise SystemExit("Не удалось извлечь данные. Структура страницы могла измениться.")
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=["question","answer","url"])
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {len(rows)} Q/A → {CSV_PATH}")

if __name__ == "__main__":
    main()
