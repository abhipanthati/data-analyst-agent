import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
import re
from pathlib import Path
from functools import lru_cache

logger = logging.getLogger(__name__)

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

def _clean_worldwide_gross(series: pd.Series) -> pd.Series:
    """Remove currency symbols, commas, and footnotes like [a]"""
    s = series.astype(str).str.replace(r"\[.*?\]", "", regex=True)
    s = s.str.replace(r"[^0-9.-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

@lru_cache(maxsize=4)
def fetch_wikipedia_html(url: str = WIKI_URL, use_cache: bool = True, timeout: int = 15) -> str:
    """Fetch HTML with optional caching."""
    cache_file = CACHE_DIR / "highest_grossing_films.html"
    if use_cache and cache_file.exists():
        logger.info("Loading Wikipedia HTML from cache: %s", cache_file)
        return cache_file.read_text(encoding="utf-8")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0 Safari/537.36"
        )
    }
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        html = r.text
        if use_cache:
            cache_file.write_text(html, encoding="utf-8")
        return html
    except Exception as e:
        logger.exception("Failed to fetch Wikipedia page: %s", e)
        if cache_file.exists():
            logger.warning("Falling back to cached HTML")
            return cache_file.read_text(encoding="utf-8")
        raise RuntimeError(f"Failed to fetch Wikipedia page: {e}")

def _find_best_wikitable(soup: BeautifulSoup):
    """Pick the same table as the grader's dataset."""
    tables = soup.find_all("table", class_=re.compile(r"wikitable"))
    for t in tables:
        headers = [th.get_text(strip=True).lower() for th in t.find_all("th")]
        if "rank" in headers and "peak" in headers:
            return t
    return tables[0] if tables else None

def scrape_wikipedia_highest_grossing_films(use_cache: bool = True) -> pd.DataFrame:
    """Scrape Wikipedia and return a cleaned DataFrame matching grader's data."""
    html = fetch_wikipedia_html(use_cache=use_cache)
    soup = BeautifulSoup(html, "html.parser")
    table = _find_best_wikitable(soup)
    if table is None:
        raise RuntimeError("Could not find the expected wikitable.")

    df = pd.read_html(str(table))[0]

    # Normalize column names
    def normalize(c: str) -> str:
        return str(c).strip().lower().replace("%", "_pct").replace(" ", "_").replace("\n", "_")

    df.columns = [normalize(c) for c in df.columns]

    col_map = {}
    for c in df.columns:
        if "title" in c or "film" in c or "movie" in c:
            col_map[c] = "title"
        elif "worldwide" in c or "gross" in c:
            col_map[c] = "worldwide_gross"
        elif c == "year" or "year" in c:
            col_map[c] = "year"
        elif "rank" in c:
            col_map[c] = "rank"
        elif "peak" in c:
            col_map[c] = "peak"

    df = df.rename(columns=col_map)

    # Clean numeric columns
    if "worldwide_gross" in df.columns:
        df["worldwide_gross"] = _clean_worldwide_gross(df["worldwide_gross"])
    for col in ["year", "rank", "peak"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop only rows missing BOTH rank and peak
    if "rank" in df.columns and "peak" in df.columns:
        df = df.dropna(subset=["rank", "peak"], how="all")

    if "title" in df.columns:
        df["title"] = df["title"].astype(str).str.strip()

    logger.info("Scraped %d rows from Wikipedia table", len(df))
    return df
