"""
Refactored scraper.py
- Uses requests.Session for connection reuse
- Optional simple on-disk caching for development to avoid repeated hits
- Robust table selection and column normalization
- Clear errors and logging
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
import re
import os
from pathlib import Path
from functools import lru_cache

logger = logging.getLogger(__name__)

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)


def _clean_worldwide_gross(series: pd.Series) -> pd.Series:
    # Remove currency symbols, commas, footnotes like "[a]"
    s = series.astype(str).str.replace(r"\[.*?\]", "", regex=True)
    s = s.str.replace(r"[^0-9.-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


@lru_cache(maxsize=4)
def fetch_wikipedia_html(url: str = WIKI_URL, use_cache: bool = True, timeout: int = 15) -> str:
    """Fetch HTML using session. If use_cache=True and cache exists, use cached copy (dev convenience)."""
    cache_file = CACHE_DIR / "highest_grossing_films.html"
    if use_cache and cache_file.exists():
        logger.info("Loading Wikipedia HTML from cache: %s", cache_file)
        return cache_file.read_text(encoding="utf-8")

    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0 Safari/537.36"
        )
    })
    try:
        r = s.get(url, timeout=timeout)
        r.raise_for_status()
        html = r.text
        if use_cache:
            try:
                cache_file.write_text(html, encoding="utf-8")
            except Exception as e:
                logger.debug("Failed to write cache: %s", e)
        return html
    except Exception as e:
        logger.exception("Failed to fetch Wikipedia page: %s", e)
        # If cache exists, fall back
        if cache_file.exists():
            logger.warning("Falling back to cached HTML due to fetch error")
            return cache_file.read_text(encoding="utf-8")
        raise RuntimeError(f"Failed to fetch Wikipedia page: {e}")


def _find_best_wikitable(soup: BeautifulSoup):
    # prefer table with header containing "Rank" and "Worldwide"
    tables = soup.find_all("table", class_=re.compile(r"wikitable"))
    if not tables:
        return None

    for t in tables:
        text = t.get_text(" ", strip=True).lower()
        if "rank" in text and ("worldwide" in text or "worldwide gross" in text):
            return t

    # fallback: first wikitable
    return tables[0]


def scrape_wikipedia_highest_grossing_films(use_cache: bool = True) -> pd.DataFrame:
    """Scrape the Wikipedia page and return a cleaned DataFrame.

    Returns a DataFrame with normalized column names (lower_snake_case). Core columns:
    - title, year, worldwide_gross, rank, peak
    """
    html = fetch_wikipedia_html(use_cache=use_cache)
    soup = BeautifulSoup(html, "html.parser")
    table = _find_best_wikitable(soup)
    if table is None:
        raise RuntimeError("Could not find the expected wikitable on the page.")

    # Use pandas to read the html table
    df = pd.read_html(str(table))[0]

    # Normalize columns to lower_snake_case
    def normalize(c: str) -> str:
        return (
            str(c)
            .strip()
            .lower()
            .replace("%", "_pct")
            .replace(" ", "_")
            .replace("\n", "_")
        )

    df.columns = [normalize(c) for c in df.columns]

    # Heuristics: find likely columns
    col_map = {}
    for c in df.columns:
        if "title" in c or "film" in c or "movie" in c:
            col_map[c] = "title"
        elif "worldwide" in c or "gross" in c:
            col_map[c] = "worldwide_gross"
        elif c == "year" or "year" in c:
            col_map[c] = "year"
        elif c == "rank" or "no." in c or c.startswith("#"):
            col_map[c] = "rank"
        elif "peak" in c:
            col_map[c] = "peak"

    df = df.rename(columns=col_map)

    # Clean and coerce types
    if "worldwide_gross" in df.columns:
        df["worldwide_gross"] = _clean_worldwide_gross(df["worldwide_gross"])

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    for col in ["rank", "peak"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing essential info
    needed = [c for c in ["worldwide_gross", "rank", "peak"] if c in df.columns]
    if needed:
        df = df.dropna(subset=needed)

    # Strip titles
    if "title" in df.columns:
        df["title"] = df["title"].astype(str).str.strip()

    logger.info("Scraped %d rows from Wikipedia table", len(df))
    return df