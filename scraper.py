# scraper.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO

def scrape_wikipedia_highest_grossing_films():
    """
    Scrapes the 'List of highest-grossing films' Wikipedia table and returns a cleaned DataFrame.
    """
    url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0 Safari/537.36"
        )
    }
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch Wikipedia page: {e}")

    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table", {"class": "wikitable"})
    if table is None:
        raise RuntimeError("Could not find the expected wikitable on the page.")

    df = pd.read_html(StringIO(str(table)))[0]
    # flatten columns and normalize names
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # coerce columns we need
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
    if 'worldwide_gross' in df.columns:
        df['worldwide_gross'] = df['worldwide_gross'].replace(r'[\$,]', '', regex=True)
        df['worldwide_gross'] = pd.to_numeric(df['worldwide_gross'], errors='coerce')

    for col in ['rank', 'peak']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # drop rows missing essential fields
    needed = [c for c in ['worldwide_gross', 'rank', 'peak'] if c in df.columns]
    if needed:
        df = df.dropna(subset=needed)

    return df
