import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO

def scrape_wikipedia_highest_grossing_films():
    url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    table = soup.find("table", {"class": "wikitable"})
    df = pd.read_html(StringIO(str(table)))[0]

    df.columns = df.columns.droplevel(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    df['year'] = pd.to_numeric(df['year'], errors='coerce')

    df['worldwide_gross'] = (
    df['worldwide_gross']
    .replace(r'[\$,]', '', regex=True)
    .replace('', pd.NA)
)

    df['worldwide_gross'] = pd.to_numeric(df['worldwide_gross'], errors='coerce')

    df['peak'] = pd.to_numeric(df['peak'], errors='coerce')
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')

    df = df.dropna(subset=['worldwide_gross', 'rank', 'peak'])  # Remove bad rows

    return df
