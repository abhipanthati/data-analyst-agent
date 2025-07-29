# Copilot Instructions for Data Analyst Agent

## Project Architecture

This project is a FastAPI-based data analysis agent. It exposes a single POST endpoint (`/`) that accepts a file upload containing a question. The workflow is:

1. **Scraping**: `scraper.py` provides `scrape_data(question)` to fetch data relevant to the question.
2. **Analysis**: `analysis.py` provides `perform_analysis(question, df)` to process and analyze the scraped data.
3. **Visualization**: `visualizer.py` provides `generate_plot(df, question)` to create a plot and return its URI.
4. **Orchestration**: `main.py` wires these together in the FastAPI route, returning results as a list or dict, with the plot URI appended or added as `plot`.

## Developer Workflows

- **Run the API server**:
  ```powershell
  uvicorn main:app --reload
  ```
- **Install dependencies**:
  ```powershell
  pip install -r requirements.txt
  ```
- **Test modules**: Run scripts directly for debugging (e.g., `python scraper.py`).

## Conventions & Patterns

- All data flows through the FastAPI endpoint in `main.py`.
- Results are returned as either a list (with plot URI appended) or a dict (with plot URI under `plot`).
- Each major function is separated into its own script for modularity.
- External dependencies (FastAPI, Uvicorn) are required; ensure they are listed in `requirements.txt`.

## Integration Points

- FastAPI is used for serving the API.
- Uvicorn is recommended for local development server.
- Data scraping, analysis, and visualization are decoupled into separate modules.

## Key Files

- `main.py`: FastAPI app and orchestration logic.
- `scraper.py`: Data scraping logic.
- `analysis.py`: Data analysis logic.
- `visualizer.py`: Plot generation logic.
- `requirements.txt`: Python dependencies.

---

If any conventions, data formats, or workflows are unclear or missing, please provide feedback to improve these instructions.
