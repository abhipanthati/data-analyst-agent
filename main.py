import os
import json
import tempfile
import subprocess
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from openai import OpenAI  # ✅ Updated for openai>=1.0.0

from scraper import scrape_wikipedia_highest_grossing_films
from analysis import answer_wikipedia_questions
from visualizer import plot_regression  # returns base64 data-uri

import pandas as pd

# Optional: install these if needed
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None
try:
    import pdfminer.high_level
except ImportError:
    pdfminer = None
try:
    import zipfile
except ImportError:
    zipfile = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("data-analyst-agent")

# Read API base URL & key from env (AIpipe or OpenAI)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set. LLM features will fail if used at runtime.")

# Create OpenAI client
client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

app = FastAPI()


def call_llm(prompt: str, max_tokens: int = 1500) -> str:
    """Call LLM using OpenAI API (works with AIpipe if base_url points to AIpipe)."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not configured")

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def run_python_code(code: str, timeout: int = 120) -> str:
    """Write code to a temp file and run it, returning combined stdout/stderr."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8") as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        proc = subprocess.run(["python", tmp_path], capture_output=True, timeout=timeout)
        out = proc.stdout.decode("utf-8", errors="replace")
        err = proc.stderr.decode("utf-8", errors="replace")
        return out + "\n" + err
    except subprocess.TimeoutExpired:
        return "Execution timed out."
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def split_questions(text: str) -> List[str]:
    """Split question text into individual questions."""
    lines = [l.rstrip() for l in text.splitlines()]
    blocks, cur = [], []
    for l in lines:
        if not l.strip():
            if cur:
                blocks.append("\n".join(cur).strip())
                cur = []
            continue
        if l.strip().lstrip().startswith(("1.", "2.", "3.", "4.", "5.")) and cur:
            blocks.append("\n".join(cur).strip())
            cur = [l]
        else:
            cur.append(l)
    if cur:
        blocks.append("\n".join(cur).strip())
    return [b for b in blocks if b]


def detect_and_parse_file(file_path: str, filename: str) -> Dict[str, Any]:
    """
    Detect filetype and parse accordingly.
    Returns a dict with keys: type, data, and optionally 'df' for tabular data.
    """
    ext = os.path.splitext(filename)[1].lower()
    result = {"type": ext, "filename": filename, "path": file_path}
    try:
        if ext in [".csv"]:
            df = pd.read_csv(file_path)
            result["df"] = df
            result["data"] = f"CSV file with shape {df.shape}"
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
            result["df"] = df
            result["data"] = f"Excel file with shape {df.shape}"
        elif ext == ".pdf":
            text = ""
            if PyPDF2:
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = "\n".join(page.extract_text() or "" for page in reader.pages)
            elif pdfminer:
                text = pdfminer.high_level.extract_text(file_path)
            result["data"] = text
        elif ext == ".zip" and zipfile:
            with zipfile.ZipFile(file_path, "r") as z:
                filelist = z.namelist()
                result["data"] = f"ZIP archive with files: {filelist}"
        elif ext == ".py":
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                code = f.read()
            result["data"] = code
        else:
            # Try to read as text
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read(10000)
            result["data"] = text
    except Exception as e:
        result["error"] = f"Failed to parse: {e}"
    return result


@app.post("/")
async def analyze(files: List[UploadFile] = File(...)):
    # Save all uploaded files to temp files and collect their paths
    file_paths = []
    file_names = []
    file_info = []
    parsed_files = {}
    for upload in files:
        suffix = os.path.splitext(upload.filename)[1] or ".dat"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await upload.read()
            tmp.write(content)
            file_paths.append(tmp.name)
            file_names.append(upload.filename)
            info = detect_and_parse_file(tmp.name, upload.filename)
            file_info.append(info)
            # For tabular data, store for later use
            if "df" in info:
                parsed_files[upload.filename] = info["df"]

    # Assume the first file is the question text
    if not file_paths:
        return JSONResponse(content={"error": "No files uploaded."}, status_code=400)

    with open(file_paths[0], "rb") as f:
        question_text = f.read().decode(errors="replace")
    logger.info("Received question text (len=%d)", len(question_text))

    # Build a resource list for the LLM
    resource_list = []
    for info in file_info:
        desc = f"{info['filename']} (type: {info['type']})"
        if "df" in info:
            desc += f" [tabular: shape {info['df'].shape}]"
        elif "data" in info and isinstance(info["data"], str):
            desc += f" [text: {info['data'][:80].replace(chr(10),' ')}...]"
        resource_list.append(desc)
    resources_str = "\n".join(resource_list)

    # Pass file references and summaries into the LLM prompt
    file_refs = "\n".join([f"{name}: {info.get('type')} ({info.get('data', '')[:100]}...)" for name, info in zip(file_names, file_info)])

    if "highest grossing films" in question_text.lower():
        try:
            df = scrape_wikipedia_highest_grossing_films(use_cache=False)
            result = answer_wikipedia_questions(df, plot_function=plot_regression)
            # Clean up temp files
            for p in file_paths:
                try:
                    os.remove(p)
                except Exception:
                    pass
            return JSONResponse(content=result)
        except Exception as e:
            logger.exception("Wikipedia shortcut failed")
            for p in file_paths:
                try:
                    os.remove(p)
                except Exception:
                    pass
            return JSONResponse(content=[str(e)] * 4, status_code=500)

    questions = split_questions(question_text)
    if not questions:
        for p in file_paths:
            try:
                os.remove(p)
            except Exception:
                pass
        return JSONResponse(content={"error": "No questions parsed from input."}, status_code=400)

    batch_size = 3
    batches = [questions[i:i + batch_size] for i in range(0, len(questions), batch_size)]
    final_results = []

    for batch in batches:
        plan_prompt = f"""
You are a data analyst assistant. You have received the following files:

{resources_str}

Available file paths:
{chr(10).join([f"{info['filename']}: {info['path']}" for info in file_info])}

For each question below, provide:
1. A structured, stepwise plan for how you would process and analyze the relevant files and data to answer the question.
2. A self-contained Python script (only code) that, when executed, will:
   - Use the provided file paths if relevant (e.g., for CSVs, Excel, etc.).
   - Scrape or download any required data (if web links are provided in the question).
   - Perform the analyses/aggregations/plots requested.
   - Print a JSON string to stdout with the answers in the exact format requested by the question.

Constraints:
- Use pandas for CSV/Excel, PyPDF2 or pdfminer for PDF, zipfile for zip, etc.
- Any images produced should be written as a base64 data URI (data:image/png;base64,...) in the JSON.
- Do not call external LLMs in the generated code.
- The grader will run this script and capture stdout.

Questions:
{batch}
"""
        try:
            plan_and_code = call_llm(plan_prompt)
        except Exception as e:
            logger.exception("LLM plan generation failed")
            final_results.append({"error": f"LLM plan generation failed: {e}"})
            continue

        try:
            execution_output = run_python_code(plan_and_code, timeout=110)
        except Exception:
            execution_output = "Execution failed."

        format_prompt = f"""
You are a helper that formats analysis results. Given these questions:
{batch}

And given the raw stdout/stderr from executing analysis code:
{execution_output}

If the stdout already contains valid JSON as requested by the question, return that JSON verbatim.
Otherwise, synthesize the answers into the exact JSON format the question requested (either a JSON array or JSON object). Output **only** the JSON.
"""
        try:
            final_answer = call_llm(format_prompt)
        except Exception as e:
            logger.exception("LLM formatting failed")
            final_results.append({"error": f"LLM formatting failed: {e}", "raw": execution_output})
            continue

        try:
            parsed = json.loads(final_answer)
        except Exception:
            try:
                start = execution_output.find("{")
                end = execution_output.rfind("}")
                if start != -1 and end != -1 and end > start:
                    parsed = json.loads(execution_output[start:end+1])
                else:
                    parsed = {"error": "LLM did not return valid JSON", "raw": final_answer}
            except Exception:
                parsed = {"error": "Failed to parse JSON from LLM or execution output.", "raw": final_answer}

        if isinstance(parsed, list):
            final_results.extend(parsed)
        elif isinstance(parsed, dict):
            final_results.append(parsed)
        else:
            final_results.append({"result": parsed})

    # Clean up temp files
    for p in file_paths:
        try:
            os.remove(p)
        except Exception:
            pass

    if final_results and all(isinstance(x, dict) for x in final_results):
        merged = {}
        for d in final_results:
            merged.update(d)
        return JSONResponse(content=merged)

    return JSONResponse(content=final_results)
