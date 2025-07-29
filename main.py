from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from scraper import scrape_wikipedia_highest_grossing_films
from analysis import answer_wikipedia_questions
from visualizer import plot_regression

app = FastAPI()

@app.post("/")
async def analyze(file: UploadFile = File(...)):
    content = await file.read()
    question = content.decode()

    if "highest grossing films" in question.lower():
        df = scrape_wikipedia_highest_grossing_films()
        result = answer_wikipedia_questions(df, plot_function=plot_regression)
        return JSONResponse(content=result)
    else:
        return JSONResponse(content=["Unsupported question type"] * 4)
