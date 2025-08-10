#!/bin/bash
# test_local.sh
# Local Wikipedia + General LLM mode test for Data Analyst Agent

set -euo pipefail

MODE=$1
ALL_OK=true

if [[ -z "$MODE" ]]; then
  echo "Usage: $0 <wiki|llm|all>"
  exit 1
fi

run_mode() {
    local mode=$1
    echo "Starting FastAPI app..."
    uvicorn main:app --reload --port 8000 > /dev/null 2>&1 &
    PID=$!
    sleep 5

    if [[ "$mode" == "wiki" ]]; then
        echo "=== Running Wikipedia Eval Mode ==="
        cat > question.txt << 'EOF'
Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.

1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
   Return as a base-64 encoded data URI, "data:image/png;base64,iVBORw0KG..." under 100,000 bytes.
EOF
        curl -s -F "file=@question.txt" http://127.0.0.1:8000/ | tee response.json
        python3 << 'PYCODE'
import json, re, base64
data = json.load(open("response.json"))
assert isinstance(data, list) and len(data) == 4, "List of length 4 required"
print("✅ JSON array length = 4")
assert data[0] == 1, f"Q1 incorrect: {data[0]}"
print("✅ Q1 correct")
assert re.search(r'titanic', str(data[1]), re.I), f"Q2 missing Titanic: {data[1]}"
print("✅ Q2 contains Titanic")
corr = float(data[2])
assert abs(corr - 0.485782) <= 0.001, f"Q3 correlation off: {corr}"
print("✅ Q3 correlation correct")
assert isinstance(data[3], str) and data[3].startswith("data:image/png;base64,"), "Q4 not valid base64 PNG"
img_data = base64.b64decode(data[3].split(",")[1])
assert len(img_data) < 100_000, f"Q4 too large ({len(img_data)} bytes)"
print(f"✅ Q4 image size OK ({len(img_data)} bytes)")
PYCODE

    elif [[ "$mode" == "llm" ]]; then
        echo "=== Running General LLM Mode ==="
        cat > question.txt << 'EOF'
The dataset is available at:
https://people.sc.fsu.edu/~jburkardt/data/csv/airtravel.csv

1. Download the dataset and compute the average number of passengers for each month.
2. Return the answer as a JSON object mapping month names to averages.
EOF
        curl -s -F "file=@question.txt" http://127.0.0.1:8000/ | tee response.json
        python3 << 'PYCODE'
import json
data = json.load(open("response.json"))
assert isinstance(data, dict), "Output is not a JSON object"
assert all(isinstance(v, (int,float)) for v in data.values()), f"Non-numeric values found: {data}"
print("✅ LLM mode returned JSON object with numeric values")
PYCODE
    fi

    kill $PID
    echo "=== $mode test complete ==="
}

if [[ "$MODE" == "all" ]]; then
    run_mode wiki
    run_mode llm
    echo -e "\n\033[1;32mALL TESTS PASSED ✅\033[0m"
else
    run_mode "$MODE"
fi
