"""Entry point: python scripts/run_api.py"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("serving.mnist_digit_api:app", host="0.0.0.0", port=port, reload=False)
