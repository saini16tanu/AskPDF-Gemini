import os
import sys

# Auto-reexec inside the project virtual environment if dependencies are missing in the current python
try:
    import fastapi
except ImportError:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    venv_py = os.path.join(base_dir, ".venv", "bin", "python")
    if os.path.exists(venv_py):
        os.execv(venv_py, [venv_py] + sys.argv)

import warnings
from contextlib import asynccontextmanager
from typing import Optional

# Suppress deprecation warnings from LangChain ecosystem
warnings.filterwarnings("ignore")

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import the existing RAG chain constructor from ask.py
try:
    from ask import build_rag_chain
except ImportError as err:
    print(f"Error importing build_rag_chain from ask.py: {err}")
    sys.exit(1)


# Global chain instance cached in memory across requests
rag_chain = None


def get_rag_chain():
    """Lazily initialize or retrieve the cached RAG chain."""
    global rag_chain
    if rag_chain is None:
        rag_chain = build_rag_chain()
    return rag_chain


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context to pre-warm the RAG chain during startup."""
    try:
        get_rag_chain()
    except Exception as e:
        print(f"[WARNING] Failed to pre-warm RAG chain at startup: {e}")
    yield


# Initialize FastAPI application
app = FastAPI(
    title="Ask Gemini API",
    description="FastAPI server wrapping the LangChain RAG pipeline with Google Gemini",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS (Cross-Origin Resource Sharing)
# Allows browsers and web clients from any origin to make requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request and response schemas
class AskRequest(BaseModel):
    question: str = Field(
        ...,
        description="The question to ask the RAG model",
        examples=["What is a binary tree?"],
    )


class AskResponse(BaseModel):
    question: str = Field(..., description="The user query that was asked")
    answer: str = Field(..., description="The generated response from the RAG chain")


from fastapi.responses import FileResponse

@app.get("/")
def root():
    """Serves the 3D chatbot web interface."""
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "status": "online",
        "message": "Ask Gemini API is running.",
        "endpoint": "/ask",
        "docs": "/docs",
    }


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "online",
        "message": "Ask Gemini API is running.",
        "endpoint": "/ask",
        "docs": "/docs",
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest):
    """
    POST /ask endpoint:
    Takes a JSON body with a question, processes it through the existing
    RAG chain, and returns the response as JSON.
    """
    cleaned_question = payload.question.strip()
    if not cleaned_question:
        raise HTTPException(
            status_code=400,
            detail="The 'question' field cannot be empty.",
        )

    try:
        chain = get_rag_chain()
        result = chain.invoke(cleaned_question)
        answer_text = str(result)
        return AskResponse(question=cleaned_question, answer=answer_text)
    except Exception as exc:
        err_msg = str(exc)
        if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
            raise HTTPException(
                status_code=429,
                detail="Rate limit reached for the Gemini API. Please wait a moment and try again.",
            )
        raise HTTPException(
            status_code=500,
            detail=f"Error executing RAG chain: {err_msg}",
        )


@app.get("/ask", response_model=AskResponse)
def ask_question_get(
    question: str = Query(..., description="The question to ask the RAG model")
):
    """
    GET /ask endpoint:
    Allows passing the question as a query parameter (e.g. /ask?question=...).
    """
    cleaned_question = question.strip()
    if not cleaned_question:
        raise HTTPException(
            status_code=400,
            detail="The 'question' parameter cannot be empty.",
        )

    try:
        chain = get_rag_chain()
        result = chain.invoke(cleaned_question)
        answer_text = str(result)
        return AskResponse(question=cleaned_question, answer=answer_text)
    except Exception as exc:
        err_msg = str(exc)
        if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
            raise HTTPException(
                status_code=429,
                detail="Rate limit reached for the Gemini API. Please wait a moment and try again.",
            )
        raise HTTPException(
            status_code=500,
            detail=f"Error executing RAG chain: {err_msg}",
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting Ask Gemini API server at http://{host}:{port}")
    uvicorn.run("server:app", host=host, port=port, reload=True)
