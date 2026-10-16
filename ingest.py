import os
import sys
import time
from pathlib import Path
import warnings

# Suppress noisy deprecation warnings from langchain-community and langchain
warnings.filterwarnings("ignore")

# Load environment variables if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Document Loader
try:
    from langchain_community.document_loaders import PyPDFLoader
except ImportError:
    print("Error: 'langchain-community' or 'pypdf' not installed.")
    print("Run: pip install langchain langchain-community langchain-google-genai chromadb pypdf")
    sys.exit(1)

# Text Splitter
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        # pyrefly: ignore [missing-import]
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        print("Error: 'langchain' text splitters not found.")
        print("Run: pip install langchain")
        sys.exit(1)

# Embeddings
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except ImportError:
    print("Error: 'langchain-google-genai' not installed.")
    print("Run: pip install langchain-google-genai")
    sys.exit(1)

# Vector Store
try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    try:
        # pyrefly: ignore [missing-import]
        from langchain_chroma import Chroma
    except ImportError:
        print("Error: 'chromadb' or 'langchain-community' not installed.")
        print("Run: pip install chromadb langchain-community")
        sys.exit(1)


def get_pdf_path() -> str:
    """Determine the PDF file to ingest."""
    # 1. From CLI argument if provided
    if len(sys.argv) > 1:
        custom_path = sys.argv[1]
        if os.path.exists(custom_path):
            return custom_path
        else:
            print(f"Error: Specified PDF file not found at '{custom_path}'")
            sys.exit(1)

    # 2. Check for sample.pdf in current directory
    if os.path.exists("sample.pdf"):
        return "sample.pdf"

    # 3. Check for any PDF in current directory
    pdf_files = list(Path(".").glob("*.pdf"))
    if pdf_files:
        return str(pdf_files[0])

    print("Error: No PDF file found in current directory and no path passed as argument.")
    print("Usage: python ingest.py [path_to_pdf]")
    sys.exit(1)


def main():
    # Verify Google API Key is set and not a placeholder
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key.strip() in ("your_gemini_api_key_here", "your-gemini-api-key", ""):
        print("ERROR: Invalid or placeholder GOOGLE_API_KEY detected.")
        print("Please replace 'your_gemini_api_key_here' in your .env file with your actual Gemini API key:")
        print("  GOOGLE_API_KEY=AIzaSy...")
        print("You can get a free API key at: https://aistudio.google.com/app/apikey\n")
        sys.exit(1)

    # Resolve PDF file path
    pdf_path = get_pdf_path()
    print(f"Loading PDF file from: {pdf_path}")

    # 1. Load PDF using LangChain's PyPDFLoader
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Successfully loaded {len(documents)} page(s) from PDF.")

    # 2. Split into chunks of 500 characters with 50 characters overlap
    print("Splitting document into chunks (chunk_size=500, chunk_overlap=50)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)

    # Print how many chunks were created
    print(f"Total chunks created: {len(chunks)}")

    # 3. Initialize Google Generative AI Embeddings
    print("Initializing Google Generative AI embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )

    # 4. Store chunks in local Chroma vector database folder called 'db'
    db_folder = "db"
    print(f"Embedding chunks and saving to local Chroma database at '{db_folder}'...")
    vector_store = Chroma(
        persist_directory=db_folder,
        embedding_function=embeddings,
    )

    batch_size = 50
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    print(f"Ingesting {len(chunks)} chunks in {total_batches} batches...")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        print(f"  Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        max_retries = 5
        for attempt in range(max_retries):
            try:
                vector_store.add_documents(batch)
                break
            except Exception as e:
                err_str = str(e)
                if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                    wait_time = 15 * (attempt + 1)
                    print(f"    [Rate Limit] Free-tier limit reached. Pausing for {wait_time}s before retry (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    raise e
        time.sleep(1)

    # In older versions of langchain/chroma, persist() needed to be called explicitly
    if hasattr(vector_store, "persist"):
        vector_store.persist()

    print(f"\n[SUCCESS] Successfully ingested {len(chunks)} chunks into '{db_folder}'!")


if __name__ == "__main__":
    main()
