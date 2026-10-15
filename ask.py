import os
import sys
from pathlib import Path

import warnings

# Suppress noisy deprecation warnings from langchain-community and langchain
warnings.filterwarnings("ignore")

# 1. Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 2. Check for required Google / Gemini API key
api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not api_key or api_key.strip() in ("your_gemini_api_key_here", "your-gemini-api-key", ""):
    print("ERROR: Invalid or placeholder GOOGLE_API_KEY detected.")
    print("Please replace 'your_gemini_api_key_here' in your .env file with your actual Gemini API key:")
    print("  GOOGLE_API_KEY=AIzaSy...")
    print("You can get a free API key at: https://aistudio.google.com/app/apikey\n")
    sys.exit(1)

# 3. Check for existence of the Chroma database directory
db_dir = "db"
if not os.path.exists(db_dir):
    print(f"ERROR: Chroma database folder '{db_dir}' was not found.")
    print("Please run 'python ingest.py' first to process and index your PDF documents.")
    sys.exit(1)

# 4. Import LangChain components
try:
    from langchain_chroma import Chroma
except ImportError:
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        print("Error: Chroma vectorstore library not found.")
        print("Run: pip install chromadb langchain-community")
        sys.exit(1)

try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
except ImportError:
    print("Error: 'langchain-google-genai' not installed.")
    print("Run: pip install langchain-google-genai")
    sys.exit(1)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs):
    """Format retrieved document chunks as formatted text for prompt context."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_rag_chain():
    """Initializes embeddings, Chroma vector store, retriever, and LCEL chain."""
    print("Loading Chroma database and embedding model...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )

    vector_store = Chroma(
        persist_directory=db_dir,
        embedding_function=embeddings,
    )

    # Check if the database has any documents indexed
    try:
        doc_count = vector_store._collection.count()
        if doc_count == 0:
            print(f"\n[WARNING] The Chroma database in '{db_dir}' contains 0 documents.")
            print("Make sure to run 'python ingest.py' first with your API key set to embed your PDF!\n")
    except Exception:
        pass

    # Top 3 most relevant chunks
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    # Gemini Chat model
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=api_key,
        temperature=0.2,
    )

    # Prompt template
    template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If the answer is not contained in the context, say that you don't know based on the provided documents.
Keep your answer clear, accurate, and concise.

Context:
{context}

Question:
{question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    # LangChain LCEL Chain:
    # 1. Takes user's question
    # 2. Passes retriever | format_docs to 'context' and question to 'question'
    # 3. Pipes into prompt
    # 4. Pipes into LLM
    # 5. Parses output into string
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def main():
    rag_chain = build_rag_chain()

    # If question passed via command-line argument:
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:]).strip()
        print(f"\nQuestion: {question}\nThinking...")
        answer = rag_chain.invoke(question)
        print("\nAnswer:")
        print(answer)
        return

    # Interactive terminal session
    print("\n" + "=" * 60)
    print("Ready! Ask questions about your ingested PDF.")
    print("Type your question and press Enter. Type 'exit' or 'q' to quit.")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("Ask a question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        print("\nSearching and generating answer...")
        try:
            answer = rag_chain.invoke(question)
            print("\nAnswer:")
            print(answer)
            print("-" * 60 + "\n")
        except Exception as e:
            err = str(e)
            if "RESOURCE_EXHAUSTED" in err or "429" in err:
                print("\n[Rate Limit] Gemini free-tier rate limit reached (100 req/min). Please wait 20-30 seconds and try again.\n")
            elif "API_KEY_INVALID" in err or "400" in err:
                print("\n[Auth Error] API key was rejected. Please verify the key in your .env file.\n")
            else:
                print(f"\nError generating answer: {e}\n")


if __name__ == "__main__":
    main()
