# Biology Textbook RAG Pipeline

This project implements a Retrieval-Augmented Generation (RAG) pipeline for the OpenStax Biology 2e textbook. It chunks the textbook by sections and uses GPT4All for answering questions.

## Features

- Extracts and chunks textbook content by sections (e.g., 1.1, 1.2, etc.)
- Uses sentence-transformers for efficient embeddings
- Employs GPT4All for local inference (works great on M1 Macs!)
- Stores vectors using ChromaDB for fast retrieval
- Provides section references for answers

## Setup

1. Download the OpenStax Biology 2e textbook PDF and place it in this directory as `biology2e_textbook.pdf`

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the pipeline:
```bash
python biology_rag.py
```

The first time you run the script, it will automatically download the GPT4All model (about 4GB).

## Usage

The script currently includes an example question. You can modify the question in the `main()` function or import the `BiologyTextbookRAG` class in your own code:

```python
from biology_rag import BiologyTextbookRAG

rag = BiologyTextbookRAG("biology2e_textbook.pdf")
sections = rag.extract_sections()
vectorstore = rag.create_vectorstore()
qa_chain = rag.setup_qa_chain(vectorstore)

result = rag.answer_question(qa_chain, "Your question here?")
print(result["answer"])
print(result["source_sections"])
```

## Technical Details

- Embedding Model: all-MiniLM-L6-v2 (lightweight and efficient)
- LLM: GPT4All-Falcon (optimized for M1 Macs)
- Vector Store: ChromaDB (persistent storage)
- Chunk Size: 500 characters with 50 character overlap
- Top-k retrieval: 3 most relevant chunks

## Note

This implementation works both on Google Colab and locally on M1 Macs. The GPT4All model is chosen specifically for its good performance on M1 chips while still being able to run on free resources.
