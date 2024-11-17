# Biology Textbook RAG Pipeline

This project implements a Retrieval-Augmented Generation (RAG) pipeline for the OpenStax Biology 2e textbook. It chunks the textbook by sections and uses Llama 2 for answering questions with detailed source tracking and performance metrics.

## Features

- Extracts and chunks textbook content by sections (e.g., 1.1, 1.2, etc.)
- Uses sentence-transformers for efficient embeddings
- Employs Llama 2 7B Chat for local inference
- Stores vectors using ChromaDB for fast retrieval
- Provides detailed source references and timing metrics for answers
- Supports streaming output for real-time answer generation

## Setup

1. Download the OpenStax Biology 2e textbook PDF
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Run the pipeline:
```bash
python biology_rag.py
```

The first time you run the script, it will automatically download the Llama 2 7B Chat model (quantized version) from Hugging Face.

## Usage

The script includes an example question about cellular respiration. You can modify the question in the `main()` function or import the `BiologyTextbookRAG` class in your own code:

```python
from biology_rag import BiologyTextbookRAG

# Initialize the RAG pipeline
rag = BiologyTextbookRAG(
    pdf_path="path/to/biology2e_textbook.pdf",
    project_dir="./biology_rag"
)

# Extract sections and create vector store
sections = rag.extract_sections()
vectorstore = rag.create_vectorstore()

# Setup QA chain and ask questions
qa_chain = rag.setup_qa_chain(vectorstore)
result = rag.answer_question(qa_chain, "Your question here?")

# Access results
print(result["answer"])
print(result["source_sections"])
print(result["inference_time"])
```

## Technical Details

- **Embedding Model**: all-MiniLM-L6-v2 (SentenceTransformers)
- **LLM**: Llama 2 7B Chat (Quantized for efficient inference)
- **Vector Store**: ChromaDB with persistent storage
- **Chunk Size**: 1000 characters with 200 character overlap
- **Top-k retrieval**: 3 most relevant chunks
- **Project Structure**:
  - `models/`: Stores the downloaded Llama 2 model
  - `vector_db/`: Contains the persistent ChromaDB vector store
  
## Performance Metrics

The system provides detailed timing information for:
- Total execution time
- Inference time per question
- Source section retrieval

## Output Format

For each question, the system returns:
- Generated answer
- Source sections used (e.g., "1.1", "1.2")
- Detailed source texts with section references
- Timing information
- Timestamp of execution

## Notes

- The implementation uses a quantized version of Llama 2 for efficient CPU inference
- GPU acceleration can be enabled by adjusting the `gpu_layers` parameter
- The system supports streaming output for real-time answer generation
- All components use persistent storage for efficient reuse across sessions

## Requirements

Main dependencies:
- PyPDF2
- chromadb
- langchain
- CTransformers
- sentence-transformers
- huggingface_hub
- tqdm
