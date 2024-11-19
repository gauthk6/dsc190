# Biology Textbook RAG Pipeline

This project implements a Retrieval-Augmented Generation (RAG) pipeline for the OpenStax Biology 2e textbook. It chunks the textbook by sections and uses Llama-2-7B-Chat for answering questions.

## Features

- Extracts and chunks textbook content by sections (e.g., 1.1, 1.2, etc.)
- Uses sentence-transformers for efficient embeddings
- Employs Llama-2-7B-Chat for powerful local inference
- Stores vectors using ChromaDB for fast retrieval
- Provides section references for answers
- Includes comprehensive evaluation pipeline for testing model performance
- Supports structured question processing with multiple-choice evaluation
- Automated evaluation results tracking and analysis

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

The first time you run the script, it will automatically download the Llama-2-7B-Chat model (about 4GB).

## Usage

### Basic Question Answering
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

### Running Evaluations
```python
from eval_pipeline import EvaluationPipeline

evaluator = EvaluationPipeline(
    questions_csv="path/to/questions.csv",
    pdf_path="biology2e_textbook.pdf",
    rag_dir="./biology_rag"
)

# Run evaluation on all questions
results = evaluator.run_evaluation()
```

The evaluation results will be automatically saved in the `biology_rag/evaluation_results` directory with timestamps.

## Technical Details

- Embedding Model: all-MiniLM-L6-v2 (lightweight and efficient)
- LLM: Llama-2-7B-Chat (GGUF format for optimized local inference)
- Vector Store: ChromaDB (persistent storage)
- Chunk Size: 500 characters with 50 character overlap
- Top-k retrieval: 3 most relevant chunks
- Evaluation metrics: Accuracy, response quality, and source relevance

## Project Structure

```
biology_rag/
├── models/                    # Stores downloaded LLM models
├── vector_db/                 # ChromaDB vector storage
├── evaluation_results/        # Timestamped evaluation results
└── ...

Key Files:
- biology_rag.py          # Core RAG implementation
- eval_pipeline.py        # Evaluation system
- question_processor.py   # Question handling and processing
```

## Note

This implementation works both on Google Colab and locally on M1 Macs. The Llama-2-7B-Chat model is chosen specifically for its good performance on M1 chips while still being able to run on free resources.
