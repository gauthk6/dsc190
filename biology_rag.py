import os
import re
from pathlib import Path
from typing import List, Dict
from PyPDF2 import PdfReader
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.llms import CTransformers
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from huggingface_hub import hf_hub_download
import time

class BiologyTextbookRAG:
    def __init__(self, pdf_path: str, project_dir: str = "./biology_rag"):
        """
        Initialize the RAG pipeline for Biology textbook
        
        Args:
            pdf_path: Path to the Biology 2e PDF file
            project_dir: Directory to store all RAG-related files
        """
        self.pdf_path = pdf_path
        
        # Set up project directory structure
        self.project_dir = Path(project_dir)
        self.models_dir = self.project_dir / "models"
        self.vector_db_path = self.project_dir / "vector_db"
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Project directory: {self.project_dir}")
        print(f"Models directory: {self.models_dir}")
        print(f"Vector database directory: {self.vector_db_path}")
        
        self.model_path = self._get_or_download_model()
        self.sections = {}
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def _get_or_download_model(self) -> str:
        """Get cached model or download if not present"""
        model_filename = "llama-2-7b-chat.Q4_K_M.gguf"
        local_model_path = self.models_dir / model_filename

        if local_model_path.exists():
            print(f"Using cached model from {local_model_path}")
            return str(local_model_path)
        
        print("Downloading Llama-2-7B-Chat model... This might take a while...")
        downloaded_path = hf_hub_download(
            repo_id="TheBloke/Llama-2-7B-Chat-GGUF",
            filename=model_filename,
            local_dir=self.models_dir
        )
        
        return downloaded_path

    def _extract_section_pattern(self, text: str) -> str:
        """Extract section number from text using regex"""
        pattern = r'\b(\d+\.\d+)\b'
        match = re.search(pattern, text[:100])  # Look in first 100 chars
        return match.group(1) if match else None

    def extract_sections(self) -> Dict[str, str]:
        """Extract sections from the PDF file"""
        print("Reading PDF and extracting sections...")
        reader = PdfReader(self.pdf_path)
        current_section = None
        current_text = []
        
        for page in tqdm(reader.pages):
            text = page.extract_text()
            
            # Try to find a section number at the start of the page
            section_number = self._extract_section_pattern(text)
            
            if section_number:
                # If we were building a previous section, save it
                if current_section:
                    self.sections[current_section] = '\n'.join(current_text)
                
                # Start new section
                current_section = section_number
                current_text = [text]
            else:
                # Continue building current section if we have one
                if current_section:
                    current_text.append(text)
        
        # Don't forget to save the last section
        if current_section:
            self.sections[current_section] = '\n'.join(current_text)
        
        print(f"Extracted {len(self.sections)} sections from the textbook")
        return self.sections

    def create_vectorstore(self) -> Chroma:
        """Create a vector store from the extracted sections"""
        print("Creating vector store...")
        print(f"Vector store will be saved to: {self.vector_db_path}")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )
        
        documents = []
        metadatas = []
        
        for section_num, content in self.sections.items():
            chunks = text_splitter.split_text(content)
            documents.extend(chunks)
            metadatas.extend([{"section": section_num} for _ in chunks])
        
        vectorstore = Chroma.from_texts(
            texts=documents,
            embedding=self.embeddings,
            metadatas=metadatas,
            persist_directory=str(self.vector_db_path)
        )
        
        return vectorstore

    def setup_qa_chain(self, vectorstore: Chroma) -> RetrievalQA:
        """Set up the question-answering chain"""
        # Initialize the Llama model with CTransformers
        callbacks = [StreamingStdOutCallbackHandler()]
        
        llm = CTransformers(
            model=self.model_path,
            model_type="llama",
            max_new_tokens=512,
            temperature=0.1,    # Low temperature for more focused responses
            context_length=2048,
            callbacks=callbacks,
            config={
                'context_length': 2048,
                'gpu_layers': 0  # Set to higher number if you have GPU
            }
        )
        
        # Create and return the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            return_source_documents=True,
        )
        
        return qa_chain

    def answer_question(self, qa_chain: RetrievalQA, question: str) -> dict:
        """
        Answer a question using the QA chain
        
        Args:
            qa_chain: The QA chain to use
            question: The question to answer
            
        Returns:
            dict containing the answer, source sections, source texts, and timing info
        """
        # Start timing the inference
        inference_start = time.time()
        
        result = qa_chain({"query": question})
        
        # End timing the inference
        inference_time = time.time() - inference_start
        
        # Extract source sections and texts
        source_sections = set()
        source_texts = []
        
        for doc in result.get("source_documents", []):
            section = doc.metadata.get("section")
            if section:
                source_sections.add(section)
                source_texts.append({
                    'section': section,
                    'text': doc.page_content
                })
        
        return {
            "answer": result["result"],
            "source_sections": list(sorted(source_sections)),
            "source_texts": source_texts,
            "inference_time": inference_time
        }

def main():
   # Start timing the total execution
    total_start = time.time()
    
    # Define project directory relative to current script
    current_dir = Path(__file__).parent
    project_dir = current_dir / "biology_rag"
    
    # Initialize the RAG pipeline
    pdf_path = "/Users/gauthamkishore/DSC 190/biology2e_textbook.pdf"
    rag = BiologyTextbookRAG(
        pdf_path=pdf_path,
        project_dir=project_dir
    )
    
    print("\nInitializing Biology Textbook RAG pipeline...")
    
    # Extract sections
    print("\nExtracting sections from PDF...")
    sections = rag.extract_sections()
    
    # Create vector store
    print("\nCreating vector store...")
    vectorstore = rag.create_vectorstore()
    
    # Setup QA chain
    print("\nSetting up QA chain...")
    # Set environment variable to prevent tokenizer warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    qa_chain = rag.setup_qa_chain(vectorstore)
    
    # Example question about cellular respiration
    test_question = "What is the role of ATP in cellular respiration? Explain it simply."
    print("\nTesting with question:", test_question)
    print("\nGenerating answer (this may take a moment)...")
    
    result = rag.answer_question(qa_chain, test_question)
    
    # Calculate total execution time
    total_time = time.time() - total_start
    
    # Print results with timing information
    print("\n" + "="*100)
    print("QUESTION:", test_question)
    print("="*100)
    print("\nANSWER:", result['answer'])
    print("\n" + "-"*100)
    print("\nSOURCE SECTIONS:", ', '.join(result['source_sections']))
    print("\nSOURCE TEXTS:")
    print("-"*100)
    for i, source in enumerate(result['source_texts'], 1):
        print(f"\nChunk {i} (Section {source['section']}):")
        print("-"*50)
        print(source['text'].strip())
        print("-"*50)
    
    print("\nTIMING INFORMATION:")
    print("-"*100)
    print(f"Inference Time: {result['inference_time']:.2f} seconds")
    print(f"Total Execution Time: {total_time:.2f} seconds")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100)

if __name__ == "__main__":
    main()