from pathlib import Path
import json
import time
from typing import List, Dict
from datetime import datetime
from tqdm import tqdm
from biology_rag import BiologyTextbookRAG
from question_processor import BiologyQuestionEvaluator

class EvaluationPipeline:
    def __init__(self, 
                 questions_csv: str,
                 pdf_path: str,
                 rag_dir: str = "./biology_rag",  # Use existing RAG directory
                 results_dir: str = "evaluation_results",
                 force_refresh_embeddings: bool = False):
        """
        Initialize the evaluation pipeline
        
        Args:
            questions_csv: Path to questions CSV file
            pdf_path: Path to biology textbook PDF
            rag_dir: Directory containing existing RAG setup
            results_dir: Directory to store evaluation results
            force_refresh_embeddings: Whether to force recreation of embeddings
        """
        # Use existing RAG directory
        self.rag_dir = Path(rag_dir)
        if not self.rag_dir.exists():
            raise ValueError(f"RAG directory {rag_dir} does not exist")
            
        # Create results directory within RAG directory
        self.results_dir = self.rag_dir / results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nInitializing Biology Evaluation Pipeline...")
        print(f"Using existing RAG directory: {self.rag_dir}")
        print(f"Results will be saved to: {self.results_dir}")
        
        # Initialize components
        self.question_processor = BiologyQuestionEvaluator(questions_csv)
        self.rag_system = BiologyTextbookRAG(
            pdf_path=pdf_path, 
            project_dir=str(self.rag_dir)
        )
        
        # Set up RAG components
        self.setup_rag(force_refresh_embeddings)
    
    def setup_rag(self, force_refresh: bool = False):
        """Set up RAG components (vectorstore and QA chain)"""
        print("\nSetting up RAG system...")
        
        # Create or load vector store
        print("Setting up vector store...")
        self.vectorstore = self.rag_system.create_vectorstore(force_refresh=force_refresh)
        
        # Setup QA chain
        print("Setting up QA chain...")
        self.qa_chain = self.rag_system.setup_qa_chain(self.vectorstore)
    
    def evaluate_single_question(self, question_data: Dict) -> Dict:
        """
        Evaluate a single question using the RAG system
        
        Args:
            question_data: Question data dictionary from BiologyQuestionEvaluator
            
        Returns:
            Dict containing evaluation results
        """
        question_id = question_data['id']
        prompt = question_data['llm_prompt']
        
        try:
            # Get RAG response
            start_time = time.time()
            rag_result = self.rag_system.answer_question(self.qa_chain, prompt)
            inference_time = time.time() - start_time
            
            # Evaluate the response
            eval_result = self.question_processor.evaluate_response(
                question_id, 
                rag_result['answer']
            )
            
            return {
                'question_id': question_id,
                'prompt': prompt,
                'llm_response': {
                    'answer': rag_result['answer'],
                    'source_sections': rag_result['source_sections'],
                    'source_texts': rag_result['source_texts'],
                    'inference_time': inference_time
                },
                'evaluation': {
                    'question_id': question_id,
                    'status': eval_result['status'],
                    'correct': eval_result['correct'],
                    'given_answer': eval_result['given_answer'],
                    'correct_answer': eval_result['correct_answer'],
                    'metadata': eval_result['metadata']
                },
                'metadata': question_data['metadata'],
                'timing': {
                    'inference_time': inference_time,
                    'timestamp': datetime.now().isoformat()
                }
            }

        except Exception as e:
            print(f"\nError evaluating question {question_id}: {str(e)}")
            return {
                'question_id': question_id,
                'status': 'error',
                'error_message': str(e),
                'timestamp': datetime.now().isoformat()
            }

    
    def run_evaluation(self, 
                  max_questions: int = None,
                  save_results: bool = True) -> Dict:
        """
        Run evaluation on all questions
        
        Args:
            max_questions: Optional limit on number of questions to evaluate
            save_results: Whether to save results to file
            
        Returns:
            Dict containing evaluation results and metrics
        """
        print("\nStarting evaluation...")
        start_time = time.time()
        
        # Get processed questions
        questions = self.question_processor.process_questions()
        if max_questions:
            questions = questions[:max_questions]
            print(f"\nLimiting evaluation to first {max_questions} questions")
        
        # Evaluate each question
        results = []
        metrics = {
            'total_questions': len(questions),
            'correct_answers': 0,
            'invalid_responses': 0,
            'errors': 0,
            'source_sections_used': set(),
            'average_inference_time': 0,
            'total_time': 0
        }
        
        print(f"\nEvaluating {len(questions)} questions...")
        for question in tqdm(questions, desc="Processing questions"):
            result = self.evaluate_single_question(question)
            results.append(result)
            
            # Update metrics
            if result.get('status') == 'error':
                metrics['errors'] += 1
                continue
                
            if result['evaluation']['status'] == 'valid_response':
                if result['evaluation']['correct']:
                    metrics['correct_answers'] += 1
            else:
                metrics['invalid_responses'] += 1
            
            metrics['source_sections_used'].update(
                result['llm_response']['source_sections']  # Changed from rag_response to llm_response
            )
            metrics['average_inference_time'] += result['timing']['inference_time']
        
        # Finalize metrics
        metrics['accuracy'] = metrics['correct_answers'] / metrics['total_questions']
        metrics['average_inference_time'] /= len(questions)
        metrics['source_sections_used'] = sorted(list(metrics['source_sections_used']))
        metrics['total_time'] = time.time() - start_time
        
        # Prepare final results
        evaluation_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'num_questions': len(questions)
            },
            'metrics': metrics,
            'detailed_results': results
        }
        
        # Save results if requested
        if save_results:
            results_file = self.results_dir / f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
            print(f"\nResults saved to {results_file}")
        
        # Print summary
        print("\nEvaluation Summary:")
        print(f"Total Questions: {metrics['total_questions']}")
        print(f"Correct Answers: {metrics['correct_answers']}")
        print(f"Accuracy: {metrics['accuracy']:.2%}")
        print(f"Invalid Responses: {metrics['invalid_responses']}")
        print(f"Errors: {metrics['errors']}")
        print(f"Average Inference Time: {metrics['average_inference_time']:.2f}s")
        print(f"Total Time: {metrics['total_time']:.2f}s")
        print(f"Unique Sections Referenced: {len(metrics['source_sections_used'])}")
        
        return evaluation_results

def main():
    # Initialize pipeline
    pipeline = EvaluationPipeline(
        questions_csv="bio_data.csv",  # Path to your questions CSV
        pdf_path="biology2e_textbook.pdf",  # Path to your PDF
        rag_dir="./biology_rag",  # Your existing RAG directory
        force_refresh_embeddings=False  # Set to True to rebuild vector store
    )
    
    # Run evaluation
    # You can limit the number of questions for testing
    test_results = pipeline.run_evaluation(max_questions=3)

if __name__ == "__main__":
    main()