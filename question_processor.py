import pandas as pd
from typing import List, Dict, Optional, Tuple
import json
import re

class BiologyQuestionEvaluator:
    """
    A class to process and evaluate biology questions for LLM evaluation.
    Handles loading questions from CSV, generating prompts, and evaluating responses.
    """
    
    def __init__(self, csv_path: str):
        """
        Initialize the evaluator with a CSV file path.
        
        Args:
            csv_path (str): Path to CSV file containing biology questions
        """
        self.csv_path = csv_path
        self.column_mapping = {
            'ID': 'id',
            'Question Text': 'question',
            'Choice A': 'option_a',
            'Choice B': 'option_b',
            'Choice C': 'option_c',
            'Choice D': 'option_d',
            'Correct Choice (0-3)': 'correct_answer',
            'Text Sections': 'text_sections',
            'Figures': 'figures',
            'Explanation': 'explanation'
        }
    
    def load_and_validate_data(self) -> pd.DataFrame:
        """
        Load and validate the question CSV data.
        
        Returns:
            pd.DataFrame: Processed and validated dataframe
        
        Raises:
            ValueError: If required columns are missing or data format is invalid
        """
        try:
            # Read CSV with handling of quotes and proper indexing
            df = pd.read_csv(self.csv_path, 
                            quoting=1,  # QUOTE_ALL
                            encoding='utf-8')
            
            # If first column is unnamed or empty, use the second column as ID
            if 'Unnamed: 0' in df.columns:
                # Assuming BIO_CH4_Q1, etc. are in the second column
                df['ID'] = df.iloc[:, 1]
                df = df.drop(columns=['Unnamed: 0'])
            
            # Clean whitespace from column names
            df.columns = df.columns.str.strip()
            
            # Clean string values while preserving non-string types
            for col in df.columns:
                if df[col].dtype == 'object':  # Only clean string columns
                    df[col] = df[col].str.strip()
            
            # Print debug info
            print("Columns found:", list(df.columns))
            print("First few rows of ID column:", df['ID'].head())
            
            # Validate required columns
            required_columns = set(self.column_mapping.keys())
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            # Rename columns to standardized format
            df = df.rename(columns=self.column_mapping)
            
            # Fill any empty IDs with the question number
            if df['id'].isna().any():
                df['id'] = [f'BIO_CH4_Q{i+1}' for i in range(len(df))]
            
            # Convert correct_answer to int and validate
            df['correct_answer'] = df['correct_answer'].astype(int)
            if not df['correct_answer'].apply(lambda x: 0 <= x <= 3).all():
                raise ValueError("Correct answer must be between 0-3")
                
            return df
            
        except Exception as e:
            # Print the first few rows of the CSV for debugging
            print("\nFirst few rows of CSV:")
            print(pd.read_csv(self.csv_path).head())
            raise ValueError(f"Error reading CSV file: {str(e)}")

 

    def process_questions(self) -> List[Dict]:
        """
        Process all questions into a structured format.
        
        Returns:
            List[Dict]: List of processed question dictionaries
        """
        df = self.load_and_validate_data()
        processed_questions = []
        
        for _, row in df.iterrows():
            question_data = {
                'id': row['id'],
                'prompt_content': {
                    'question': row['question'],
                    'options': {
                        'A': row['option_a'],
                        'B': row['option_b'],
                        'C': row['option_c'],
                        'D': row['option_d']
                    }
                },
                'metadata': {
                    'text_sections': row['text_sections'],
                    'figures': row['figures'],
                    'explanation': row['explanation'],
                    'correct_answer': int(row['correct_answer']),
                    'correct_option': ['A', 'B', 'C', 'D'][int(row['correct_answer'])]
                }
            }
            
            # Generate the actual prompt
            question_data['llm_prompt'] = self.generate_llm_prompt(row)
            processed_questions.append(question_data)
            
        return processed_questions

    def generate_llm_prompt(self, question_data: Dict) -> str:
        """
        Generate a prompt suitable for LLM evaluation.
        
        Args:
            question_data: Dictionary containing question data
            
        Returns:
            str: Formatted prompt string
        """
        prompt = (
            f"{question_data['question']}\n\n"
            f"A) {question_data['option_a']}\n"
            f"B) {question_data['option_b']}\n"
            f"C) {question_data['option_c']}\n"
            f"D) {question_data['option_d']}\n\n"
            "Please provide your answer as a single letter (A, B, C, or D). You must answer with one of the four options. If you are not sure of your answer, make an educated guess based off the information provided.\n\n"
        )
        return prompt

    def evaluate_response(self, question_id: str, response: str) -> Dict:
        """
        Evaluate an LLM's response for a specific question.
        
        Args:
            question_id (str): The question ID to evaluate
            response (str): The LLM's response text
            
        Returns:
            Dict: Evaluation results including correctness and metadata
        """
        questions = self.process_questions()
        question = next((q for q in questions if q['id'] == question_id), None)
        
        if not question:
            raise ValueError(f"Question ID {question_id} not found")
        
        # Extract answer (look for single letter response)
        response = response.strip().upper()
        valid_answers = {'A', 'B', 'C', 'D'}
        
        # Find the first valid answer letter in the response
        answer = next((char for char in response if char in valid_answers), None)
        
        if not answer:
            return {
                'question_id': question_id,
                'status': 'invalid_response',
                'correct': False,
                'metadata': question['metadata'],
                'original_response': response
            }
            
        return {
            'question_id': question_id,
            'status': 'valid_response',
            'correct': answer == question['metadata']['correct_option'],
            'given_answer': answer,
            'correct_answer': question['metadata']['correct_option'],
            'metadata': question['metadata'],
            'original_response': response
        }

    def batch_evaluate(self, responses: List[Tuple[str, str]]) -> Dict:
        """
        Evaluate multiple LLM responses and generate performance metrics.
        
        Args:
            responses: List of (question_id, response_text) tuples
            
        Returns:
            Dict: Evaluation results and aggregate metrics
        """
        results = []
        metrics = {
            'total_questions': len(responses),
            'correct_answers': 0,
            'invalid_responses': 0
        }
        
        for question_id, response in responses:
            result = self.evaluate_response(question_id, response)
            results.append(result)
            
            if result['status'] == 'valid_response':
                if result['correct']:
                    metrics['correct_answers'] += 1
            else:
                metrics['invalid_responses'] += 1
        
        metrics['accuracy'] = (metrics['correct_answers'] / metrics['total_questions'] 
                             if metrics['total_questions'] > 0 else 0)
        
        return {
            'individual_results': results,
            'aggregate_metrics': metrics
        }

    def export_results(self, results: Dict, filepath: str):
        """
        Export evaluation results to a JSON file.
        
        Args:
            results (Dict): Results from batch_evaluate
            filepath (str): Output file path
        """
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

def main():
    """Example usage of the BiologyQuestionEvaluator"""
    # Initialize evaluator
    evaluator = BiologyQuestionEvaluator('sample_questions.csv')
    
    # Process questions
    questions = evaluator.process_questions()
    
    # Example of getting a prompt
    if questions:
        print("\nSample Question Prompt:")
        print(questions[0]['llm_prompt'])
        print("\nMetadata:")
        print(json.dumps(questions[0]['metadata'], indent=2))
    
    # Example of evaluating responses
    sample_responses = [
        ('BIO_CH4_Q1', 'The answer is B'),
        ('BIO_CH4_Q2', 'C')
    ]
    
    evaluation_results = evaluator.batch_evaluate(sample_responses)
    print("\nEvaluation Results:")
    print(json.dumps(evaluation_results['aggregate_metrics'], indent=2))

if __name__ == "__main__":
    main()