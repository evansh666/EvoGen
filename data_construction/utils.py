import re
from typing import List, Tuple
import base64
import random
import torch
import numpy as np


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def clean_gpt_response_qa(gpt_response: str) -> Tuple[List[str], List[List[str]], List[str]]:
    """
    Extracts structured questions, choices, and answers from GPT response.

    Args:
        gpt_response (str): Raw text response from GPT containing questions, choices, and answers.

    Returns:
        Tuple[List[str], List[List[str]], List[str]]: 
            - questions (List[str]): Extracted questions.
            - choices (List[List[str]]): List of answer choices for each question.
            - answers (List[str]): Correct answers corresponding to each question.
    """
    questions = []
    choices = []
    answers = []

    # Split response into sections (Questions and Answers)
    question_section = []
    answer_section = []
    collecting_questions = False
    collecting_answers = False

    for line in gpt_response.strip().split("\n"):
        line = line.strip()

        if line.startswith("Questions:"):
            collecting_questions = True
            collecting_answers = False
            continue
        elif line.startswith("Answers:"):
            collecting_questions = False
            collecting_answers = True
            continue

        if collecting_questions:
            question_section.append(line)
        elif collecting_answers:
            answer_section.append(line)

    # Process Questions & Choices
    current_question = None
    current_choices = []

    for line in question_section:
        question_match = re.match(r"\d+\.\s*(.+)", line)  # Matches "1. Question text"
        choices_match = re.match(r"^\s*Choices:\s*(\[[^\]]+\])", line)  # Matches "Choices: [..]"

        if question_match:
            # Store the previous question before starting a new one
            if current_question:
                questions.append(current_question)
                choices.append(current_choices if current_choices else [])
            
            # Extract new question
            current_question = question_match.group(1)
            current_choices = []
        
        elif choices_match:
            current_choices = choices_match.group(1).strip()
            
    # Store the last question
    if current_question:
        questions.append(current_question)
        choices.append(current_choices if current_choices else [])

    # Process Answers
    for line in answer_section:
        answer_match = re.match(r"\d+\.\s*(.+)", line)
        if answer_match:
            answers.append(answer_match.group(1))

    return questions, choices, answers

def clean_gpt_response_answer(gpt_response: str):
    answers = []
    res = gpt_response.strip().split("\n")
    for line in res:
        answer_match = re.match(r"\d+\.\s*(.+)", line)
        if answer_match:
            answers.append(answer_match.group(1))
    return answers 
    
def clean_gpt_response_revised_prompt(gpt_response: str):
    match = re.search(r"Modified text prompt:\s*(.+)", gpt_response, re.DOTALL)
    return match.group(1).strip() if match else ""


def encode_image(image_path: str) -> str:
    """Encodes an image as a base64 string for OpenAI API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")