import torch
import argparse
import os
from openai import OpenAI
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from typing import List, Tuple
from template import Question_Generation_Instruction, Answer_Generation_Instruction, Reverse_Caption_Modification_Instruction
from utils import clean_gpt_response_qa, clean_gpt_response_answer, clean_gpt_response_revised_prompt, encode_image, set_random_seed
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

def init_gpt_api():
    return OpenAI(api_key=os.getenv("API_KEY"))

def call_gpt_api(messages, client, model):
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return completion.choices[0].message.content


def load_models(sd_model: str, clip_model_name: str) -> Tuple[StableDiffusionPipeline, CLIPModel, CLIPProcessor]:
    """Load selected models."""
    
    pipe = DiffusionPipeline.from_pretrained(sd_model).to(device)
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    
    return pipe, clip_model, clip_processor

def generate_images(prompt: str, num_images: int, seed: int, pipe) -> List[str]:
    """Generate images using Stable Diffusion with a fixed seed."""
    generator = torch.Generator(device).manual_seed(seed)
    image_paths = []
    
    for i in range(num_images):
        image = pipe(prompt, generator=generator).images[0]
        image_path = f"generated_image_{i}.png"
        image.save(image_path)
        image_paths.append(image_path)

    return image_paths


def decompose_prompt(client, prompt: str, categories: str, llm_model: str) -> Tuple[List[str], List[str]]:
    """Generate multiple-choice questions and correct answers"""
    
    gpt_prompt = Question_Generation_Instruction.replace("{prompt}", prompt).replace("{category}", categories)
    messages = [{"role": "user", "content": [{"type": "text", "text": gpt_prompt}]}]
    output_text = call_gpt_api(messages, client, llm_model)
    questions, choices, true_answers = clean_gpt_response_qa(output_text)
    return questions, choices, true_answers

def batch_answer_vqa(client, image_paths: List[str], questions: List[str], choices: List[str], vqa_model: str) -> dict:
    """Answer VQA questions in batch for all images."""
    image_answers = {}

    for image in image_paths:
        question_text = ""
        
        for i in range(len(questions)):
            question_text += f"{i}. {questions[i]}\n{choices[i]}\n\n"
        encoded_image = encode_image(image)
        vqa_prompt = Answer_Generation_Instruction.replace("{questions}", question_text)
        messages = [{
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": vqa_prompt},
                            {"type": "image_url", 
                             "image_url": {
                                 "url": f"data:image/jpeg;base64,{encoded_image}",}
                            }
                        ]
                    }] 
        response = call_gpt_api(messages, client, vqa_model)
        answers = clean_gpt_response_answer(response) 
        image_answers[image] = answers
    
    return image_answers

def compute_alignment_score(true_answers: List[str], predicted_answers: List[str]) -> float:
    """Calculate the alignment score between true and predicted answers."""
    correct_count = sum([1 if pred.lower() == true.lower() else 0 for pred, true in zip(predicted_answers, true_answers)])
    
    return correct_count / len(true_answers)

def batch_compute_clip_score(image_paths: List[str], prompt: str, clip_model, clip_processor) -> dict:
    """Compute CLIP similarity scores in batch for all images."""
    images = [Image.open(image_path) for image_path in image_paths]
    inputs = clip_processor(text=[prompt] * len(image_paths), images=images, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        outputs = clip_model(**inputs)
    scores = outputs.logits_per_image.squeeze().cpu().tolist()
    
    if isinstance(scores, float):  # If only one image, wrap it in a list
        scores = [scores]
         
    return {img: score for img, score in zip(image_paths, scores)}

def refine_caption(client, prompt: str, example_img: str, image: str, llm_model) -> str:
    """Revise caption dynamically if alignment is too low."""
    encoded_example_img = encode_image(example_img)
    encoded_img = encode_image(image)
    
    messages = [{
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": Reverse_Caption_Modification_Instruction},
                        {"type": "image_url", 
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_example_img}",}
                        },
                        {"type": "text", "text": f"Now, process the following image and refine its text prompt accordingly.\nOriginal text prompt: {prompt}\nModified text prompt: "},
                        {"type": "image_url", 
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_img}",}
                        } 
                    ]
                }]  
    response = call_gpt_api(messages, client, llm_model)
    return clean_gpt_response_revised_prompt(response)

def select_best_image(client,
                      prompt: str, 
                      images: List[str], 
                      example_image: str,
                      categories: str, 
                      theta: float, 
                      low_threshold: float, 
                      vqa_model: str, 
                      llm_model: str, 
                      clip_model, 
                      clip_processor) -> str:
    
    questions, choices, true_answers = decompose_prompt(client, prompt, categories, llm_model)
    # Step 1: Generate VQA answers in batch
    image_vqa_answers = batch_answer_vqa(client, images, questions, choices, vqa_model)
    # Step 2: Compute alignment scores
    image_scores = {img: compute_alignment_score(true_answers, ans) for img, ans in image_vqa_answers.items()}
    # Step 3: Filter images above theta
    filtered_images = {img: score for img, score in image_scores.items() if score >= theta}
    
    # Step 4: Compute CLIP scores and select the best image
    if filtered_images:
        clip_scores = batch_compute_clip_score(list(filtered_images.keys()), prompt, clip_model, clip_processor)
        best_image = max(clip_scores, key=clip_scores.get)
        return prompt, best_image

    # Step 5: If no images meet theta, check if refinement is needed
    best_score = max(image_scores.values())
    best_image = max(image_scores, key=image_scores.get)
     
    if best_score < low_threshold:
        new_prompt = refine_caption(client, prompt, example_image, best_image, llm_model)
        return new_prompt, best_image
    
    print("no best found, please repeat this process. ")
    return prompt, None
    
def main():
    parser = argparse.ArgumentParser(description="Generate and select the best image based on alignment scores.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation.")
    parser.add_argument("--categories", type=str, required=True, help="Categories of the text prompt")
    parser.add_argument("--num_images", type=int, default=10, help="Number of candidate images to generate.")
    parser.add_argument("--theta", type=float, default=0.8, help="Threshold for alignment score filtering.")
    parser.add_argument("--low_threshold", type=float, default=0.5, help="Threshold to trigger prompt revision.")
    parser.add_argument("--sd_model", type=str, default="runwayml/stable-diffusion-v1-5", help="Stable Diffusion model name.")
    parser.add_argument("--vqa_model", type=str, default="gpt-4o", help="VQA model name.")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch16", help="CLIP model name.")
    parser.add_argument("--llm_model", type=str, default="gpt-4o", help="LLM to decompose text prompt into QA")
    parser.add_argument("--example_image", type=str, help="example image path for vqa to refine prompt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    # Set the random seed
    set_random_seed(args.seed)
    client = init_gpt_api()
    pipe, clip_model, clip_processor = load_models(args.sd_model, args.clip_model)
    generated_images = generate_images(args.prompt, args.num_images, args.seed, pipe)
    prompt, best_image = select_best_image(client, 
                                           args.prompt, 
                                           generated_images, 
                                           args.example_image,
                                           args.categories, 
                                           args.theta, 
                                           args.low_threshold, 
                                           args.vqa_model, 
                                           args.llm_model,
                                           clip_model, 
                                           clip_processor)
    print(prompt, best_image)

if __name__ == "__main__":
    main()
