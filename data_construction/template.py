Question_Generation_Instruction = """Task:
Given a image descriptions and its corresponding category, generate one or two multiple-choice questions
that verifies if the image description is correct. For each category, generate a question for each type. 
For each question, also generate correct answer together based on the given image description. 



Examples:

- Case 1
Prompt: A blue bowl and a pink mug. 
Category: counting, color

Questions:
1. Is there a bowl in the image? 
Choices: [yes, no]
2. Is there a mug in the image?
Choices: [yes, no]
3. How many bowls are in the image?
Choices: [0, 1, 2, 3, 4, 5+]
4. How many mugs are in the image?
Choices: [0, 1, 2, 3, 4, 5+]
5. What color is the bowl?
Choices: [blue, pink, other]
6. What color is the mug?
Choices: [blue, pink, other]

Answers:
1. yes
2. yes
3. 1
4. 1
5. blue
6. pink

- Case 2
Prompt: A brown bear and a white cat, both wearing spacesuits, are playing frisbee on Mars.
Category: counting, object, color, action, scene

Questions:
1. Is there a bear?
Choices: [yes, no]
2. Is there a cat?
Choices: [yes, no]
3. How many bears are in the image?
Choices: [0, 1, 2, 3, 4, 5+]
4. How many cats are in the image?
Choices: [0, 1, 2, 3, 4, 5+]
5. What color is the bear?
Choices: [brown, white, other]
6. What color is the cat?
Choices: [brown, white, other]
7. Does the bear wear a spacesuit?
Choices: [yes, no]
8. Does the bear wear a spacesuit?
Choices: [yes, no]
9. Is the bear playing frisbee?
Choices: [yes, no]
10. Is the cat playing frisbee?
Choices: [yes, no]
11. What are they playing?
Choices: [frisbee, basketball, other]
12. Where are they playing?
Choices: [mars, earth, other]

Answers:
1. yes
2. yes
3. 1
4. 1
5. brown
6. white
7. yes
8. yes
9. yes
10. yes
11. frisbee
12. mars

Given the following prompt and category, please generate questions and answers:
Prompt: {prompt}
Category: {category}
"""

Answer_Generation_Instruction = """
You are a helpful assistant. Your task is to analyze the given image and answer multiple choice questions based on its content.

Requirements:
- Carefully examine the objects, colors, shapes, texture, actions, spatial-relationships, scenes, and people in the image.
- Select the best answer from the given Choices for each question.
- Ensure accuracy in counting, color, scene, object recognition, and action/spatial relationship understanding.
- Answer only with the provided Choices.

Example Input Question Format:
Questions:
1. Is there a woman in the image?  
   Choices: [yes, no]  

2. Is there a man in the image?  
   Choices: [yes, no]  

3. How many people are in the image?  
   Choices: [0, 1, 2, 3, 4, 5+]

4. What action is the woman performing?  
   Choices: [passing, throwing, catching, other]  

5. What action is the man performing?  
   Choices: [receiving, throwing, catching, other]  

6. What object is being passed?  
   Choices: [tennis ball, basketball, football, other]  


Expected Output Format:
Answers:
1. yes" 
2. yes 
3. 2  
4. passing
5. receiving  
6. tennis ball


Please answer each question with the **best choice** from the given options.
Questions:
{questions}

Answers:
"""

Reverse_Caption_Modification_Instruction = """
Given the original text prompt describing the image, identify any parts that inaccurately reflect the image. 
Then,generate a revised text prompt with correct descriptions, making minimal semantic changes. 
Focusing on the counting, color, shape, texture, scene, spatial relationship, non-spatial relationship.
Remember the modified text prompt should have the similar format, number of words as the original text prompt.

Examples:
Original text prompt: Three puppies are playing on the sandy field on a sunny day, with two black ones walking toward a brown one.
Modified text prompt: Four puppies are standing on a sandy field on a sunny day, with three black puppies and one brown puppy facing forward.
Given the image, the original text prompt is not able to accurately describe the image. Therefore, the original text prompt is modified to precisely describe the image.
"""