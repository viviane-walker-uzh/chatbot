# Simple Chatbot

I created this project this autumn 2024 to illustrate that I can carry out a complete chatbot application - from development (programming) including tests to deployment on the HuggingFace Spaces website. With this project, I want to show that I can put what I have learnt from my studies into practice with my curiosity, which should illustrate my solution-oriented way of working.


## How to use

1. Make sure that you installed all required packages by creating a virtual environment with the requirements.txt.
2. Check whether you want to execute the code on CPU (default) or to GPU (you have to adjust the code).
3. Execute the simple_chatbot/demo_chatbot.py file


## Technical Specifications

LLM used: [togethercomputer/RedPajama-INCITE-Chat-3B-v1 from HuggingFace](https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1)

LLM parameters:
- max_new_tokens=50
- do_sample=True
- top_p=0.95
- top_k=100
- temperature=0.8
- num_beams=1






### Credits
The idea for this chatbot and the skills to build it were aquired thanks to the course
_Text Generation with Language Models_ by Dr. phil. Jannis Vamvas during the spring semester 2024 at the University of Zurich.

The code was taken from the Gradio chatbot examples shared on their website.

### Contact
If you have any questions, do not hesitate to contact me :)


