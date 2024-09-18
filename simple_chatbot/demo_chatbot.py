import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread

tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1", torch_dtype=torch.float16)
model = model.to("cpu")

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [29, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def construct_history(message, history):
    history_transformer_format = history + [[message, ""]]
    messages = "".join(["".join(["\n<human>:"+item[0], "\n<bot>:"+item[1]])
                for item in history_transformer_format])
    return messages

def predict(message, history):
    stop = StopOnTokens()
    model_inputs = tokenizer([construct_history(message, history)], return_tensors="pt").to("cpu")
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=40,
        do_sample=True,
        top_p=0.95,
        top_k=100,
        temperature=0.8,
        num_beams=1,
        stopping_criteria=StoppingCriteriaList([stop])
        )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    partial_message = ""
    for new_token in streamer:
        if new_token != '<':
            partial_message += new_token
            yield partial_message

if __name__ == '__main__':
    gr.ChatInterface(fn=predict,
                    examples=["Can you write me the first two sentences of my science fiction book?",
                            "Suggest me a vegetarian recipe for dinner tonight",
                            "Hi ChatBot, how are you today? What did you do?"],
                            title="Simple Chatbot by Viviane Walker",
                            description="""This is a very simple chatbot to demonstrate my understanding of LLMs as well as my skills to build a very simple LLM application.
                            The default model parameters, such as the temperature, are shown in the README within the simple_chatbot repo
                            --> https://github.com/viviane-walker-uzh/chatbot.""").launch()