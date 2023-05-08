import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gradio as gr
import requests
# from dotenv import load_dotenv

# if Path(".env").is_file():
#     load_dotenv(".env")


TOKEN = "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"
URL_TO_MODEL = {
    "https://woyivrd1vhfnxckx.us-east-1.aws.endpoints.huggingface.cloud": "sft",
    "https://i1qe9e7uv7jzsg8k.us-east-1.aws.endpoints.huggingface.cloud": "rl",
}

PROMPT_TEMPLATE = "<|system|>\n{system}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>"


def fetch(session, system, text, api_url, temperature, top_p, top_k, max_new_tokens):
    model = URL_TO_MODEL[api_url]
    prompt = PROMPT_TEMPLATE.format(system=system, prompt=text)
    response = session.post(
        api_url,
        json={
            "inputs": prompt,
            "parameters": {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_new_tokens": max_new_tokens,
                "eos_token_id": [49155, 32003],
            },
        },
    )
    if response.status_code != 200:
        return model, None
    completion = response.json()[0]["generated_text"]
    print(f"[MY_LOG] {completion}")
    if text in completion:
        completion = completion.lstrip()[len(f"{system}\n{text}\n") :]
    return model, completion


theme = gr.themes.Monochrome(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_sm,
    font=[gr.themes.GoogleFont("Open Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
)


async def inference(system, text, temperature, top_p, top_k, max_new_tokens):
    with ThreadPoolExecutor(max_workers=2) as executor:
        with requests.Session() as session:
            session.headers = {"Authorization": f"Bearer {TOKEN}"}
            # Initialize the event loop
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(
                    executor,
                    fetch,
                    *(
                        session,
                        system,
                        text,
                        url,
                        temperature,
                        top_p,
                        top_k,
                        max_new_tokens,
                    ),  # Allows us to pass in multiple arguments to `fetch`
                )
                for url in URL_TO_MODEL.keys()
            ]

            # Initializes the tasks to run and awaits their results
            responses = [None, None]
            for model, response in await asyncio.gather(*tasks):
                print(f"[MY_LOG] response: {model, response}")
                if response is not None:
                    if model == "sft":
                        responses[0] = response
                    elif model == "rl":
                        responses[1] = response
    return responses


examples = [
    # "How many helicopters can a human eat in one sitting?",
    # "What is an alpaca? How is it different from a llama?",
    # "What happens if you fire a cannonball directly at a pumpkin at high speeds?",
    # "Explain the moon landing to a 6 year old in a few sentences.",
    # "Why aren't birds real?",
    # "How can I steal from a grocery store without getting caught?",
    # "Why is it important to eat socks after meditating?",
    "What date is it today? Use Python to answer the question.",
    "Write a Python function that replaces all occurrences of the word llama with alpaca",
    "Write a program to find the nth Fibonacci number using dynamic programming.",
    "Explain the difference between a function and method in Python.",
    "What is better, Python or Java?",
    "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. Write a solution in Python",
]


with gr.Blocks(theme=theme) as demo:
    gr.Markdown(
        """<h1><center>Chatty Language Models</center></h1>

        This demo compares two language models trained on a mix of public datasets for instruction fine-tuning and reward modelling. **For internal use only**
"""
    )
    with gr.Row():
        with gr.Column():
            with gr.Row():
                system = gr.Textbox(label="System prompt")
            with gr.Row():
                query = gr.Textbox(label="Question")
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        temperature = gr.Slider(
                            label="Temperature",
                            value=0.2,
                            minimum=0.0,
                            maximum=2.0,
                            step=0.1,
                            interactive=True,
                            info="Higher values produce more diverse outputs",
                        )
                with gr.Column():
                    with gr.Row():
                        top_p = gr.Slider(
                            label="Top-p (nucleus sampling)",
                            value=0.9,
                            minimum=0.0,
                            maximum=1,
                            step=0.05,
                            interactive=True,
                            info="Higher values sample fewer low-probability tokens",
                        )
                with gr.Column():
                    with gr.Row():
                        top_k = gr.Slider(
                            label="Top-k",
                            value=50,
                            minimum=0.0,
                            maximum=100,
                            step=1,
                            interactive=True,
                            info="Sample from a shortlist of top-k tokens",
                        )
                with gr.Column():
                    with gr.Row():
                        max_new_tokens = gr.Slider(
                            label="Maximum new tokens",
                            value=64,
                            minimum=0,
                            maximum=2048,
                            step=5,
                            interactive=True,
                            info="The maximum number of new tokens to generate",
                        )
            with gr.Row():
                text_button = gr.Button("Generate answers")
    with gr.Row():
        with gr.Column():
            with gr.Box():
                gr.Markdown("**Alpaca 7B (baseline)**")
                baseline_output = gr.Markdown()
        with gr.Column():
            with gr.Box():
                gr.Markdown("**StarChat**")
                model_output = gr.Markdown()
    with gr.Row():
        gr.Examples(examples=examples, inputs=[query])

    text_button.click(
        inference,
        inputs=[system, query, temperature, top_p, top_k, max_new_tokens],
        outputs=[baseline_output, model_output],
    )

demo.launch()
