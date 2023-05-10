import json
import os
import shutil

import gradio as gr
from huggingface_hub import Repository
from text_generation import Client

from dialogues import DialogueTemplate
from share_btn import (community_icon_html, loading_icon_html, share_btn_css,
                       share_js)

HF_TOKEN = os.environ.get("HF_TOKEN", None)
API_TOKEN = os.environ.get("API_TOKEN", None)
API_URL = os.environ.get("API_URL", None)

client = Client(
    API_URL,
    headers={"Authorization": f"Bearer {API_TOKEN}"},
)

repo = None
if HF_TOKEN:
    try:
        shutil.rmtree("./data/")
    except:
        pass

    repo = Repository(
        local_dir="./data/", clone_from="HuggingFaceH4/starchat-prompts", use_auth_token=HF_TOKEN, repo_type="dataset"
    )
    repo.git_pull()


def save_inputs_and_outputs(now, inputs, outputs, generate_kwargs):
    current_hour = now.strftime("%Y-%m-%d_%H")
    file_name = f"prompts_{current_hour}.jsonl"
    
    if repo is not None:
        repo.git_pull(rebase=True)
        with open(os.path.join("data", file_name), "a") as f:
            json.dump({"inputs": inputs, "outputs": outputs, "generate_kwargs": generate_kwargs}, f, ensure_ascii=False)
            f.write("\n")
        repo.push_to_hub()


def get_total_inputs(inputs, chatbot, preprompt, user_name, assistant_name, sep):
    past = []
    for data in chatbot:
        user_data, model_data = data

        if not user_data.startswith(user_name):
            user_data = user_name + user_data
        if not model_data.startswith(sep + assistant_name):
            model_data = sep + assistant_name + model_data

        past.append(user_data + model_data.rstrip() + sep)

    if not inputs.startswith(user_name):
        inputs = user_name + inputs

    total_inputs = preprompt + "".join(past) + inputs + sep + assistant_name.rstrip()

    return total_inputs


def has_no_history(chatbot, history):
    return not chatbot and not history


def generate(
    system_message,
    user_message,
    chatbot,
    history,
    temperature,
    top_k,
    top_p,
    max_new_tokens,
    repetition_penalty,
    do_save=True,
):
    # Don't return meaningless message when the input is empty
    if not user_message:
        print("Empty input")

    history.append(user_message)

    past_messages = []
    for data in chatbot:
        user_data, model_data = data

        past_messages.extend(
            [{"role": "user", "content": user_data}, {"role": "assistant", "content": model_data.rstrip()}]
        )

    if len(past_messages) < 1:
        dialogue_template = DialogueTemplate(
            system=system_message, messages=[{"role": "user", "content": user_message}]
        )
        prompt = dialogue_template.get_inference_prompt()
    else:
        dialogue_template = DialogueTemplate(
            system=system_message, messages=past_messages + [{"role": "user", "content": user_message}]
        )
        prompt = dialogue_template.get_inference_prompt()

    generate_kwargs = {
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }

    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)

    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        truncate=999,
        seed=42,
        stop_sequences=["<|end|>"],
    )

    stream = client.generate_stream(
        prompt,
        **generate_kwargs,
    )

    output = ""
    for idx, response in enumerate(stream):
        if response.token.special:
            continue
        output += response.token.text
        if idx == 0:
            history.append(" " + output)
        else:
            history[-1] = output

        chat = [(history[i].strip(), history[i + 1].strip()) for i in range(0, len(history) - 1, 2)]

        yield chat, history, user_message, ""

    if HF_TOKEN and do_save:
        try:
            now = datetime.datetime.now()
            current_time = now.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{current_time}] Pushing prompt and completion to the Hub")
            save_inputs_and_outputs(now, prompt, output, generate_kwargs)
        except Exception as e:
            print(e)

    return chat, history, user_message, ""


examples = [
    "How can I write a Python function to generate the nth Fibonacci number?",
    "How do I get the current date using shell commands? Explain how it works.",
    "What's the meaning of life?",
    "Write a function in Javascript to reverse words in a given string.",
    "Give the following data {'Name':['Tom', 'Brad', 'Kyle', 'Jerry'], 'Age':[20, 21, 19, 18], 'Height' : [6.1, 5.9, 6.0, 6.1]}. Can you plot one graph with two subplots as columns. The first is a bar graph showing the height of each person. The second is a bargraph showing the age of each person? Draw the graph in seaborn talk mode.",
    "Create a regex to extract dates from logs",
    "How to decode JSON into a typescript object",
    "Write a list into a jsonlines file and save locally",
]


def clear_chat():
    return [], []


def process_example(args):
    for [x, y] in generate(args):
        pass
    return [x, y]


title = """<h1 align="center">⭐ StarChat Playground 💬</h1>"""
custom_css = """
#banner-image {
    display: block;
    margin-left: auto;
    margin-right: auto;
}

#chat-message {
    font-size: 14px;
    min-height: 300px;
}
"""

with gr.Blocks(analytics_enabled=False, css=custom_css) as demo:
    gr.HTML(title)

    with gr.Row():
        with gr.Column():
            gr.Image("thumbnail.png", elem_id="banner-image", show_label=False)
        with gr.Column():
            gr.Markdown(
                """
            💻 This demo showcases an **alpha** version of **[StarChat](https://huggingface.co/HuggingFaceH4/starchat-alpha)**, a variant of **[StarCoderBase](https://huggingface.co/bigcode/starcoderbase)** that was fine-tuned on the [Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k) and [OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst1) datasets to act as a helpful coding assistant.  The base model has 16B parameters and was pretrained on one trillion tokens sourced from 80+ programming languages, GitHub issues, Git commits, and Jupyter notebooks (all permissively licensed).

            📝 For more details, check out our [blog post](https://huggingface.co/blog/starchat-alpha).

            ⚠️ **Intended Use**: this app and its [supporting model](https://huggingface.co/HuggingFaceH4/starchat-alpha) are provided as educational tools to explain large language model fine-tuning; not to serve as replacement for human expertise.

            ⚠️ **Known Failure Modes**: this alpha version of **StarChat** has not been aligned to human preferences with techniques like RLHF, so the model can produce problematic outputs (especially when prompted to do so). Since the base model was pretrained on a large corpus of code, it may produce code snippets that are syntactically valid but semantically incorrect.  For example, it may produce code that does not compile or that produces incorrect results.  It may also produce code that is vulnerable to security exploits.  We have observed the model also has a tendency to produce false URLs which should be carefully inspected before clicking. For more details on the model's limitations in terms of factuality and biases, see the [model card](https://huggingface.co/HuggingFaceH4/starchat-alpha#bias-risks-and-limitations).

            ⚠️ **Data Collection**: by default, we are collecting the prompts entered in this app to further improve and evaluate the model. Do **NOT** share any personal or sensitive information while using the app! You can opt out of this data collection by removing the checkbox below.
    """
            )

    with gr.Row():
        do_save = gr.Checkbox(
            value=True,
            label="Store data",
            info="You agree to the storage of your prompt and generated text for research and development purposes:",
        )
    with gr.Accordion(label="System Prompt", open=False, elem_id="parameters-accordion"):
        system_message = gr.Textbox(
            elem_id="system-message",
            placeholder="Below is a conversation between a human user and a helpful AI coding assistant.",
            show_label=False,
        )
    with gr.Row():
        with gr.Box():
            output = gr.Markdown()
            chatbot = gr.Chatbot(elem_id="chat-message", label="Chat")

    with gr.Row():
        with gr.Column(scale=3):
            user_message = gr.Textbox(placeholder="Enter your message here", show_label=False, elem_id="q-input")
            with gr.Row():
                send_button = gr.Button("Send", elem_id="send-btn", visible=True)

                # regenerate_button = gr.Button("Regenerate", elem_id="send-btn", visible=True)

                clear_chat_button = gr.Button("Clear chat", elem_id="clear-btn", visible=True)

            with gr.Accordion(label="Parameters", open=False, elem_id="parameters-accordion"):
                temperature = gr.Slider(
                    label="Temperature",
                    value=0.2,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    interactive=True,
                    info="Higher values produce more diverse outputs",
                )
                top_k = gr.Slider(
                    label="Top-k",
                    value=50,
                    minimum=0.0,
                    maximum=100,
                    step=1,
                    interactive=True,
                    info="Sample from a shortlist of top-k tokens",
                )
                top_p = gr.Slider(
                    label="Top-p (nucleus sampling)",
                    value=0.95,
                    minimum=0.0,
                    maximum=1,
                    step=0.05,
                    interactive=True,
                    info="Higher values sample more low-probability tokens",
                )
                max_new_tokens = gr.Slider(
                    label="Max new tokens",
                    value=512,
                    minimum=0,
                    maximum=512,
                    step=4,
                    interactive=True,
                    info="The maximum numbers of new tokens",
                )
                repetition_penalty = gr.Slider(
                    label="Repetition Penalty",
                    value=1.2,
                    minimum=0.0,
                    maximum=10,
                    step=0.1,
                    interactive=True,
                    info="The parameter for repetition penalty. 1.0 means no penalty.",
                )
            # with gr.Group(elem_id="share-btn-container"):
            #     community_icon = gr.HTML(community_icon_html, visible=True)
            #     loading_icon = gr.HTML(loading_icon_html, visible=True)
            # share_button = gr.Button("Share to community", elem_id="share-btn", visible=True)
            with gr.Row():
                gr.Examples(
                    examples=examples,
                    inputs=[user_message],
                    cache_examples=False,
                    fn=process_example,
                    outputs=[output],
                )

    history = gr.State([])
    # To clear out "message" input textbox and use this to regenerate message
    last_user_message = gr.State("")

    user_message.submit(
        generate,
        inputs=[
            system_message,
            user_message,
            chatbot,
            history,
            temperature,
            top_k,
            top_p,
            max_new_tokens,
            repetition_penalty,
            do_save,
        ],
        outputs=[chatbot, history, last_user_message, user_message],
    )

    send_button.click(
        generate,
        inputs=[
            system_message,
            user_message,
            chatbot,
            history,
            temperature,
            top_k,
            top_p,
            max_new_tokens,
            repetition_penalty,
            do_save,
        ],
        outputs=[chatbot, history, last_user_message, user_message],
    )

    clear_chat_button.click(clear_chat, outputs=[chatbot, history])
    # share_button.click(None, [], [], _js=share_js)

demo.queue(concurrency_count=16).launch(debug=True)
