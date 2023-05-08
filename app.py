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

theme = gr.themes.Monochrome(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_sm,
    font=[gr.themes.GoogleFont("Open Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
)

# if HF_TOKEN:
#     try:
#         shutil.rmtree("./data/")
#     except:
#         pass

#     repo = Repository(
#         local_dir="./data/", clone_from="trl-lib/star-chat-prompts", use_auth_token=HF_TOKEN, repo_type="dataset"
#     )
#     repo.git_pull()


def save_inputs_and_outputs(inputs, outputs, generate_kwargs):
    with open(os.path.join("data", "prompts.jsonl"), "a") as f:
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
    # model,
    system_message,
    user_message,
    chatbot,
    history,
    temperature=0.5,
    top_p=0.25,
    top_k=50,
    max_new_tokens=512,
    do_save=True,
):
    # Don't return meaningless message when the input is empty
    if not user_message:
        return chatbot, history, user_message, ""

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
        "top_p": top_p,
        "top_k": top_k,
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
        do_sample=True,
        truncate=999,
        seed=42,
        repetition_penalty=1.2,
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

        # if HF_TOKEN and do_save:
        #     try:
        #         print("Pushing prompt and completion to the Hub")
        #         save_inputs_and_outputs(prompt, output, generate_kwargs)
        #     except Exception as e:
        #         print(e)

        yield chat, history, user_message, ""


examples = [
    "What's the capital city of Brunei?",
    "How can I sort a list in Python?",
    "What date is it today? Use Python to answer the question.",
    "What's the meaning of life?",
    "How can I write a Java function to generate the nth Fibonacci number?",
]


# def regenerate(
#     system_message,
#     user_message,
#     chatbot,
#     history,
#     temperature=0.5,
#     top_p=0.25,
#     top_k=50,
#     max_new_tokens=512,
#     do_save=True,
# ):
#     # Do nothing if there's no history
#     if has_no_history(chatbot, history):
#         return (
#             chatbot,
#             history,
#             user_message,
#             "",
#         )

#     chatbot = chatbot[:-1]
#     history = history[:-2]

#     return generate(system_message, user_message, chatbot, history, temperature, top_p, top_k, max_new_tokens, do_save)


def clear_chat():
    return [], []


def process_example(args):
    for [x, y] in generate(args):
        pass
    return [x, y]


title = """<h1 align="center">⭐ Chat with StarCoder Demo 💬</h1>"""
custom_css = """
#banner-image {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 40%;
}

#chat-message .message {
 padding: 15px;
 border-color: #a5b4fc;
 background-color: #eef2ff;
}

#chat-message .message.bot {
 padding: 15px;
 border-color: #e2e8f0;
    background-color: #f8fafc;
}

#system-message {
    min-height: 622px;
}

#system-message textarea {
    min-height: 562px;
}

#chat-message {
    font-size: 14px;
    min-height: 500px;
}

message pending

"""

# css = share_btn_css + custom_css

with gr.Blocks(theme=theme, analytics_enabled=False, css=custom_css) as demo:
    gr.HTML(title)
    gr.Image("StarCoderBanner.png", elem_id="banner-image", show_label=False)
    gr.Markdown(
        """
            This demo showcases an instruction fine-tuned model based on [StarCoder](https://huggingface.co/bigcode/starcoder), a 16B parameter model trained on one trillion tokens sourced from 80+ programming languages, GitHub issues, Git commits, and Jupyter notebooks (all permissively licensed). With an enterprise-friendly license, 8,192 token context length, and fast large-batch inference via [multi-query attention](https://arxiv.org/abs/1911.02150), StarCoder is currently the best open-source choice for code-based applications. For more details, check out our [blog post]().

            ⚠️ **Intended Use**: this app and its [supporting model](https://huggingface.co/HuggingFaceH4/starcoderbase-finetuned-oasst1) are provided as educational tools to explain instruction fine-tuning; not to serve as replacement for human expertise. For more details on the model's limitations in terms of factuality and biases, see the [model card](https://huggingface.co/HuggingFaceH4/starcoderbase-finetuned-oasst1#bias-risks-and-limitations).

            ⚠️ **Data Collection**: by default, we are collecting the prompts entered in this app to further improve and evaluate the model. Do NOT share any personal or sensitive information while using the app! You can opt out of this data collection by removing the checkbox below.
    """
    )

    with gr.Row():
        do_save = gr.Checkbox(
            value=True,
            label="Store data",
            info="You agree to the storage of your prompt and generated text for research and development purposes:",
        )

    with gr.Row():
        with gr.Column(scale=1):
            system_message = gr.Textbox(
                elem_id="system-message",
                placeholder="Below is a conversation between a human user and a helpful AI coding assistant.",
                label="System prompt",
            )

        with gr.Column(scale=2):
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

        with gr.Column(scale=1):
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
                value=384,
                minimum=0,
                maximum=2048,
                step=4,
                interactive=True,
                info="The maximum numbers of new tokens",
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
            top_p,
            top_k,
            max_new_tokens,
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
            top_p,
            top_k,
            max_new_tokens,
            do_save,
        ],
        outputs=[chatbot, history, last_user_message, user_message],
    )

    # regenerate_button.click(
    #     regenerate,
    #     inputs=[
    #         system_message,
    #         last_user_message,
    #         chatbot,
    #         history,
    #         temperature,
    #         top_p,
    #         top_k,
    #         max_new_tokens,
    #         do_save,
    #     ],
    #     outputs=[chatbot, history, last_user_message, user_message],
    # )

    clear_chat_button.click(clear_chat, outputs=[chatbot, history])
    # share_button.click(None, [], [], _js=share_js)

demo.queue(concurrency_count=16).launch(debug=True)
