import os
import random
from itertools import zip_longest

import gradio as gr
from text_generation import Client, InferenceAPIClient

from share_btn import community_icon_html, loading_icon_html, share_js, share_btn_css

TOKEN = os.environ.get("API_TOKEN", None)

# 13 models
model_names = [
    # "vicuna-13b",
    # "koala-13b",
    "oasst-pythia-12b",
    # "alpaca-13b",
    # "chatglm-6b",
    # "llama-13b",
    # "stablelm-tuned-alpha-7b",
    "bloom",
    "bloomz",
    "flan-t5-xxl",
    "flan-ul2",
    "santacoder",
    "gpt-neox-20b"
]

model_path = [
    # "HuggingFaceH4/stable-vicuna-13b-2904",
    # "TheBloke/koala-13B-HF",
    "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
    # "anon8231489123/gpt4-x-alpaca-13b-native-4bit-128g",
    # "THUDM/chatglm-6b-int4-qe",
    # "decapoda-research/llama-13b-hf",
    # "stabilityai/stablelm-tuned-alpha-7b",
    "bigscience/bloom",
    "bigscience/bloomz",
    "google/flan-t5-xxl",
    "google/flan-ul2",
    "bigcode/santacoder",
    "EleutherAI/gpt-neox-20b"
]

model_endpoints = [
    "https://huggingface.co/HuggingFaceH4/stable-vicuna-13b-2904",
    "https://huggingface.co/TheBloke/koala-13B-HF",
    "https://huggingface.co/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
    "https://huggingface.co/anon8231489123/gpt4-x-alpaca-13b-native-4bit-128g",
    "https://huggingface.co/THUDM/chatglm-6b-int4-qe",
    "https://huggingface.co/decapoda-research/llama-13b-hf",
    "https://huggingface.co/stabilityai/stablelm-tuned-alpha-7b",
    "https://huggingface.co/bigscience/bloom",
    "https://huggingface.co/bigscience/bloomz",
    "https://huggingface.co/google/flan-t5-xxl",
    "https://huggingface.co/google/flan-ul2",
    "https://huggingface.co/bigcode/santacoder",
    "https://huggingface.co/EleutherAI/gpt-neox-20b"
]

model_descriptions = [
    "An open-source chatbot trained by fine-tuning LLaMA on user-shared conversations collected from ShareGPT by LMSYS",  # vicuna
    "A 13B model dialogue model created at Berkeley",  # koala
    "A Pythia 12B fine-tuned on human demonstrations of assistant conversations collected through the https://open-assistant.io/",  # oasst
    "An instruction-following LLaMA model released by Stanford",  # alpaca
    "A 6B open bilingual language model based on General Language Model (GLM) framework",  # chatglm
    "A large language model released by Meta AI",  # llama
    "A 7B model released by Stability AI",  # stablelm
    "BigScience Large Open-science Open-access Multilingual Language Model",  # bloom
    "A family of models capable of following human instructions in dozens of languages zero-shot finetuned from BLOOM & mT5",  # bloomz
    "An enhanced version of T5 that has been finetuned in a mixture of tasks",  # flan-t5
    "An encoder decoder model based on the T5 architecture",  # flan-ul2
    # santacoder
    "A series of 1.1B parameter models trained on the Python, Java, and JavaScript subset of The Stack (v1.1)",
    "A 20 billion parameter autoregressive language model trained on the Pile using the GPT-NeoX library"  # gpt-neox-20b
]

model_to_path_dict = dict([(name, path) for name, path in zip(model_names, model_path)])
model_to_endpoint_dict = dict([(name, endpoint)
                              for name, endpoint in zip(model_names, model_endpoints)])
model_list = sorted(model_names, key=str.casefold)
table_data = [[name, desc, source]
              for name, desc, source in zip(model_names, model_descriptions, model_path)]

CHATBOT_A_DEFAULT = "bloom"
CHATBOT_B_DEFAULT = "bloomz"

theme = gr.themes.Monochrome(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_sm,
    font=[gr.themes.GoogleFont("Open Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
)

openchat_preprompt = (
    "\n<human>: Hi!\n<bot>: My name is Bot, model version is 0.15, part of an open-source kit for "
    "fine-tuning new bots! I was created by Together, LAION, and Ontocord.ai and the open-source "
    "community. I am not human, not evil and not alive, and thus have no thoughts and feelings, "
    "but I am programmed to be helpful, polite, honest, and friendly.\n"
)


def get_client(model: str):
    if model in ["vicuna-13b", "koala-13b", "oasst-pythia-12b", "alpaca-13b", "chatglm-6b", "llama-13b", "stablelm-tuned-alpha-7b"]:
        print("JUST CLIENT")
        return Client(model_to_endpoint_dict[model])
    return InferenceAPIClient(model_to_path_dict[model], token=TOKEN)


def get_usernames(model: str):
    """
    Returns:
        (str, str, str, str): pre-prompt, username, bot name, separator
    """
    if model in ("OpenAssistant/oasst-sft-1-pythia-12b", "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"):
        return "", "<|prompter|>", "<|assistant|>", "<|endoftext|>"
    if model == "togethercomputer/GPT-NeoXT-Chat-Base-20B":
        return openchat_preprompt, "<human>: ", "<bot>: ", "\n"
    return "", "User: ", "Assistant: ", "\n"


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


def get_iterator(model, client, total_inputs, user_name, assistant_name, temperature, max_new_tokens, top_p, repetition_penalty):
    if model in ("OpenAssistant/oasst-sft-1-pythia-12b"):
        iterator = client.generate_stream(
            total_inputs,
            typical_p=0.2,  # fixed
            truncate=1000,
            max_new_tokens=max_new_tokens,
        )
    else:
        iterator = client.generate_stream(
            total_inputs,
            top_p=top_p if top_p < 1.0 else None,
            top_k=50,
            truncate=1000,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            stop_sequences=[user_name.rstrip(), assistant_name.rstrip()],
        )
    return iterator


def has_no_history(chatbot_a, chatbot_b, history_a, history_b):
    return not chatbot_a and not history_a and not chatbot_b and not history_b


def generate(inputs,
             model_a,
             chatbot_a,
             history_a,
             model_b,
             chatbot_b,
             history_b,
             temperature=0.9,
             max_new_tokens=256,
             top_p=0.25,
             repetition_penalty=1.0,
             do_save=True):

    # Don't return meaningless message when the input is empty
    if not inputs:
        yield chatbot_a, chatbot_b, history_a, history_b, inputs, ""

    client_a = get_client(model_a)
    client_b = get_client(model_b)

    preprompt_a, user_name_a, assistant_name_a, sep_a = get_usernames(model_a)
    preprompt_b, user_name_b, assistant_name_b, sep_b = get_usernames(model_b)

    history_a.append(inputs)
    history_b.append(inputs)

    total_inputs_a = get_total_inputs(inputs, chatbot_a, preprompt_a,
                                      user_name_a, assistant_name_a, sep_a)
    total_inputs_b = get_total_inputs(inputs, chatbot_b, preprompt_b,
                                      user_name_b, assistant_name_b, sep_b)

    partial_words_a = ""
    partial_words_b = ""
    iterator_a = get_iterator(model_a, client_a, total_inputs_a, user_name_a,
                              assistant_name_a, temperature, max_new_tokens, top_p, repetition_penalty)
    iterator_b = get_iterator(model_b, client_b, total_inputs_b, user_name_b,
                              assistant_name_b, temperature, max_new_tokens, top_p, repetition_penalty)
    for i, (response_a, response_b) in enumerate(zip_longest(iterator_a, iterator_b, fillvalue="")):
        text_a = "" if not response_a or response_a.token.special else response_a.token.text
        text_b = "" if not response_b or response_b.token.special else response_b.token.text

        partial_words_a = partial_words_a + text_a
        partial_words_b = partial_words_b + text_b
        if partial_words_a.endswith(user_name_a.rstrip()):
            partial_words_a = partial_words_a.rstrip(user_name_a.rstrip())
        if partial_words_b.endswith(user_name_b.rstrip()):
            partial_words_b = partial_words_b.rstrip(user_name_b.rstrip())

        if partial_words_a.endswith(assistant_name_a.rstrip()):
            partial_words_a = partial_words_a.rstrip(assistant_name_a.rstrip())
        if partial_words_b.endswith(assistant_name_b.rstrip()):
            partial_words_b = partial_words_b.rstrip(assistant_name_b.rstrip())

        if i == 0:
            history_a.append(" " + partial_words_a)
            history_b.append(" " + partial_words_b)

        if text_a not in user_name_a:
            history_a[-1] = partial_words_a

        if text_b not in user_name_b:
            history_b[-1] = partial_words_b

        chat_a = [
            (history_a[i].strip(), history_a[i + 1].strip())
            for i in range(0, len(history_a) - 1, 2)
        ]

        chat_b = [
            (history_b[i].strip(), history_b[i + 1].strip())
            for i in range(0, len(history_b) - 1, 2)
        ]

        # return inputs to store the latest input in last_user_message and an empty string to clear out message input textbox
        yield chat_a, chat_b, history_a, history_b, inputs, ""
    # if HF_TOKEN and do_save:
    #     try:
    #         print("Pushing prompt and completion to the Hub")
    #         save_inputs_and_outputs(formatted_message, output, generate_kwargs)
    #     except Exception as e:
    #         print(e)

    # return [output, output]


examples = [
    "A llama is in my lawn. How do I get rid of him?",
    "What's the capital city of Brunei?",
    "How can I sort a list in Python?",
    "What's the meaning of life?",
    "How can I write a Java function to generate the nth Fibonacci number?",
]


def regenerate(inputs, model_a, chatbot_a, history_a, model_b, chatbot_b, history_b, temperature, max_new_tokens, top_p, repetition_penalty, do_save):
    # Do nothing if there's no history
    if has_no_history(chatbot_a, chatbot_b, history_a, history_b):
        print("NOTHING")
        return

    chatbot_a = chatbot_a[:-1]
    chatbot_b = chatbot_b[:-1]
    history_a = history_a[:-2]
    history_b = history_b[:-2]

    for chat_a, chat_b, history_a, history_b, inputs, _ in generate(inputs, model_a, chatbot_a, history_a, model_b, chatbot_b, history_b, temperature, max_new_tokens, top_p, repetition_penalty, do_save):
        yield chat_a, chat_b, history_a, history_b, inputs, ""


def clear_chat():
    return [], [], [], []


def process_example(args):
    print("process_examples")
    print(args)
    for [x, y] in generate(args):
        pass
    return [x, y]


title = """<h1 align="center">🥊 LLM vs LLM 🏆</h1>"""
custom_css = """
#banner-image {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 50%;
}

.model-dropdown {
  color: black !important;
}

#chatbot-a .message {
 padding: 15px;
 border-color: #a5b4fc;
 background-color: #eef2ff;
}

#chatbot-b .message {
 padding: 15px;
 border-color: #fdba74;
 background-color: #fff7ed;
}

#chatbot-a .message.bot {
 padding: 15px;
 border-color: #e2e8f0;
    background-color: #f8fafc;
}

#chatbot-b .message.bot {
 padding: 15px;
 border-color: #e2e8f0;
    background-color: #f8fafc;
}

#chatbot-a {
    min-height: 600px;
}

#chatbot-b {
    min-height: 600px;
}

"""

css = share_btn_css + custom_css

with gr.Blocks(theme=theme, analytics_enabled=False, css=css) as demo:
    gr.HTML(title)
    gr.Image("llms.png", elem_id="banner-image", show_label=False)
    gr.Markdown(
        """
            Compare two language language models side-by-side.
            
            ⚠️ **Data Collection**: by default, we are collecting the prompts entered in this app to further improve and evaluate the model. Do not share any personal or sensitive information while using the app! You can opt out of this data collection by removing the checkbox below:
      """
    )

    gr.Dataframe(
        value=table_data,
        headers=["Model", "Description", "Source"],
        row_count=[2, "dynamic"],
        col_count=[3, "fixed"],
        datatype=["str", "str", "str"],
        type="array",
        show_label=False
    )

    with gr.Row():
        with gr.Column():
            with gr.Box():
                model_a = gr.Dropdown(model_list, elem_classes="model-dropdown", label="Model A", value=CHATBOT_A_DEFAULT)
                output_a = gr.Markdown()
                chatbot_a = gr.Chatbot(label="Model A", elem_id="chatbot-a", show_label=False)
        with gr.Column():
            with gr.Box():
                model_b = gr.Dropdown(model_list, elem_classes="model-dropdown", label="Model B", value=CHATBOT_B_DEFAULT)
                output_b = gr.Markdown()
                chatbot_b = gr.Chatbot(label="Model B", elem_id="chatbot-b", show_label=False)

    with gr.Row():
        with gr.Column(scale=3):
            do_save = gr.Checkbox(
                value=True,
                label="Store data",
                info="You agree to the storage of your prompt and generated text for research and development purposes:")
            message = gr.Textbox(placeholder="Enter your message here",
                                 show_label=False, elem_id="q-input")
            with gr.Row():
                send_button = gr.Button("Send", elem_id="send-btn", visible=True)
                regenerate_button = gr.Button("Regenerate", elem_id="send-btn", visible=True)

                clear_chat_button = gr.Button("Clear chat", elem_id="clear-btn", visible=True)

            with gr.Group(elem_id="share-btn-container"):
                community_icon = gr.HTML(community_icon_html, visible=True)
                loading_icon = gr.HTML(loading_icon_html, visible=True)
                share_button = gr.Button("Share to community", elem_id="share-btn", visible=True)
            with gr.Row():
                gr.Examples(
                    examples=examples,
                    inputs=[message],
                    cache_examples=False,
                    fn=process_example,
                    outputs=[output_a, output_b],
                )

        with gr.Column(scale=1):
            temperature = gr.Slider(
                label="Temperature",
                value=0.9,
                minimum=0.0,
                maximum=2.0,
                step=0.1,
                interactive=True,
                info="Higher values produce more diverse outputs",
            )
            max_new_tokens = gr.Slider(
                label="Max new tokens",
                value=256,
                minimum=0,
                maximum=512,
                step=4,
                interactive=True,
                info="The maximum numbers of new tokens",
            )
            top_p = gr.Slider(
                label="Top-p (nucleus sampling)",
                value=0.25,
                minimum=0.0,
                maximum=1,
                step=0.05,
                interactive=True,
                info="Higher values sample more low-probability tokens",
            )
            repetition_penalty = gr.Slider(
                label="Repetition penalty",
                value=1.2,
                minimum=1.0,
                maximum=2.0,
                step=0.05,
                interactive=True,
                info="Penalize repeated tokens",
            )

    history_a = gr.State([])
    history_b = gr.State([])
    # To clear out "message" input textbox and use this to regenerate message
    last_user_message = gr.State("")

    message.submit(generate,
                   inputs=[message,
                           model_a,
                           chatbot_a,
                           history_a,
                           model_b,
                           chatbot_b,
                           history_b,
                           temperature,
                           max_new_tokens,
                           top_p,
                           repetition_penalty,
                           do_save],
                   outputs=[chatbot_a, chatbot_b, history_a, history_b, last_user_message, message])

    send_button.click(generate,
                      inputs=[message,
                              model_a,
                              chatbot_a,
                              history_a,
                              model_b,
                              chatbot_b,
                              history_b,
                              temperature,
                              max_new_tokens,
                              top_p,
                              repetition_penalty,
                              do_save],
                      outputs=[chatbot_a, chatbot_b, history_a, history_b, last_user_message, message])

    regenerate_button.click(regenerate, inputs=[last_user_message,
                                                model_a,
                                                chatbot_a,
                                                history_a,
                                                model_b,
                                                chatbot_b,
                                                history_b,
                                                temperature,
                                                max_new_tokens,
                                                top_p,
                                                repetition_penalty,
                                                do_save],
                            outputs=[chatbot_a, chatbot_b, history_a, history_b, message])

    clear_chat_button.click(clear_chat,
                            outputs=[chatbot_a, chatbot_b, history_a, history_b])

    share_button.click(None, [], [], _js=share_js)

demo.queue(concurrency_count=16).launch(debug=True)
