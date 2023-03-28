# Description
# This is an gradio app that allows users to input a data line by line, (input1)
# using openai chatgpt,
# it will suggests labels for the data (output1, also input2)
# and label (NER) the data with given labels (output2)

import openai
import json
from typing import List, Tuple, Union
import gradio as gr
import colorsys
import pandas as pd
import re
import io

def query_prompt(openai_key: str, prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 1024, temperature: float = 0.5):
    openai.api_key = openai_key

    model = "gpt-3.5-turbo"
    completions = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=temperature,
    )

    output = completions.choices[0].message.content.strip()

    # model = "text-davinci-003"
    # completions = openai.Completion.create(
    #     engine=model,
    #     prompt=prompt,
    #     max_tokens=1024,
    #     n=1
    # )

    # output = completions.choices[0].text.strip()

    return output

def predict_labels(openai_key: str, data: str, explain_data: str = "", max_tokens: int = 1024, temperature: float = 0.5):
    # prepend '- ' to each line
    data_list_string = "\n".join([f"- {item}" for item in data.splitlines()])

    prompt = (
        f"What is good labels for NER in terms of given items but in general? Please match the given output format.\n"
        "Your output format should be like this (print only labels):\n"
        "- label 1\n"
        "- label 2\n"
        "- ...\n\n"
        f"Items list (hint: {explain_data}):\n{data_list_string}\n\n"
    )

    output = query_prompt(openai_key, prompt, max_tokens=max_tokens, temperature=temperature)

    # remove until line starts with '- '
    output = output[output.find("- "):]

    # remove '- ' from each line
    output = "\n".join([item[2:] for item in output.splitlines()])

    # remove parenthesis and it's content which is added by openai
    output = re.sub(r"\([^)]*\)", "", output)

    return output


def predict_ner(openai_key, data: str, entities: str, explain_data: str = "", max_tokens: int = 1024, temperature: float = 0.5):
    # add id to labels
    # e.g. "B-LOC" -> "0: B-LOC"
    entities_dict = {}
    for i, entity in enumerate(entities.splitlines()):
        entities = entities.replace(entity, f"{i}: {entity}")
        entities_dict[str(i)] = entity

    

    prompt = (
        f"Perform NER on the each item. Group the words if you need.\nFormat the output as (<word>|<entity id>) (<word>|<entity id>).\nExample (IPhone XS): (IPhone|101) (XS|107)\n\n"
        f"Entity id and name\n---\n{entities}\n\n"
        f"Items list (hint: {explain_data})\n---\n{data}\n\n"
    )

    print(prompt)

    output = query_prompt(openai_key, prompt, max_tokens=max_tokens, temperature=temperature)
    # ner_results_by_items = [x[1:] for x in item.split(") ()") for item in output.splitlines()]
    ner_results_by_items = [] 
    for item in output.splitlines():
        # use regex to remove () and split by space
        ner_results_by_items.append(re.findall(r"\((.*?)\)", item))

    print(ner_results_by_items)
    
    # parse
    parsed_ner_results_by_items = []
    items = data.splitlines()
    for i, ner_result in enumerate(ner_results_by_items):
        parsed_ner_result = [items[i]]
        for j, ner in enumerate(ner_result):
            # remove {}
            # ner = ner[1:-1]
            text, entity_id = [item.strip() for item in ner.split("|")]
            entity = entities_dict[entity_id]
            parsed_ner_result.append([text, entity])
        parsed_ner_results_by_items.append(parsed_ner_result)

    return parsed_ner_results_by_items

def predict_ner_format_dataframe(openai_key: str, data: str, labels: str, explain_data: str = "", max_tokens: int = 1024, temperature: float = 0.5):
    global latest_output
    # format to match gradio dataframe output
    ner_data = predict_ner(openai_key, data, labels, explain_data, max_tokens, temperature)
    # ner_data is list of list
    # each list will be a row
    # each element is a cell
    # each cell is a list [text, label] and we will convert it to a "text: label" string format
    # first cell is full text, so skip the converting process for the first cell
    df = pd.DataFrame(ner_data)
    df = df.applymap(lambda x: f"{x[0]}: {x[1]}" if isinstance(x, list) else x)
    # df.columns = ["text", *labels.splitlines()]
    df.columns = ["text", *[f"label_{i}" for i in range(len(df.columns)-1)]]

    xlsx = df_to_excel(df)

    return df, xlsx

def df_to_excel(df):
    df.to_excel("/tmp/tmp.xlsx", index=False)
    return "/tmp/tmp.xlsx"

with gr.Blocks() as demo:
    gr.Markdown("# OpenAI ChatGPT Weak Labeler")
    with gr.Row(variant="panel"):
        with gr.Column(variant="panel"):
            openai_key = gr.Textbox(placeholder="Enter your OpenAI API key here...", label="OpenAI API Key")
            input_data = gr.Textbox(lines=5, placeholder="Enter your data here...", label="Data")
            btn_predict_labels = gr.Button("Predict labels")
            
            with gr.Accordion("Advanced Options", open=False):
                explain_data = gr.Textbox(placeholder="Explain your data", label="Explain Data")
                max_tokens = gr.Slider(1, 4096, 1024, label="Max Tokens")
                temperature = gr.Slider(0.0, 1.0, 0.5, label="Temperature")
        
        with gr.Column():
            output_labels = gr.Textbox(lines=5, label="Labels")
            btn_predict_ner = gr.Button("Predict NER", variant="primary")

        # with gr.Column(scale=2, variant="panel"):
            # output_ner = gr.HighlightedText(label="NER Output", elem_id="htext")
    with gr.Row(variant="panel"):
        with gr.Column():
            output_ner = gr.Dataframe(label="NER Output")
            output_ner_file = gr.File()

        # output_ner = gr.HighlightedText(label="NER Output", elem_id="htext")
        # output_ner = gr.HTML()

    btn_predict_labels.click(fn=predict_labels, inputs=[openai_key, input_data, explain_data, max_tokens, temperature], outputs=[output_labels])
    # btn_predict_ner.click(fn=predict_ner, inputs=[input_data, output_labels], outputs=[output_ner])
    # btn_predict_ner.click(fn=predict_ner_format, inputs=[input_data, output_labels], outputs=[output_ner])
    btn_predict_ner.click(fn=predict_ner_format_dataframe, inputs=[openai_key, input_data, output_labels, explain_data, max_tokens, temperature], outputs=[output_ner, output_ner_file])

# break line in css
demo.css = "#htext .no-cat {white-space: pre-wrap;}"
demo.launch()