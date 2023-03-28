# Weak Labeling Tool using ChatGPT

![screenshot1](https://github.com/ainbr/chatgpt-weak-labeler-web-ui/raw/master/misc/screenshot1.png)

This is a simple Named Entity Recognition (NER) tool that uses the ChatGPT language model provided by OpenAI. The purpose of this tool is to demonstrate that ChatGPT is good enough to use as an NER model and in some cases can be better than heuristic rule-based models.

Named Entity Recognition is a subtask of information extraction that involves identifying and classifying named entities in unstructured text into predefined categories such as person names, organizations, locations, and others.

## How to use the tool
Install the required packages by running the following command:

1. Clone or download the repository.
2. `$ pip install -r requirements.txt`
3. `$ gradio app.py`
4. Put your OpenAI API Key and have fun 

## How it works
The tool uses the ChatGPT API to predict the named entities in the input data and labels.

The output of the tool is a sequence of tokens with their corresponding predicted label. The tool then shows the output as table.

## Conclusion
In conclusion, this simple NER tool demonstrates that ChatGPT is good enough to use as an NER model and sometimes better than heuristic rule-based models. However, it should be noted that the performance of the model heavily depends on the quality and quantity of the optimizing the prompt.
