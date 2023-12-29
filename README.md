# Anti-Hallucination Chatbot

## Overview
The Anti-Hallucination Chatbot is a Streamlit application designed to evaluate the consistency of language model outputs using various techniques like BERTScore and cosine similarity. It integrates OpenAI's API for language model completions and uses Spacy for NLP tasks.

## Features

- **Text Input**: Users can input text to be evaluated by the language model.
- **Consistency Evaluation**: The app evaluates the consistency of the language model's output using self-similarity scores.
- **Display Output**: Based on the consistency score, the app displays the language model's output or a message indicating insufficient information for a reliable answer.

## Functions

- `generate_3_samples`: Generates three sample passages based on the input prompt.
- `get_output_and_samples`: Obtains the language model's output and sample passages.
- `get_cos_sim`: Calculates the cosine similarity between the model's output and sample passages.
- `get_bertscore`: Computes BERTScores for assessing hallucinations in the output.
- `get_self_check_nli`: Uses NLI models to predict the probability of contradiction in the output.
- `llm_evaluate`: Evaluates the consistency of text with given context using a language model.
