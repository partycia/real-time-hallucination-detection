import streamlit as st
import utils
import pandas as pd
from time import time
import re

# Streamlit app layout
st.title('Anti-Hallucination Chatbot')

# Text input
user_input = st.text_input("Enter your text:")

if user_input:

    prompt = user_input

    output, sampled_passages = utils.get_output_and_samples(prompt)


    # LLM score
    start = time()
    self_similarity_score = utils.llm_evaluate(output,sampled_passages)
    try:
        self_similarity_score = float(self_similarity_score)
    except:
        self_similarity_score = re.findall(r'\d+', self_similarity_score)
    end = time()

    # Display the output
    st.write("**LLM output:**")
    if float(self_similarity_score) > 0.5:
        st.write(output)
    else:
        st.write("I'm sorry, but I don't have the specific information required to answer your question accurately. ")