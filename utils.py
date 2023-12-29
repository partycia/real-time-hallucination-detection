from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pairwise_cos_sim
from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore, SelfCheckNLI
import pandas as pd
import torch 
import spacy

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")

# Load Spacy English model
nlp = spacy.load("en_core_web_sm")
client = OpenAI(api_key="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")

def llm_evaluate(sentences,sampled_passages):
    prompt = f"""You will be provided with a text passage \
                and your task is to rate the consistency of that text to \
                that of the provided context. Your answer must be only \
                a number between 0.0 and 1.0 rounded to the nearest two \
                decimal places where 0.0 represents no consistency and \
                1.0 represents perfect consistency and similarity. \n\n \
                Text passage: {sentences}. \n\n \
                Context: {sampled_passages[0]} \n\n \
                {sampled_passages[1]} \n\n \
                {sampled_passages[2]}."""

    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt}
    ]
    )

    return completion.choices[0].message.content

def generate_3_samples(prompt):
    sampled_passages = []
    for i in range(1,4):
        completion = client.completions.create(
            model="text-davinci-003",
            prompt = prompt,
            max_tokens = 200,
            temperature=0.7
        )
        globals()[f'sample_{i}'] = completion.choices[0].text.lstrip('\n')
        sampled_passages.append(globals()[f'sample_{i}'])
    return sampled_passages

def get_output_and_samples(prompt):
    completion = client.completions.create(
    model="text-davinci-003",
    prompt = prompt,
    max_tokens = 100,
    temperature=0.7,
    )
    output = completion.choices[0].text

    sampled_passages = generate_3_samples(prompt)
    return output, sampled_passages

def get_cos_sim(output,sampled_passages):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentence_embeddings = model.encode(output).reshape(1, -1)
    sample1_embeddings = model.encode(sampled_passages[0]).reshape(1, -1)
    sample2_embeddings = model.encode(sampled_passages[1]).reshape(1, -1)
    sample3_embeddings = model.encode(sampled_passages[2]).reshape(1, -1)
    cos_sim_with_sample1 = pairwise_cos_sim(
    sentence_embeddings, sample1_embeddings
    )
    cos_sim_with_sample2  = pairwise_cos_sim(
    sentence_embeddings, sample2_embeddings
    )
    cos_sim_with_sample3  = pairwise_cos_sim(
    sentence_embeddings, sample3_embeddings
    )
    cos_sim_mean = (cos_sim_with_sample1 + cos_sim_with_sample2 + cos_sim_with_sample3) / 3
    cos_sim_mean = cos_sim_mean.item()
    return round(cos_sim_mean,2)

def get_bertscore(output, sampled_passages):
    # spacy sentence tokenization
    sentences = [sent.text.strip() for sent in nlp(output).sents] 
    selfcheck_bertscore = SelfCheckBERTScore(rescale_with_baseline=True)
    sent_scores_bertscore = selfcheck_bertscore.predict(
        sentences = sentences, # list of sentences
        sampled_passages = sampled_passages, # list of sampled passages
    )
    df = pd.DataFrame({
    'Sentence Number': range(1, len(sent_scores_bertscore) + 1),
    'Hallucination Score': sent_scores_bertscore
    })
    return df

def get_self_check_nli(output, sampled_passages):
    # spacy sentence tokenization
    sentences = [sent.text.strip() for sent in nlp(output).sents] 
    selfcheck_nli = SelfCheckNLI(device=mps_device) # set device to 'cuda' if GPU is available
    sent_scores_nli = selfcheck_nli.predict(
        sentences = sentences, # list of sentences
        sampled_passages = sampled_passages, # list of sampled passages
    )
    df = pd.DataFrame({
    'Sentence Number': range(1, len(sent_scores_nli) + 1),
    'Probability of Contradiction': sent_scores_nli
    })
    return df