import spacy
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# nltk.download('stopwords')
# nltk.download('punkt_tab')
nlp_spacy = spacy.load("en_core_med7_lg")
show_prints = True
to_csv = False
create_df = True


def remove_stop_words(text, show_prints=show_prints):
    """
    Convert text to lowercase and split to a list of words and remove stopwords
    :return:
    """
    tokens = word_tokenize(text)
    english_stopwords = stopwords.words('english')
    tokens_wo_stopwords = [t for t in tokens if t.isalpha() and t.lower() not in english_stopwords]
    text_wo_stopwords = " ".join(tokens_wo_stopwords)
    if show_prints:
        print("Text without stop words:", text_wo_stopwords)
    return text_wo_stopwords


def json2df(file_path):
    """
    file_path: str -> path to json file
    :return:
    """
    df = pd.read_json(file_path, orient='records')
    return df


if create_df:
    dir_path = "../../Manual-Corpus-Generation/data/MEDMCQA/"
    file_name = "medmcqa_data.json"
    medMC_path = dir_path + file_name
    med_df = json2df(file_path=medMC_path)
    med_df['cleaned_question'] = med_df['question'].apply(lambda q: remove_stop_words(text=q, show_prints=False))
    if to_csv:
        med_df.to_csv('dfs/medMC_df.csv', index=False)
else:
    df_path = "dfs/medMC_df.csv"
    med_df = pd.read_csv(df_path)

tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all")
# tokenizer = AutoTokenizer.from_pretrained("Clinical-AI-Apollo/Medical-NER")
# model = AutoModelForTokenClassification.from_pretrained("Clinical-AI-Apollo/Medical-NER")
# tokenizer = AutoTokenizer.from_pretrained("NeverLearn/Medical-NER-finetuned-ner")
# model = AutoModelForTokenClassification.from_pretrained("NeverLearn/Medical-NER-finetuned-ner")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)
for example in med_df['cleaned_question'].head(10):
    print(f"Cleaned Question: {example}")
    ner_results = nlp(example)
    for res in ner_results:
        print(res)
        entity = res['entity']
        word = res['word']
        score = res['score']
        print(f"Item:{word} -> Recognized entity: {entity} (score={score})")
    print("====================================================================")


def find(text: dict, n=10):
    for text in med_df['cleaned_question'].head(n):
        ner_results = nlp(text)
        prev_word = ""
        prev_word_end = -1
        prev_word_start = -1
        cont = False
        for i, item in enumerate(ner_results):
            word = item['word']
            w_start = item['start']
            w_end = item['end']
            if word.startswith("##") and w_start == prev_word_end:
                cont = True
                prev_word = prev_word.join(word[2:])
                prev_word_end = w_end
            elif not word.startswith("##") and cont:
                cont = False
                words.append(prev_word)
            elif not cont and word.startswith("##"):
                print("Problem.")
                break
            elif not cont:
                prev_word = word
                prev_word_end = w_end
                prev_word_start = w_start

