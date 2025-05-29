import os
import pandas as pd
import spacy
import re

nlp = spacy.load("en_core_web_lg")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
VALID_TOKEN_PATTERN = re.compile(r"^[A-Za-z$]+$")
EMOJI_CLEANER = re.compile(r"[^\w\s.,!?'\u4e00-\u9fff]")

'''
def extract_proper_nouns_from_doc(doc):
    return [
        token.text.lower()
        for token in doc
        if (token.pos_ == "PROPN" or token.is_oov) and VALID_TOKEN_PATTERN.match(token.text)
    ]
'''

def extract_proper_nouns_from_doc(doc):
    return [
        token.text.lower()
        for token in doc
        if not token.is_stop and VALID_TOKEN_PATTERN.match(token.text)
    ]


def remove_emojis(text):
    return EMOJI_CLEANER.sub("", text)

input_folder = "comments"
output_folder = "not_just_proper"
#output_folder = "yt_propernouns"
os.makedirs(output_folder, exist_ok=True)

for fname in os.listdir(input_folder):
    if not fname.endswith(".csv"):
        continue
    try: 
        input_path = os.path.join(input_folder, fname)
        stem = fname.replace(".csv", "")
        print(f"[+] Processing {stem}")
        df = pd.read_csv(input_path)
        df['text'] = df['text'].apply(str)
        df['text'] = df['text'].apply(remove_emojis)
        df_new = df[["cid", "text", "time", "time_parsed"]].copy()
        comments = df_new["text"].fillna("").astype(str).apply(lambda x: URL_PATTERN.sub("", x)).tolist()
        docs = nlp.pipe(comments, batch_size=1000)
        df_new["proper_nouns"] = [extract_proper_nouns_from_doc(doc) for doc in docs]
        output_path = os.path.join(output_folder, f"{stem}_propernouns.csv")
        df_new.to_csv(output_path, index=False)
    except: 
        pass
