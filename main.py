from fastapi import FastAPI
import torch
from torch.nn import functional as F
from transformers import pipeline
import numpy as np
from transformers import AutoTokenizer, AutoModel
model_name = 'roberta-base'
from bs4 import BeautifulSoup
import re
import emoji
import string

punct = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*',
         '+', '\\', '•', '~', '@', '£',
         '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥️', '←', '×', '§', '″', '′', 'Â', '█',
         '½', 'à', '…',
         '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥',
         '▓', '—', '‹', '─',
         '▒', '：', '¼', '⊕', '▼', '▪️', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦️', '¤', '▲', 'è', '¸', '¾',
         'Ã', '⋅', '‘', '∞',
         '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤️', 'ï', 'Ø', '¹',
         '≤', '‡', '√', ]

punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
                 "—": "-", "–": "-", "’": "'", "_": "-",
                 "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/',
                 'α': 'alpha', '•': '.', 'à': 'a', '−': '-',
                 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', '!': ' '}


def clean_text(text):
    '''Clean emoji, Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = emoji.demojize(text)
    text = re.sub(r'\:(.*?)\:', '', text)
    text = str(text).lower()  # Making Text Lowercase
    text = re.sub('\[.*?\]', '', text)
    # The next 2 lines remove html text
    text = BeautifulSoup(text, 'lxml').get_text()
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",", "'")
    text = re.sub(r"[^a-zA-Z?.!,¿']+", " ", text)
    return text


def clean_contractions(text, mapping):
    '''Clean contraction using contraction mapping'''
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    for word in mapping.keys():
        if "" + word + "" in text:
            text = text.replace("" + word + "", "" + mapping[word] + "")
    # Remove Punctuations
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    text = re.sub(r"([?.!,¿])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    return text


def clean_special_chars(text, punct, mapping):
    '''Cleans special characters present(if any)'''
    for p in mapping:
        text = text.replace(p, mapping[p])

    for p in punct:
        text = text.replace(p, f' {p} ')

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}
    for s in specials:
        text = text.replace(s, specials[s])

    return text


def remove_space(text):
    '''Removes awkward spaces'''
    # Removes awkward spaces
    text = text.strip()
    text = text.split()
    return " ".join(text)


def text_preprocessing_pipeline(text):
    '''Cleaning and parsing the text.'''
    text = clean_text(text)
    text = clean_special_chars(text, punct, punct_mapping)
    text = remove_space(text)
    return text

MAX_LEN = 100
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 64
EPOCHS = 1
LEARNING_RATE = 1e-5
tokenizer = AutoTokenizer.from_pretrained(model_name)
app = FastAPI()


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        # self.l2 = torch.nn.Dropout(0.3)
        # self.l1 = torch.nn.Linear(768, 256)
        self.fc = torch.nn.Linear(768, 5)

    def forward(self, ids, mask, token_type_ids):
        _, features = self.roberta(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        # features = F.relu(self.l1(features))
        # output_2 = self.l2(output_1)
        output = F.softmax(self.fc(features), dim=1)
        return output


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pipe = pipeline("summarization", model="miscjose/mt5-small-finetuned-genius-music")

model = BERTClass()
model.load_state_dict(torch.load('model4.bin',map_location=device))


def predict_genre(text):
    model.eval()
    inputs = tokenizer.encode_plus(
        text,
        truncation=True,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        return_token_type_ids=True
    )
    ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0).to(device)
    mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0).to(device)
    token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long).unsqueeze(0).to(device)
    outputs = model(ids, mask, token_type_ids)
    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return np.argmax(outputs)


genres = dict({
    0: 'metal',
    1: 'pop',
    2: 'rap',
    3: 'rhythm and blues',
    4: 'rock'
})


def get_summarize(text):
    return pipe(text)[0]['summary_text']


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/predict/{lyrics}")
async def api_predict(lyrics: str):
    lyrics = text_preprocessing_pipeline(lyrics)
    genre = genres[int(predict_genre(lyrics))]
    answer = {"title": f"Hello {get_summarize(lyrics)}",
            "predict": genre}
    return answer
