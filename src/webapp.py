import sys
import os
import streamlit as st
import base64
from transformers import AutoModel, AutoTokenizer
import torch
from torch.nn import functional as F
model_name = 'roberta-base'
from clean_text import text_preprocessing_pipeline
sys.path.insert(0, '../src')
from utils import predict

# background
def set_page_bg(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)
    return


# load labels 
with open(os.getcwd() + '/model/labels.txt', 'r') as f:
    labels = [a for a in f.readlines()]
    f.close()

# load model

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        # self.l2 = torch.nn.Dropout(0.3)
        # self.l1 = torch.nn.Linear(768, 256)
        self.fc = torch.nn.Linear(768,5)

    def forward(self, ids, mask, token_type_ids):
        _, features = self.roberta(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        #features = F.relu(self.l1(features))
        # output_2 = self.l2(output_1)
        output = F.softmax(self.fc(features), dim=1)
        return output

model = BERTClass()
model.load_state_dict(torch.load(os.getcwd() + '/model/model4.bin'))
#model.to('cpu')
tokenizer = AutoTokenizer.from_pretrained('roberta-base')

# set background
set_page_bg(os.getcwd() + '/img/bg.png')

st.title('Genre classifier')

# text input
lyrics = st.text_area("Enter lyrics:", height=500)

# display the name when the submit button is clicked
# .title() is used to get the input text string
if(st.button('Submit')):
    result = text_preprocessing_pipeline(lyrics.title())

    label = predict(result, model, tokenizer)
    genre = label
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<p style="color: black; font-size: 30px;">{"Genre: "}{genre}</p>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<p style="color: black; font-size: 30px;">{"Possible title: "}{12345}</p>', unsafe_allow_html=True)
    st.balloons()

    def highlight_words(text, words):
        morph = pymorphy2.MorphAnalyzer()
        tokens = text.split("\n")  # Разделяем текст по переносам строк
        highlighted_lines = []
        for line in tokens:
            line_tokens = re.findall(r'\b\w+\b|[^\w\s]', line, re.UNICODE)
            highlighted_line = ""
            for token in line_tokens:
                parsed_token = morph.parse(token)[0]
                normal_form = parsed_token.normal_form
                if normal_form in words:
                    highlighted_line += f"<span style='background-color: #ffff99'>{token}</span> "
                else:
                    highlighted_line += f"{token} "
            highlighted_lines.append(highlighted_line.strip())
        return "<br>".join(highlighted_lines)

    css_style = """
        <style>
            body {
                font-size: 16px;
                line-height: 1.3;
            }
        </style>
    """

    st.markdown(css_style, unsafe_allow_html=True)


    # Список слов для выделения
    words_to_highlight_rap = ["nigga", "niggas", "shit", "money", "bitch", "fuck", "love", "man", "know", "ass", "bitches", "niggas", "lil", "baby", "girl"]
    words_to_highlight_metal = ["death", "die", "life", "time", "blood", "end", "never", "eye", "away", "heart", "light", "fuck", "left", "feel", "world"]
    words_to_highlight_rock = ["love", "like", "never", "know", "come", "time", "make", "take", "want", "see", "say", "feel", "think", "heart", "need", "want", "baby"]
    words_to_highlight_pop = ["love", "like", "wanna", "know", "heart", "feel", "way", "cause", "see", "say", "make", "baby", "tell", "give", "girl"]
    words_to_highlight_rb = ["love", "est", "way", "know", "make", "want", "baby", "feel", "need", "feel", "like", "want", "let", "dance", "girl"]
    genre_to_list = {"rap": words_to_highlight_rap, "metal": words_to_highlight_metal, "rock": words_to_highlight_rock,
         "pop": words_to_highlight_pop, "rb": words_to_highlight_rb}

    # Выделение слов в тексте
    words_to_highlight = genre_to_list[genre]
    highlighted_text = highlight_words(lyrics, words_to_highlight)

    # Отображение текста с выделенными словами
    st.markdown(highlighted_text, unsafe_allow_html=True)




