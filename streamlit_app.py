########## Import librarires ##########
import pickle
import streamlit as st
import pandas as pd
import requests
from io import BytesIO
import re
from bs4 import BeautifulSoup
import tensorflow as tf
from text_processing import lemm_text, stemm_text
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model_metric import F1_score, bi_lstm, labels

########## Load dataframe to be used for emotion-to-artwork query ##########
@st.cache(show_spinner=False)
def load_df(df_path):
    df = pd.read_csv(df_path)
    style_list = df['art_style'].unique()
    artist_list = df['artist'].unique()
    return df, style_list, artist_list

df, style_list, artist_list = load_df('edited_artemis_dataset.csv')

########## Functions to preprocess inputs ##########
max_length = 200
trunc_type = 'pre'
padding_type = 'pre'

# Load tokenizer
@st.cache(show_spinner=False)
def load_tok(tok_path):
    with open(tok_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

tokenizer = load_tok('tokenizer.pickle')

@st.cache(show_spinner=False)
def preprocess_text(texts):
    inputs = texts.lower()
    inputs = lemm_text(inputs)
    inputs = stemm_text(inputs)
    sequence = tokenizer.texts_to_sequences([inputs])
    padded_sequence = pad_sequences(sequence,
                                    maxlen=max_length,
                                    padding=padding_type,
                                    truncating=trunc_type)
    return padded_sequence

########## Functions to scrape & display artwork image & information from WikiArt ##########
base_web = 'https://www.wikiart.org/'

#----------------------------
@st.cache(show_spinner=False)
def query_df(emotion, artists, styles, min_votes):
    if len(artists)!=0 and len(styles)!=0:
        filtered = df[(df['emotion']==emotion)&(df['artist'].isin(artists))&(df['art_style'].isin(styles))]
    elif len(artists)!=0:
        filtered = df[(df['emotion']==emotion)&(df['artist'].isin(artists))]
    elif len(styles)!=0:
        filtered = df[(df['emotion']==emotion)&(df['art_style'].isin(styles))]
    else:
        filtered = df[df['emotion']==emotion]
    
    if filtered.empty:
        message = 'No artworks found. Please widen your artist and/or style search. Getting an artwork outside of your search for now.'
        artists = []
        styles = []
        results = None
    else:
        filtered_count = filtered.groupby('painting')[['utterance']].count().reset_index().rename(columns={'utterance':'count'})
        results = filtered_count[filtered_count['count']>= min_votes]
        if results.empty:
            max_value = filtered_count['count'].max()
            message = f'No artworks found. Please lower the consensus scale. Getting an artwork at max available value ({max_value}) for now.'
            min_votes = max_value
        else:
            message = ''

    return message, filtered, results, min_votes, artists, styles

#----------------------------
def scrape_artwork(emotion, artists, styles, min_votes):
    while True:
        message, filtered, results, min_votes, artists, styles = query_df(emotion, artists, styles, min_votes)
        if "No artworks found" in message:
            st.write(' ')
            st.write(message)
        else:
            rand_painting = results.sample()['painting'].values[0]
            rand_painting = filtered[filtered['painting']==rand_painting]
            style = rand_painting['art_style'].values[0]
            term = rand_painting['term'].values[0]
            artwork_url = f'https://www.wikiart.org/en/{term}'
            artwork_r = requests.get(artwork_url)
            if artwork_r:
                utterance = rand_painting['utterance']
                break
                
    artwork_soup = BeautifulSoup(artwork_r.text, 'html.parser')

    title = artwork_soup.h1.text
    artist_name = artwork_soup.h2.text
    artist_url = base_web  + artwork_soup.h2.a['href']
    image_url = artwork_soup.img['src']
    date_created = artwork_soup.find('span', {'itemprop':'dateCreated'})
    if date_created:
        date_created = date_created.text
    location_created = artwork_soup.find('span', {'itemprop':'locationCreated'})
    if location_created:
        location_created = location_created.text
    genre = artwork_soup.find('a', {'target':'_self', 'href': re.compile('by-genre')}).text.title()
    location = artwork_soup.article.find('li', {'class':'dictionary-values-gallery'})
    if location:
        location = location.span.text
    desc = artwork_soup.find('p', {'itemprop':'description'})
    if desc:
        desc = desc.text

    return utterance, title, artist_name, artist_url, image_url, date_created, location_created, style, genre, location, desc

#----------------------------
@st.cache(show_spinner=False)
def scrape_artist(artist_url):
    artist_r = requests.get(artist_url)
    artist_soup = BeautifulSoup(artist_r.text, 'html.parser')

    artist_full_name = artist_soup.h2.text
    artist_img_url = artist_soup.img['src']
    birthday = artist_soup.find('span', {'itemprop':'birthDate'})
    if birthday:
        birthday = birthday.text
    birthplace = artist_soup.find('span', {'itemprop':'birthPlace'})
    if birthplace:
        birthplace = birthplace.text
    deathday = artist_soup.find('span', {'itemprop':'deathDate'})
    if deathday:
        deathday = deathday.text
    deathplace = artist_soup.find('span', {'itemprop':'deathPlace'})
    if deathplace:
        deathplace = deathplace.text
    nationality = artist_soup.find('span', {'itemprop':'nationality'})
    if nationality:
        nationality = nationality.text
    art_movement = artist_soup.find('a', {'target':'_self', 'href': re.compile('art-movement')}).text.strip()
    bio = artist_soup.find('p', {'itemprop':'description'})
    if bio:
        bio = bio.get_text(separator='\n')

    return artist_full_name, artist_img_url, birthday, birthplace, deathday, deathplace, nationality, art_movement, bio

#----------------------------
@st.cache(show_spinner=False)
def scrape_style(style):
    style = style.lower().replace(' ', '-')
    style_url = f'https://www.wikiart.org/en/artists-by-art-movement/{style}'
    style_r = requests.get(style_url)
    style_soup = BeautifulSoup(style_r.text, 'html.parser')

    style_desc = style_soup.find('p', {'class':'dictionary-description-text'})
    if style_desc:
        style_desc = style_desc.get_text(separator='\n').strip().split('This is a part of the Wikipedia article used under the Creative Commons Attribution')[0].split('See also')[0]
    else:
        style_desc = "No entry for this art style on WikiArt."

    return style_desc

#----------------------------
def show_img(url):
    img_r = requests.get(url)
    img_bytes = BytesIO(img_r.content)
    st.image(img_bytes,
             use_column_width='auto')

#----------------------------
def show_results(emotion, artists, styles, min_votes):
    utterance, title, artist_name, artist_url, image_url, date_created, location_created, style, genre, location, desc = scrape_artwork(emotion, artists, styles, min_votes)
    if date_created:
        st.subheader(f"{title} ({date_created}), {artist_name}")
    else:
        st.subheader(f"{title}, {artist_name}")
    
    st.write(" ")
    show_img(image_url)

    st.write("**Title:**", title)
    st.write("**Artist:**", artist_name)
    if date_created:
        if location_created:
            st.write("**Created:**", date_created, location_created)
        else:
            st.write("**Created:**", date_created)
    st.write("**Style:**", style)
    st.write("**Genre:**", genre)
    if location:
        st.write("**Location:**", location)
    if desc:
        st.write("**Description:**", desc)
    
    with st.beta_expander('🧑‍🎨 See artist'):
        col1, col2 = st.beta_columns(2)
        with col1:
            artist_full_name, artist_img_url, birthday, birthplace, deathday, deathplace, nationality, art_movement, bio = scrape_artist(artist_url)
            show_img(artist_img_url)
        with col2:
            st.write("**Name:**", artist_name)
            if artist_full_name:
                st.write("**Full Name:**", artist_full_name)
            if birthday:
                if birthplace:
                    st.write("**Born:**", birthday, 'in', birthplace)
                else:
                    st.write("**Born:**", birthday)
            if deathday:
                if deathplace:
                    st.write("**Died:**", deathday, 'in', deathplace)
                else:
                    st.write("**Died:**", deathday)
            if nationality:
                st.write("**Nationality:**", nationality)
            st.write("**Art Movement:**", art_movement)
        if bio:
            st.write("**Biography:**")
            bio = bio.replace('. \n', '.\n\n').replace('.\n', '.\n\n').replace(' ,', ',').replace(' .', '.').replace('\n,', ',').replace('\n.', '.')
            bio = bio.split('\n\n')
            for b in bio:
                st.write(b)

    with st.beta_expander('🎨 See style'):
        style_desc = scrape_style(style)
        style_desc = style_desc.replace('. \n', '.\n\n').replace('.\n', '.\n\n').replace(' ,', ',').replace(' .', '.').replace('\n,', ',').replace('\n.', '.')
        style_desc = style_desc.split('\n\n')
        for s in style_desc:
            st.write(s)
    
    with st.beta_expander(f'❓ Why this evokes "{emotion}"'):
        st.write(f'Explanations by real people who viewed the artwork:')
        for u in utterance:
            st.write(">", u)
        st.write(' ')
        st.write(":warning: *Note: These labels & comments are part of the ArtEmis dataset. The dataset consists of 454K such attributions & explanations, provided by 6K human annotators on 80K artworks. The comments therefore are highly subjective, and may not reflect how you personally feel about the artwork.*")

########## Streamlit layout ##########

# Sidebar layout ---------------------
st.sidebar.title('About')
with st.sidebar.beta_expander("🎯 Purpose"):
    st.write("A simple machine learning powered application that matches the user's expressed mood with an artwork, to help explore art & art history in a more personal way.")
with st.sidebar.beta_expander("🛠 How It Works"):
    st.write("It takes in descriptions of a sensation or mental image, predicts among 9 different emotions, and recommends an artwork that *real people* have decided evokes that emotion.")
    st.write("Text-to-sentiment analysis is performed by a simple Bidirectional LSTM machine learning model.")
    st.write("The 80K artworks used here were hand-labeled by 6K humans as part of the ArtEmis dataset, with pictures & artwork details scraped from WikiArt (see credits).")
with st.sidebar.beta_expander("🙏 Credits"):
    "Dataset: ArtEmisDataset.org"
    "Images & Info: WikiArt.org"
    "Support: CoderSchool.vn"
with st.sidebar.beta_expander("📩 Contact"):
    "lyyytong@gmail.com"
st.sidebar.title(' ')
st.sidebar.title('Filters')
with st.sidebar.beta_expander("🍺 Consensus Scale"):
    min_votes = st.slider("Votes per label", min_value=1, max_value=20, value=3)
    st.write("With 3 as default setting, if your predicted sentiment is 'excitement', only artworks with at least 3 people labeling them with 'excitement' will be shown.")
    st.write("The higher you go, the more people have to agree the artwork evokes your predicted sentiment for it to be shown, but the smaller the result pool will be.")
with st.sidebar.beta_expander(("🔎 Advanced Search")):
    artists = st.multiselect('Choose artists', artist_list)
    styles = st.multiselect('Choose styles', style_list)

# Get input data ---------------------
texts = st.text_input("Describe a mental image or sensation 🖼", "sand dunes and salty air, quaint little piglets here and there", help='Please be as visual, descriptive, explit as possible. The model makes better predictions with vivid imageries!')

# Predict & display results -----------
if texts:
    padded_sequence = preprocess_text(texts)
    pred = bi_lstm.predict(padded_sequence)

    # Result & proba to dictionary to be sorted
    result_dict = {}
    for pct, index in zip(pred[0], range(len(pred[0]))):
        result_dict[index] = pct
    
    # Get top 3 highest proba and their labels
    tops = sorted(result_dict, key=result_dict.get, reverse=True)[:3]
    
    top_1 = labels[tops[0]]
    top_1_pct = pred[0][tops[0]]*100
    top_1_text = f"{top_1} ({'%.0f%%'%top_1_pct})"

    top_2 = labels[tops[1]]
    top_2_pct = pred[0][tops[1]]*100
    top_2_text = f"{top_2} ({'%.0f%%'%top_2_pct})"

    top_3 = labels[tops[2]]
    top_3_pct = pred[0][tops[2]]*100
    top_3_text = f"{top_3} ({'%.0f%%'%top_3_pct})"
    
    # Get selection of sentiment to show artwork of
    st.write('That sounds like...')
    button_1 = st.button(top_1_text)
    button_2 = st.button(top_2_text)
    button_3 = st.button(top_3_text)

    st.text("% is how confident the model is of each prediction.")
    st.text("Click button to see an artwork evoking that emotion.")

    if button_1:
        show_results(top_1, artists, styles, min_votes)
    if button_2:
        show_results(top_2, artists, styles, min_votes)
    if button_3:
        show_results(top_3, artists, styles, min_votes)