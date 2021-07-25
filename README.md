# Explore Art with NLP [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/lyyytong/exploring-art-with-nlp)

## Objective

The art world is rich in history and nuances, spanning more than 32,000 years (from the Stone Age) and, within each time period, a wide range of geographical and cultural territories. Having no real knowledge of the arts and a very modest desire to be slightly more cultured, I always felt overwhelmed by the amount of information out there and didn't know where to start.

![](https://www.aci-iac.ca/wp-content/uploads/2020/09/art-books_9_william-kurelek-the-maze-kw.jpg)
*"The Maze" (1953) by William Kurelek. Also how I feel looking at art history.*

I wanted to make an application to (help myself) solve this problem. As I like reading & writing, my approach is as followed:
1. Predict user's sentiment based on a 1- to 2-sentence writing, using a simple Bi-Directional LSTM model.
2. Use predicted sentiment to query corresponding artworks. Emphasis on showing abundant artworks' information - artist, style, genre, year, and other background details.

Assuming that a new piece of information is much likely to stick if anchored by something personally relevant, my main objective here will be a minimal recommendation system for visual artworks, driven by the user's mood/sentiment/thinking of a given moment, to facilitate daily exploration of art and art history.

## Deployment
The final application can be found on Streamlit here: [link](https://share.streamlit.io/lyyytong/exploring-art-with-nlp).

## Dataset
This project was only possible thanks to the [ArtEmis Dataset](https://github.com/optas/artemis). It is, according to the authors, **"a large-scale dataset aimed at providing a detailed understanding of the interplay between visual content, its emotional effect, and explanations for the latter in language"**.

Each of the 81,446 artworks used for the dataset was viewed & hand-labeled by 6,377 real humans, who also provided text explanations of their label choices. The results look like below.

![](https://www.artemisdataset.org/img/effect_of_grounding_with_emotion.png)

The dataset consists of:
- 439,121 emotion attributions & explanations, with 36,347 distinct words
- 6,377 human annotators, with at least 5 annotators working on each artwork
- 81,446 artworks from 1,119 artists, 27 art-styles and 45 genres, curated from WikiArt

## Disclaimers
- While I'm still using this application every day and it meets my personal needs, there's much to improve to make this a real recommendation system.
- The dataset is unbalanced, heavily skewed toward positive sentiments, with negative sentiments like 'disgust' and 'anger' making up only 5% and 1.5%, respectively. No resampling was done to correct this issue. The application is therefore much better at predicting positive sentiments. It is not suited for tasks that need to address negative emotions, such as art therapy.
- The application scrapes artwork files on-the-go from [WikiArt](https://www.wikiart.org/). Changes to the WikiArt database and links will affect artwork availability. As of July 2021 when I last checked, only about 65,000 artworks (out of the 81,446 used in the dataset) were still available to be scraped.
- Some artworks and text explanations are NSFW and not suitable for children.
