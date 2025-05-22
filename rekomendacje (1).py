#%%
import math
from collections import defaultdict
from sklearn.model_selection import train_test_split
import statistics
import pandas as pd
import difflib
from sklearn.cluster import KMeans
import numpy as np
import streamlit as st

main_df = pd.read_csv(r"TMDB_movie_dataset_v11.csv")
sdf = main_df[main_df['adult'] != True]
df = sdf[sdf['vote_average'] != 0]

df = df[['title', 'vote_average','vote_count', 'release_date', 'genres', 'keywords']]
def preprocess(x):
    if isinstance(x, list):
        return " ".join(x)
    elif isinstance(x, str):
        return x
    return ""
df.loc[:, 'genres'] = df['genres'].apply(preprocess)
df.loc[:, 'keywords'] = df['keywords'].apply(preprocess)
df.loc[:, 'title'] = df['title'].apply(preprocess)
df.loc[:, 'combined'] = df['genres'] + ", " + df['keywords'] + ", " + df['title']

#%%

def recommend_movies_soft(df, user_prefs, min_vote_avg=4, min_vote_count=500, top_n=5):
    def membership_vote_average(x):
        if x <= 5.0:
            return 0.0
        elif x >= 8.0:
            return 1.0
        else:
            return (x - 5.0) / 3.0

    def membership_vote_count(x):
        if x <= 0:
            return 0.0
        elif x >= 1000:
            return 1.0
        else:
            return x / 1000.0

    def soft_similarity(user_set, movie_set):
        total = 0
        for u_phrase in user_set:
            best_match = max(
                (difflib.SequenceMatcher(None, u_phrase, m_word).ratio() for m_word in movie_set),
                default=0
            )
            total += best_match
        return total / len(user_set) if user_set else 0

    def total_score(row):
        sim = soft_similarity(user_prefs, row['token_set'])
        if sim == 0.0:
            return 0.0
        else:
            score_avg = membership_vote_average(row['vote_average']) if row['vote_average'] >= min_vote_avg else 0
            score_count = membership_vote_count(row['vote_count']) if row['vote_count'] >= min_vote_count else 0
            title_bonus = 0.2 if row.get('title_match', False) else 0
            return 0.5 * sim + 0.3 * score_avg + 0.2 * score_count + title_bonus

    df['user_score'] = df.apply(total_score, axis=1)
    return df.sort_values('user_score', ascending=False).head(top_n)[['title', 'vote_average', 'vote_count', 'genres', 'user_score']]

#%%
class Bayes:
    def fit(self, X, y):
        self.klasy = list(set(y))
        self.srednie = {}
        self.wariancje = {}
        self.prawd = {}
        self.dane_klasy = defaultdict(list)

        for cechy, etykieta in zip(X, y):
            self.dane_klasy[etykieta].append(cechy)

        for klasa in self.klasy:
            cechy_klasy = list(zip(*self.dane_klasy[klasa]))
            self.srednie[klasa] = [statistics.mean(cecha) for cecha in cechy_klasy]
            self.wariancje[klasa] = [statistics.variance(cecha) + 1e-9 for cecha in cechy_klasy]
            self.prawd[klasa] = len(self.dane_klasy[klasa])/len(X)

    def gestosc_gauss(self, srednia, wariancja, x):
        wykladnik = math.exp(-(x-srednia)**2 /(2*wariancja))
        return (1 / math.sqrt(2 * math.pi * wariancja)) * wykladnik

    def predict_proba(self, X):
       wynik = []
       for cechy in X:
           prawdopodobienstwa = {}
           for klasa in self.klasy:
               prawd = self.prawd[klasa]
               likelihood = 1
               for wartosc_cechy, srednia, wariancja in zip(cechy, self.srednie[klasa], self.wariancje[klasa]):
                   likelihood *= self.gestosc_gauss(srednia, wariancja, wartosc_cechy)
               prawdopodobienstwa[klasa] = likelihood * prawd
           wynik.append(prawdopodobienstwa)
       return wynik

    def predict(self, X):
        wynik = self.predict_proba(X)
        return [max(p, key=p.get) for p in wynik]


features = df[['vote_average', 'vote_count']].copy()
kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
means = [features[kmeans.labels_ == i]['vote_average'].mean() for i in range(2)]
liked_cluster = np.argmax(means)
df['liked'] = (kmeans.labels_ == liked_cluster).astype(int)

X = df[['vote_average', 'vote_count']].values.tolist()
y = df['liked'].tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Bayes()
model.fit(X_train, y_train)

probas = model.predict_proba(X)

df['bayes_score'] = [p.get(1, 0) for p in probas]

def recommend_movies_bayes(df_test, X_test, model, top_n=10):
    proby = model.predict_proba(X_test)

    pozytywna_klasa = max(model.klasy)
    prawd_pozytywne = [p.get(pozytywna_klasa, 0) for p in proby]

    df_test = df_test.copy()
    df_test['bayes_score'] = prawd_pozytywne

    return df_test.sort_values('bayes_score', ascending=False).head(top_n)[
        ['title', 'vote_average', 'vote_count', 'genres', 'bayes_score']
    ]
#%%
from sklearn.preprocessing import MinMaxScaler

def decision_module(soft_df, bayes_df, n=10):
    merged = pd.merge(
        soft_df, bayes_df, on='title', suffixes=('_soft', '_bayes')
    )

    # scaler = MinMaxScaler()
    # merged[['user_score', 'bayes_score']] = scaler.fit_transform(merged[['user_score', 'bayes_score']])
    merged['avg_score'] = 0.7 * merged['user_score'] + 0.3 * merged['bayes_score']

    merged = merged.sort_values('avg_score', ascending=False)
    merged = merged.drop_duplicates(subset='title', keep='first')

    return merged.head(n)[
        ['title', 'user_score', 'avg_score']
    ]
#%%


st.title("System rekomendacji filmów")

user_input = st.text_input("Wprowadź swoje zainteresowania (oddzielone przecinkami):", "science fiction, love, star wars")

if st.button("Rekomenduj filmy"):
    user_keywords = [x.strip().lower() for x in user_input.split(",") if x.strip()]
    user_prefs = set(user_keywords)
    #user_prefs = {'war', 'space', 'science fiction', 'love'}

    df_copy = df.copy()
    df_copy['token_set'] = df_copy['combined'].apply(lambda x: set(x.lower().split()))


    def title_bonus(row, user_keywords):
        title = row['title'].lower()
        keywords = row['combined'].lower() if isinstance(row['combined'], str) else ""

        for phrase in user_keywords:
            if phrase in title and phrase not in keywords:
                return True
        return False

    df_copy['title_match'] = df_copy.apply(lambda row: title_bonus(row, user_keywords), axis=1)

    #SoftSet
    recommended_soft = recommend_movies_soft(df_copy.copy(), user_prefs, top_n=len(df_copy))

    # Bayes
    X_bayes = df_copy[['vote_average', 'vote_count']].values.tolist()
    probas = model.predict_proba(X_bayes)
    df_copy['bayes_score'] = [p.get(1, 0) for p in probas]
    recommended_bayes = df_copy[['title', 'vote_average', 'vote_count', 'genres', 'bayes_score']].copy()

    final_recommendations = decision_module(recommended_soft, recommended_bayes, 20)
    st.subheader("Movie recommendations")
    st.dataframe(final_recommendations)