import pandas as pd
import streamlit as st
import altair as alt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS, CountVectorizer
import numpy as np
from collections import Counter
from datasets import load_dataset
import joblib
import re
import requests
from io import StringIO

# ==================================================================================== #
# --- Page config ---
st.set_page_config(
    page_title="Analyse des avis employés",
    page_icon="💼",
    layout="centered",  
)

# --- Custom CSS to expand width ---
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1600px;  
        padding-left: 5rem;
        padding-right: 5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==================================================================================== #
#---Load cleaned dataset----
@st.cache_data
# Load dataset from Hugging Face Datasets Hub
def load_data():
    dataset = load_dataset("jedha0padavan/glassdoor_reviews_processed", split="train")
    # Convert to pandas DataFrame
    df = pd.DataFrame(dataset)
    return df

df = load_data() 

# ==================================================================================== #
#---Load model-----

@st.cache_resource
def load_model(model_name, vectorizer_name):
    try:
        model = joblib.load(f'{model_name}.pkl')
        vectorizer = joblib.load(f'{vectorizer_name}.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.error(f"Erreur : Fichier '{model_name}.pkl' introuvable.")
        st.stop()
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        st.stop()

model_name = "best_model"
vectorizer_name = "tfidf_vectorizer"
model, vectorizer = load_model(model_name, vectorizer_name)

# ==================================================================================== #
# --- Initiate of session_state to avoid changing tabs error---
if "show_demo_analysis" not in st.session_state:
    st.session_state.show_demo_analysis = False

# ====================================================================================== #
#----Creation of 2 tabs---

tab_existing, tab_demo = st.tabs(["Analyse existante", "Analyse entreprise inconnue"])

with tab_existing:
    # ==================================================================================== #
    # --- Title ---
    st.title("Vu sur les entreprises à travers des avis des employés")

    #---- intro---
    st.markdown("""
    Bienvenue sur notre tableau de bord interactif basé sur les avis des employés publiés sur Glassdoor.

    **Objectif** : aider les candidats à mieux comprendre l’environnement de travail dans différentes entreprises grâce à une analyse des retours d’employés.

    **Fonctionnalités** :
    - Filtrer par entreprise pour explorer les notes, les avantages et les inconvénients mentionnés.
    - Visualiser les répartitions de notes selon plusieurs critères (management, équilibre vie pro/perso, etc.).
    - Explorer les mots-clés fréquemment utilisés dans les avis positifs et négatifs.
    - Visualiser le Score de sentiment

    *Ce tableau de bord est basé sur un échantillon de données nettoyées et préparées à des fins d’analyse NLP.*  
    """)
    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- Sidebar: company selection ---
    company_list = df['firm'].unique().tolist()
    selected_company = st.sidebar.selectbox("Choisissez une entreprise :", company_list)

    # --- Filtering by firm ---
    company_df = df[df['firm'] == selected_company]


    # ==================================================================================== #
    #----Horizontal barchart with average notes per category----

    # Dict with french names of columns
    nom_colonnes = {
        'overall_rating': "Note globale",
        'work_life_balance': "Équilibre vie pro/perso",
        'culture_values': "Culture et valeurs",
        'career_opp': "Opportunités de carrière",
        'comp_benefits': "Rémunération et avantages",
        'senior_mgmt': "Management"
    }

    # --- List of columns with notes ---
    colonnes_notes = [
        'overall_rating',
        'work_life_balance',
        'culture_values',
        'career_opp',
        'comp_benefits',
        'senior_mgmt'
    ]

    # ==================================================================================== #
    st.markdown(f"### Données pour {selected_company}")
    st.markdown("<br>", unsafe_allow_html=True)

    # --- Affichage des moyennes par catégorie (bar chart horizontal) ---

    col1, spacer, col2 = st.columns([2, 0.2, 2])  

    with col1:
        st.markdown("#### Moyenne des notes par catégorie")

        mean_ratings = company_df[colonnes_notes].mean().sort_values()
        mean_ratings.index = [nom_colonnes.get(col, col) for col in mean_ratings.index]

        fig, ax = plt.subplots(figsize=(6, 3))  # небольшой размер
        colors = sns.color_palette("viridis", len(mean_ratings))

        mean_ratings.plot(kind='barh', ax=ax, color=colors)
        ax.set_xlim(0, 5)
        ax.set_title("")
        ax.set_xlabel("")
        ax.tick_params(axis='both', labelsize=10)

        for i, v in enumerate(mean_ratings):
            ax.text(v + 0.05, i, f"{v:.2f}", va='center', fontsize=12)

        st.pyplot(fig, clear_figure=True)

    with col2:
        

        # ==================================================================================== #
        # --- Show global Sentiment score (placeholder) ---
        st.markdown("#### Sentiment global")


        def print_model_results(model,vectorizer, input_data, selected_company):
            # Effectuer la prédiction
            try:
                # On utilise la méthode predict
                vectorized_data = vectorizer.transform(input_data)
                prediction = model.predict(vectorized_data)
                results = pd.Series(prediction).value_counts()

                        # Récupérer le pourcentage d'avis positifs et négatifs
                positive_count = results.get(1, 0)
                negative_count = results.get(0, 0)
                total = positive_count + negative_count

                if total == 0:
                    st.warning("Aucun avis disponible pour cette entreprise.")
                    return

                positive_percent = (positive_count / total) * 100
                negative_percent = (negative_count / total) * 100
                sentiment_score = positive_count / total

                # Affichage des résultats
                st.success(f"Score de sentiment : {sentiment_score:.2f} (proportion d'avis positifs)")

                st.markdown("#### Répartition des sentiments")

                # Ligne "positif"
                st.markdown(f"**👍 Positifs ({positive_percent:.1f}%)**")
                st.markdown(
                    f"""
                    <div style="background-color: #f0f0f0; border-radius: 5px; height: 24px; width: 100%;">
                        <div style="background-color: #a6d9a6; width: {positive_percent}%; height: 100%; border-radius: 5px;"></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Ligne "négatif"
                st.markdown(f"**👎 Négatifs ({negative_percent:.1f}%)**")
                st.markdown(
                    f"""
                    <div style="background-color: #f0f0f0; border-radius: 5px; height: 24px; width: 100%;">
                        <div style="background-color: #e6a8a8; width: {negative_percent}%; height: 100%; border-radius: 5px;"></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.error(f"Une erreur est survenue lors de la prédiction : {e}")

        # Appel de la fonction d'affichage des résultats du modèle
        input_data = company_df['headline_clean'].fillna('') + ' ' + company_df['pros_clean'].fillna('') + ' ' + company_df['cons_clean'].fillna('')

        # Appelez la fonction pour charger le modèle au démarrage de l'application Streamlit
        model_name = "best_model"
        vectorizer_name = "tfidf_vectorizer"
        model,vectorizer = load_model(model_name,vectorizer_name)
        print_model_results(model,vectorizer, input_data, selected_company)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ==================================================================================== #
    # --- WordClouds: Avantages vs Inconvénients ---
    st.subheader(f"Mot-clés les plus fréquents pour {selected_company}")

    col1, spacer, col2 = st.columns([2, 0.2, 2])
    # Creation des fonctions
    # Function to identify most common words of dataframe 
    def most_common_words_identification(df, nb_words=30):
        """
        Fonction calculant et retournant les mots les plus fréquents.
        """
        # Combine all pros texts into a single string
        text = ' '.join(df.astype(str))

        # Split to words and count frequency
        word_counts = Counter(text.split())

        # Nombre of words
        total_word = len(text.split())

        # Checking top-nb_words
        common_words = word_counts.most_common(nb_words)
        return common_words

    # Common word for the whole dataset
    def common_word_list(df, nb_top_words = 100):

        personal_word_tuple = []
        personal_word = ["jp", "morgan", "jpmorgan", "mc", "donald", "na","mcdonald", "con" ]
        for word in personal_word:
            personal_word_tuple.append((word.lower(),1))

        common_word_pros = most_common_words_identification(df["pros_clean"], nb_top_words)
        word_to_keep = ["benefit", "learn", "environment", "opportunity"]
        common_word_pros = [(mot, count) for mot, count in common_word_pros if mot not in word_to_keep]

        common_word_cons = most_common_words_identification(df["cons_clean"], nb_top_words)
        word_to_keep = ["management", "pay"]
        common_word_cons = [(mot, count) for mot, count in common_word_cons if mot not in word_to_keep]

        common_word = common_word_pros+common_word_cons + personal_word_tuple

        return common_word

    # Creation of a wordcloud from a dataframe
    def process_wordcloud(df, custom_stopwords={}):
        """
        Fonction calculant et retournant le nuage de mots.
        """
        stop_words = set(STOPWORDS)
        

        if custom_stopwords:
            if isinstance(custom_stopwords, list):
                stop_words.update(custom_stopwords) # Ajouter les mots à l'ensemble
            else: #Si ce n'est pas une liste, on suppose que c'est déjà un set.
                stop_words.update(custom_stopwords)

        # Generate word cloud
        wordcloud = WordCloud(max_words = 40, width=1000, height=600, background_color='white', stopwords=stop_words).generate(' '.join(df.fillna('').astype(str)))


        return wordcloud

    # Function to display a wordcloud
    def display_wordcloud(wordcloud, main_subject_title):
        """
        Fonction affichant le nuage de mots.
        """
        # Display the word cloud
        fig = plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('')
        st.pyplot(fig, clear_figure=True)


    common_words = common_word_list(df,nb_top_words = 30)
    personnal_stop_words = ["want", "think"]
    custom_stop_word = {selected_company.lower()}
    custom_stop_word.update({word for word,count in common_words})
    custom_stop_word.update({word for word in personnal_stop_words})

    # Creation of two columns
    with col1:
        st.markdown("#### Avis positifs")
        # pros_text = ' '.join(company_df['pros_clean'].dropna().tolist())
        pros_text = company_df['pros_clean'].astype(str)
        if not pros_text.dropna().empty:
            wordcloud = process_wordcloud(pros_text,custom_stop_word)
            display_wordcloud(wordcloud, "Pros")

        else:
            st.write("Aucune donnée disponible.")

    with col2:
        st.markdown("#### Avis négatifs")
        # cons_text = ' '.join(company_df['cons_clean'].dropna().tolist())
        cons_text = company_df['cons_clean'].astype(str)
        if not cons_text.dropna().empty:
            wordcloud = process_wordcloud(cons_text,custom_stop_word)
            display_wordcloud(wordcloud, "Cons")

        else:
            st.write("Aucune donnée disponible.")
    st.markdown("<br><br>", unsafe_allow_html=True)

    # ==================================================================================== #
    # PROS & CONS

    # Vectorize the pros column
    def generate_vectorizer(df, n_gram_range=(1, 2), custom_stop_word={}):
        """
        Fonction pour générer le TF-IDF.
        """
        text = []

        stop_words = set(STOPWORDS) # Utilisez un ensemble pour une recherche plus rapide

        if custom_stop_word:
            if isinstance(custom_stop_word, list):
                stop_words.update(custom_stop_word) # Ajouter les mots à l'ensemble
            else: #Si ce n'est pas une liste, on suppose que c'est déjà un set.
                stop_words.update(custom_stop_word)

        for sentence in df:
            t = ' '.join([word for word in sentence.split() if word not in stop_words])
            text.append(t)

        vectorizer_TFIDF = TfidfVectorizer(max_features=1000, ngram_range=n_gram_range)
        vectorizer_Count = CountVectorizer(max_features=1000, ngram_range=n_gram_range, stop_words='english')
        
        X_Tfidf = vectorizer_TFIDF.fit_transform(text)
        X_Count = vectorizer_Count.fit_transform(text)
        return X_Tfidf, X_Count, vectorizer_TFIDF,vectorizer_Count

    def display_top_topics(vectorizer, model, n_top_words=10):
        """
        Fonction pour afficher les topics les plus fréquents.
        """
        terms = vectorizer.get_feature_names_out()
        for i, comp in enumerate(model.components_):
            terms_in_topic = [terms[j] for j in comp.argsort()[:-(n_top_words+1):-1]]
            st.write(f"Topic {i+1}: {' | '.join(terms_in_topic)}")

    def filtrage_ngrams(words_df, nb_topics=10):
        selected_n_grams = []
        covered_words = set() # Ensemble pour stocker les mots déjà "couverts"

        for index, row in words_df.iterrows():
            current_n_gram = row['term']
            current_n_gram_words = set(current_n_gram.split()) # Divise le n-gramme en mots individuels

            # Vérifie si le n-gramme actuel contient des mots déjà couverts
            is_redundant = False
            for word in current_n_gram_words:
                if word in covered_words:
                    is_redundant = True
                    break
            
            if not is_redundant:
                selected_n_grams.append(row['term']) # Ajoute le n-gramme à notre liste sélectionnée
                # Ajoute tous les mots de ce n-gramme à l'ensemble des mots couverts
                covered_words.update(current_n_gram_words)
            
            # Arrêtez une fois que nous avons suffisamment de n-grammes
            if len(selected_n_grams) >= nb_topics:
                break

        # Convertir la liste de dictionnaires en DataFrame
        # final_top_n_grams = pd.DataFrame(selected_n_grams)
        return selected_n_grams                                           

    def find_ngram_sentences(top_n_grams, corpus):
        """
        Trouve une phrase du corpus où chaque n-gram est utilisé.

        Args:
            top_n_grams (list): Une liste de n-grams (chaînes de caractères).
            corpus (list): Une liste de chaînes de caractères représentant les phrases du corpus.

        Returns:
            dict: Un dictionnaire où les clés sont les n-grams et les valeurs sont
                des listes de phrases où le n-gram est trouvé.
        """
        ngram_occurrences = {}

        for ngram in top_n_grams:
            found_sentences = []
            # On utilise re.escape pour échapper les caractères spéciaux
            # et \b pour les limites de mots afin de ne pas trouver des parties de mots
            # (ex: "mot" ne doit pas matcher "moteur" si on cherche le mot entier)
            # re.IGNORECASE pour être insensible à la casse
            pattern = re.compile(r'\b' + re.escape(ngram) + r'\b', re.IGNORECASE)

            for sentence in corpus:
                if pattern.search(sentence):
                    found_sentences.append(sentence)
                    # Vous pouvez choisir de ne prendre que la première phrase trouvée
                    # break
            
            # Si aucune phrase n'est trouvée, on peut l'indiquer
            if not found_sentences:
                ngram_occurrences[ngram] = ["Aucune phrase trouvée dans le corpus pour ce n-gram."]
            else:
                ngram_occurrences[ngram] = found_sentences
        
        return ngram_occurrences

    col_pros, col_cons = st.columns(2)

    with col_pros:
        # st.markdown("#### Avantages ###")
        #  ********* PROS ********* #
        st.subheader("Avantages principaux")

        # Generate TF-IDF matrix
        X_Tfidf ,X_Count, vectorizer_TFIDF, vectorizer_Count = generate_vectorizer(company_df['pros_clean'].astype(str),
                                                                                n_gram_range=(2, 3),
                                                                                custom_stop_word=custom_stop_word)

        nb_topics = 3
        feature_names = vectorizer_Count.get_feature_names_out()
        word_counts = np.asarray(X_Count.sum(axis=0)).flatten() # Convertir la matrice somme en un array 1D

        # Créer un DataFrame Pandas pour faciliter le tri
        words_df = pd.DataFrame({'term': feature_names, 'count': word_counts})

        # Trier par fréquence et sélectionner le top 3
        top_words = words_df.sort_values(by='count', ascending=False).reset_index()
        top_words = top_words.drop('index', axis=1)
        top_words_filtered = filtrage_ngrams(top_words,nb_topics = 3)
        for index, word in enumerate(top_words_filtered):
            sentence = f"{index+1}. {word.capitalize()}"
            st.write(sentence)

    with col_cons:
        # st.markdown("#### Inconvénients ###")
        #  ********* CONS ********* #
        st.subheader("Inconvénients principaux")
        # Generate TF-IDF matrix
        X_Tfidf ,X_Count, vectorizer_TFIDF, vectorizer_Count = generate_vectorizer(company_df['cons_clean'].astype(str),
                                                                                n_gram_range=(2, 3),
                                                                                custom_stop_word=custom_stop_word)

        nb_topics = 3
        feature_names = vectorizer_Count.get_feature_names_out()
        word_counts = np.asarray(X_Count.sum(axis=0)).flatten() # Convertir la matrice somme en un array 1D

        # Créer un DataFrame Pandas pour faciliter le tri
        words_df = pd.DataFrame({'term': feature_names, 'count': word_counts})

        # 3.3. Trier par fréquence et sélectionner le top 3
        top_words = words_df.sort_values(by='count', ascending=False).reset_index()
        top_words = top_words.drop('index', axis=1)
        top_words_filtered = filtrage_ngrams(top_words,nb_topics = 3)
        for index, word in enumerate(top_words_filtered):
            sentence = f"{index+1}. {word.capitalize()}"
            st.write(sentence)
# ==================================================================================== #

# ---Function to run analyse for new company----

def analyse_custom_reviews(df, model, vectorizer, company_name="Entreprise inconnue"):
    st.markdown(f"### Analyse des avis pour **{company_name}**")

    # --- Préparation du texte complet
    df = df.copy()
    df['full_text'] = df[['headline', 'pros', 'cons']].fillna('').agg(' '.join, axis=1)

    # --- Prédiction du sentiment
    input_data = df['full_text']
    vectorized = vectorizer.transform(input_data)
    preds = model.predict(vectorized)
    df['sentiment'] = preds

    pos_count = (preds == 1).sum()
    neg_count = (preds == 0).sum()
    total = len(preds)

    if total == 0:
        st.warning("Aucun avis valide trouvé.")
        return

    # --- Stopwords personnalisés
    base_stopwords = {"work", "good", "great", "job", "time", "employee", "pay", "hour", "people"}
    company_stopwords = {company_name.lower(), "pizza", "hut"}    
    custom_stop_words = base_stopwords.union(company_stopwords)

    #---Score-----
    st.subheader("Score de sentiment")
    positive_ratio = pos_count / total
    negative_ratio = neg_count / total
    st.markdown(f"**👍 Positifs ({positive_ratio:.1%})**")
    st.progress(positive_ratio)

    st.markdown(f"**👎 Négatifs ({negative_ratio:.1%})**")
    st.progress(negative_ratio)

    # --- Layout with 2 columns
    col1, col2 = st.columns(2)

    with col1:


        # --- Wordcloud pros
        st.subheader("Avis positifs")
        pros_text = df['pros'].fillna('').astype(str)
        if not pros_text.empty:
            wordcloud = process_wordcloud(pros_text, custom_stopwords=custom_stop_words)
            display_wordcloud(wordcloud, "Pros")
        else:
            st.write("Aucun texte positif.")

        # --- Top 3 avantages
        st.subheader("Top 3 avantages")
        X_Tfidf, X_Count, _, vectorizer_Count = generate_vectorizer(pros_text, n_gram_range=(2, 3))
        terms = vectorizer_Count.get_feature_names_out()
        counts = np.asarray(X_Count.sum(axis=0)).flatten()
        top_words = pd.DataFrame({'term': terms, 'count': counts})

        # Filtrer les n-gram contenant des mots interdits
        pattern = '|'.join([re.escape(word) for word in custom_stop_words])
        top_words = top_words[~top_words['term'].str.contains(pattern, case=False)]

        top3_pros = filtrage_ngrams(top_words.sort_values(by="count", ascending=False).reset_index(drop=True), 3)
        for i, phrase in enumerate(top3_pros, 1):
            st.write(f"{i}. {phrase.capitalize()}")

    with col2:
                
        # --- Wordcloud cons
        st.subheader("Avis négatifs")
        cons_text = df['cons'].fillna('').astype(str)
        if not cons_text.empty:
            wordcloud = process_wordcloud(cons_text, custom_stopwords=custom_stop_words)
            display_wordcloud(wordcloud, "Cons")
        else:
            st.write("Aucun texte négatif.")

        # --- Top 3 inconvénients
        st.subheader("Top 3 inconvénients")
        X_Tfidf, X_Count, _, vectorizer_Count = generate_vectorizer(cons_text, n_gram_range=(2, 3))
        terms = vectorizer_Count.get_feature_names_out()
        counts = np.asarray(X_Count.sum(axis=0)).flatten()
        top_words = pd.DataFrame({'term': terms, 'count': counts})

        # Filtrer les n-gram contenant des mots interdits
        top_words = top_words[~top_words['term'].str.contains(pattern, case=False)]

        top3_cons = filtrage_ngrams(top_words.sort_values(by="count", ascending=False).reset_index(drop=True), 3)
        for i, phrase in enumerate(top3_cons, 1):
            st.write(f"{i}. {phrase.capitalize()}")

# === New firm demo tab ====

with tab_demo:
    st.header("Analyse d'une entreprise inconnue")
    st.markdown("""
    Cette section simule le cas d'une entreprise **non connue du modèle**.

    👉 Nous utilisons ici un **fichier d'exemple** basé sur 500 avis réels pour _Pizza-Hut_ — une entreprise absente de l'entraînement.

    📊 Cela permet de démontrer que notre système peut analyser des entreprises nouvelles, jamais vues auparavant.
    """)

    if st.button("Charger un exemple d'avis (Pizza-Hut)"):
        st.session_state.show_demo_analysis = True

    if st.session_state.show_demo_analysis:
        url = "https://fullstackds-projects-bucket.s3.eu-west-3.amazonaws.com/data/new_company_reviews_sample.csv"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                csv_data = StringIO(response.text)
                df_demo = pd.read_csv(csv_data)

                required_cols = {'headline', 'pros', 'cons'}
                if required_cols.issubset(df_demo.columns):
                    analyse_custom_reviews(df_demo, model, vectorizer, company_name="Pizza-Hut (Entreprise inconnue)")
                    with st.expander("Afficher un aperçu des données utilisées"):
                        st.dataframe(df_demo.head(5))
                else:
                    st.error("Le fichier d'exemple ne contient pas les colonnes requises.")
            else:
                st.error("Erreur : le fichier n'a pas pu être téléchargé.")
        except Exception as e:
            st.error(f"Erreur lors du chargement : {e}")