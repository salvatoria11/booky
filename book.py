import streamlit as st
import requests
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import numpy as np
import nltk

# Configuration de la page
st.set_page_config(
    page_title="Booky",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clé API Google Books
API_KEY = st.secrets["GOOGLE_BOOKS_API_KEY"]


# Téléchargements NLTK essentiels
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class NLPProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('french') + stopwords.words('english')+ list(string.punctuation))
    
    def preprocess_text(self, text):
        tokens = nltk.word_tokenize(text.lower())  # Découpage en mots + minuscules
        filtered_tokens = [
            self.lemmatizer.lemmatize(token)  # Lemmatisation
            for token in tokens 
            if token not in self.stop_words and token.isalnum()  # Filtrage des stopwords et ponctuation
        ]
        return ' '.join(filtered_tokens)  # Retourne une chaîne nettoyée

def search_google_books(query, max_results=20):
    base_url = "https://www.googleapis.com/books/v1/volumes"
    params = {
        "q": query, #requete de la recherche d'utilisateur
        "key": API_KEY, #clé api sécurisée
        "maxResults": max_results, #limite de resultats
        "printType": "BOOKS" #filtre uniquement des livre
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        items = response.json().get('items', [])
        
        if not items:
            return []
        
        nlp = NLPProcessor()
        books_data = [] # pour stocker les métadonnées des livres
        corpus = [] # pour stocker le texte prétraité pour l'analyse NLP
        
        for item in items:
            volume = item.get('volumeInfo', {})
            book = {
                "id": item.get('id'),
                "title": volume.get('title', 'Titre inconnu'),
                "authors": ", ".join(volume.get('authors', ['Auteur inconnu'])),
                "description": volume.get('description', 'Pas de description'),
                "thumbnail": volume.get('imageLinks', {}).get('thumbnail', ''),
                "categories": ", ".join(volume.get('categories', [])),
                "full_text": nlp.preprocess_text(
                    f"{volume.get('title', '')} " * 3 +  # Pondération explicite du titre
                    f"{volume.get('description', '')} " +
                    f"{' '.join(volume.get('categories', []))} " *2 +
                    f"{' '.join(volume.get('authors', []))}"
                )# texte combiné pour la recherche
            }
            books_data.append(book)
            corpus.append(book['full_text'])
        
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Vecteur TF-IDF (mots et bigrammes)
        tfidf_matrix = vectorizer.fit_transform(corpus)  # Conversion en matrice
        
        processed_query = nlp.preprocess_text(query)
        query_vector = vectorizer.transform([processed_query])
        cos_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        sorted_indices = np.argsort(cos_similarities)[::-1]  # Tri par pertinence
        results = []
        for idx in sorted_indices:
            book = books_data[idx]
            book['relevance'] = cos_similarities[idx]  # Score en pourcentage
            results.append(book)
        
        return results[:max_results]
        
    except Exception as e:
        st.error(f"Erreur API : {str(e)}")
        return []

def main():
    st.title("Booky votre Moteur de Recherche de Livres")
    
    query = st.text_input("Recherchez un livre :", placeholder="Ex: Harry Potter")
    
    if st.button("Rechercher") and query:
        with st.spinner('Analyse des résultats...'):
            results = search_google_books(query)
        
        if not results:
            st.warning("Aucun résultat trouvé")
            return
            
        st.subheader(f" {len(results)} résultats trouvés")
        
        cols = st.columns(3)
        for idx, book in enumerate(results):
            with cols[idx % 3]:
                st.markdown(
                    f"""
                    <div style="
                        border: 1px solid #ddd;
                        border-radius: 10px;
                        padding: 15px;
                        margin: 10px;
                        height: 650px;
                        overflow: hidden;
                        position: relative;
                        background-color: white;
                        color: black;
                    ">
                        <div style="
                            position: absolute;
                            right: 10px;
                            top: 10px;
                            background: #4CAF50;
                            color: white;
                            padding: 5px 10px;
                            border-radius: 5px;
                            font-size: 0.9em;
                        ">
                            {book['relevance']:.2f}%
                        </div>
                        <div style="text-align: center;">
                            <img src="{book['thumbnail']}" 
                                 style="
                                     height: 200px;
                                     border-radius: 5px;
                                     margin-bottom: 15px;
                                 ">
                        </div>
                        <h3 style="margin: 0 0 10px 0;">{book['title']}</h3>
                        <p style="margin: 0 0 10px 0;">
                             {book['authors']}
                        </p>
                        <div style="
                            max-height: 300px;
                            overflow-y: auto;
                            padding-right: 5px;
                        ">
                            <p>{book['description'][:300]}...</p>
                            <div style="
                                background-color: #f9f9f9;
                                padding: 10px;
                                border-radius: 5px;
                                margin-top: 10px;
                            ">
                                <small>
                                    Catégories: {book['categories']}
                                </small>
                            </div>
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
# Ajouter en haut du code :
import base64

def get_base64(bg_file):
    with open(bg_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_image = "cover2.jpg"  # Assurez-vous que le fichier existe
bg_base64 = get_base64(bg_image)
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/jpg;base64,{bg_base64});
        background-size: cover;
        background-opacity: 0.1;  # Ajustez l'opacité si nécessaire
    }}
    </style>
    """,
    unsafe_allow_html=True
)
if __name__ == "__main__":
    main()