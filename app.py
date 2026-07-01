import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, f1_score, precision_recall_curve
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')

# Mathematical formulations as markdown
TFIDF_FORMULATION = r"""
### Term Frequency-Inverse Document Frequency (TF-IDF)

TF-IDF is a numerical statistic that reflects how important a word is to a document in a corpus.

**Term Frequency (TF):**
$$TF(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$$

Where $f_{t,d}$ is the frequency of term $t$ in document $d$.

**Inverse Document Frequency (IDF):**
$$IDF(t, D) = \log\left(\frac{N}{|\{d \in D : t \in d\}|}\right)$$

Where:
- $N$ = total number of documents in corpus $D$
- $|\{d \in D : t \in d\}|$ = number of documents containing term $t$

**TF-IDF Score:**
$$TF\text{-}IDF(t, d, D) = TF(t, d) \times IDF(t, D)$$

**Interpretation:** High TF-IDF indicates a term is frequent in a document but rare across the corpus, making it a good discriminator.
"""

BOW_FORMULATION = r"""
### Bag of Words (BoW)

Bag of Words represents text as a vector of word counts, ignoring grammar and word order.

**Document Vector:**
$$\vec{d} = (c_1, c_2, \ldots, c_n)$$

Where $c_i$ is the count of the $i$-th word in the vocabulary.

**Vocabulary Construction:**
$$V = \{w_1, w_2, \ldots, w_n\}$$

**Document-Term Matrix:**
$$DTM_{i,j} = count(w_j \text{ in } d_i)$$

**Interpretation:** Each document becomes a sparse vector where most entries are zero. Similar documents will have similar vectors.
"""

LOGREG_FORMULATION = r"""
### Logistic Regression

Logistic regression models the probability of a binary outcome.

**Sigmoid Function:**
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Prediction:**
$$P(y=1|\vec{x}) = \sigma(\vec{w}^T \vec{x} + b) = \frac{1}{1 + e^{-(\vec{w}^T \vec{x} + b)}}$$

**Loss Function (Binary Cross-Entropy):**
$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]$$

**Decision Rule:**
$$\hat{y} = \begin{cases} 1 & \text{if } P(y=1|\vec{x}) \geq 0.5 \\ 0 & \text{otherwise} \end{cases}$$
"""

RF_FORMULATION = r"""
### Random Forest

An ensemble of decision trees using bagging and feature randomization.

**Prediction (Classification):**
$$\hat{y} = \text{mode}\{h_1(\vec{x}), h_2(\vec{x}), \ldots, h_B(\vec{x})\}$$

Where $h_b$ is the $b$-th decision tree and $B$ is the number of trees.

**Gini Impurity (Split Criterion):**
$$Gini(S) = 1 - \sum_{i=1}^{C} p_i^2$$

Where $p_i$ is the proportion of class $i$ samples in set $S$.

**Information Gain:**
$$IG(S, A) = Gini(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} Gini(S_v)$$
"""

SVM_FORMULATION = r"""
### Support Vector Machine (Linear)

SVM finds the optimal hyperplane that maximizes the margin between classes.

**Decision Function:**
$$f(\vec{x}) = \vec{w}^T \vec{x} + b$$

**Optimization Objective:**
$$\min_{\vec{w}, b} \frac{1}{2}||\vec{w}||^2 + C\sum_{i=1}^{N}\xi_i$$

Subject to:
$$y_i(\vec{w}^T \vec{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

**Decision Rule:**
$$\hat{y} = \text{sign}(\vec{w}^T \vec{x} + b)$$

Where $C$ is the regularization parameter and $\xi_i$ are slack variables.
"""

NB_FORMULATION = r"""
### Multinomial Naive Bayes

Based on Bayes' theorem with the "naive" assumption of feature independence.

**Bayes' Theorem:**
$$P(c|\vec{x}) = \frac{P(\vec{x}|c) \cdot P(c)}{P(\vec{x})}$$

**Naive Assumption:**
$$P(\vec{x}|c) = \prod_{i=1}^{n} P(x_i|c)$$

**Multinomial Likelihood:**
$$P(x_i|c) = \frac{N_{ci} + \alpha}{N_c + \alpha \cdot |V|}$$

Where:
- $N_{ci}$ = count of feature $i$ in class $c$
- $N_c$ = total count of all features in class $c$
- $\alpha$ = smoothing parameter (Laplace smoothing when $\alpha=1$)
- $|V|$ = vocabulary size

**Prediction:**
$$\hat{c} = \arg\max_c \left[ \log P(c) + \sum_{i=1}^{n} x_i \cdot \log P(x_i|c) \right]$$
"""

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now download NLTK data
download_nltk_data()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1DA1F2;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #657786;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============== Data Loading & Preprocessing ==============
@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the Twitter sentiment dataset."""
    df = pd.read_csv('twitter_training.csv', names=['tweet_id', 'subject', 'sentiment', 'review_text'])
    
    ENGLISH_STOPWORDS_SET = set(stopwords.words('english'))
    
    def remove_stopwords(token_list):
        return [word for word in token_list if word not in ENGLISH_STOPWORDS_SET]
    
    def remove_punctuation(token_list):
        return [word for word in token_list if word not in punctuation]
    
    # Filter to binary classification
    df = df[df["sentiment"].isin(["Positive", "Negative"])]
    df["review_text"] = df["review_text"].astype("string")
    df["cleaned_text"] = df["review_text"].str.lower()
    df = df.dropna(subset=["cleaned_text"])
    df["cleaned_text"] = df["cleaned_text"].str.replace(r'\s+', ' ', regex=True)
    df = df.drop_duplicates(subset=["cleaned_text"])
    
    # Tokenization
    df["tokens"] = df["cleaned_text"].apply(word_tokenize)
    df["tokens"] = df["tokens"].apply(remove_punctuation)
    df["tokens"] = df["tokens"].apply(remove_stopwords)
    
    # N-grams
    df["bigrams"] = df["tokens"].apply(lambda x: list(nltk.ngrams(x, 2)))
    df["trigrams"] = df["tokens"].apply(lambda x: list(nltk.ngrams(x, 3)))
    df["doc_len"] = df["tokens"].apply(len)
    
    return df

# ============== Model Training ==============
@st.cache_resource
def train_models(_df):
    """Train all classification models."""
    models = {}
    
    # Prepare data
    y = (_df['sentiment'] == 'Positive').astype(int)
    
    # TF-IDF Vectorization
    tfidf_vec = TfidfVectorizer(max_features=500, min_df=5, max_df=0.95, stop_words='english', ngram_range=(1, 2))
    X_tfidf = tfidf_vec.fit_transform(_df['cleaned_text'])
    
    # Bag of Words Vectorization
    bow_vec = CountVectorizer(max_features=500, min_df=5, max_df=0.95, stop_words='english', ngram_range=(1, 2))
    X_bow = bow_vec.fit_transform(_df['cleaned_text'])
    
    # Combined features
    X_combined = hstack([X_tfidf, X_bow])
    
    # Train/Test Split
    X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    X_train_bow, X_test_bow, _, _ = train_test_split(X_bow, y, test_size=0.2, random_state=42)
    X_train_comb, X_test_comb, _, _ = train_test_split(X_combined, y, test_size=0.2, random_state=42)
    
    # 1. Logistic Regression
    lr_model = LogisticRegression(max_iter=500, random_state=42, class_weight='balanced')
    lr_model.fit(X_train_tfidf, y_train)
    lr_pred = lr_model.predict(X_test_tfidf)
    models['Logistic Regression'] = {
        'model': lr_model, 'vectorizer': tfidf_vec, 'y_test': y_test, 'y_pred': lr_pred,
        'accuracy': accuracy_score(y_test, lr_pred), 'f1': f1_score(y_test, lr_pred)
    }
    
    # 2. Random Forest (TF-IDF)
    rf_tfidf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    rf_tfidf.fit(X_train_tfidf, y_train)
    rf_tfidf_pred = rf_tfidf.predict(X_test_tfidf)
    models['Random Forest (TF-IDF)'] = {
        'model': rf_tfidf, 'vectorizer': tfidf_vec, 'y_test': y_test, 'y_pred': rf_tfidf_pred,
        'accuracy': accuracy_score(y_test, rf_tfidf_pred), 'f1': f1_score(y_test, rf_tfidf_pred)
    }
    
    # 3. Random Forest (BoW)
    rf_bow = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    rf_bow.fit(X_train_bow, y_train)
    rf_bow_pred = rf_bow.predict(X_test_bow)
    models['Random Forest (BoW)'] = {
        'model': rf_bow, 'vectorizer': bow_vec, 'y_test': y_test, 'y_pred': rf_bow_pred,
        'accuracy': accuracy_score(y_test, rf_bow_pred), 'f1': f1_score(y_test, rf_bow_pred)
    }
    
    # 4. SVM
    svm_model = SVC(kernel='linear', random_state=42, probability=True)
    svm_model.fit(X_train_bow, y_train)
    svm_pred = svm_model.predict(X_test_bow)
    models['SVM (Linear)'] = {
        'model': svm_model, 'vectorizer': bow_vec, 'y_test': y_test, 'y_pred': svm_pred,
        'accuracy': accuracy_score(y_test, svm_pred), 'f1': f1_score(y_test, svm_pred)
    }
    
    # 5. Naive Bayes
    nb_vec = CountVectorizer(stop_words='english', ngram_range=(1, 2))
    X_nb = nb_vec.fit_transform(_df['cleaned_text'])
    X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(X_nb, y, test_size=0.2, random_state=42, stratify=y)
    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(X_train_nb, y_train_nb)
    nb_pred = nb_model.predict(X_test_nb)
    models['Naive Bayes'] = {
        'model': nb_model, 'vectorizer': nb_vec, 'y_test': y_test_nb, 'y_pred': nb_pred,
        'accuracy': accuracy_score(y_test_nb, nb_pred), 'f1': f1_score(y_test_nb, nb_pred)
    }
    
    return models

# ============== Text Preprocessing for Prediction ==============
def preprocess_for_prediction(text):
    """
    Preprocess user input text for better prediction.
    Handles cases like 'reddead' -> 'red dead' by adding spaces between camelCase
    and detecting common patterns.
    """
    import re
    
    # Convert to lowercase
    text = text.lower().strip()
    
    # Add spaces between lowercase and uppercase letters (camelCase)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # Add spaces between letters and numbers
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    
    # Split concatenated words (simple heuristic for common patterns)
    # This handles cases like 'reddead' -> 'red dead'
    common_words = ['red', 'dead', 'game', 'play', 'good', 'bad', 'great', 'awesome', 
                    'terrible', 'worst', 'best', 'love', 'hate', 'like', 'amazing',
                    'horrible', 'excellent', 'poor', 'nice', 'awful', 'fantastic',
                    'redemption', 'xbox', 'playstation', 'nintendo', 'steam']
    
    # Try to split concatenated words
    result = text
    for word in sorted(common_words, key=len, reverse=True):
        # Look for the word at the start followed by other letters
        pattern = rf'({word})([a-z])'
        result = re.sub(pattern, r'\1 \2', result)
        # Look for letters followed by the word
        pattern = rf'([a-z])({word})'
        result = re.sub(pattern, r'\1 \2', result)
    
    # Clean up multiple spaces
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result

def detect_text_column(columns):
    preferred_columns = ['review_text', 'tweet', 'text', 'content', 'message', 'body']
    normalized_columns = {column.lower().strip(): column for column in columns}

    for column in preferred_columns:
        if column in normalized_columns:
            return normalized_columns[column]

    for column in columns:
        normalized = column.lower().strip()
        if 'tweet' in normalized or 'text' in normalized:
            return column

    return None

def build_batch_predictions(input_df, text_column, model_data):
    output_df = input_df.copy()
    raw_text = output_df[text_column].fillna('').astype(str)
    cleaned_text = raw_text.apply(preprocess_for_prediction)
    valid_mask = cleaned_text.str.len() > 0

    output_df['cleaned_text'] = cleaned_text
    output_df['predicted_sentiment'] = ''

    if not valid_mask.any():
        return output_df, valid_mask

    vectorizer = model_data['vectorizer']
    model = model_data['model']
    X_batch = vectorizer.transform(cleaned_text[valid_mask])
    predictions = model.predict(X_batch)
    output_df.loc[valid_mask, 'predicted_sentiment'] = [
        'Positive' if prediction == 1 else 'Negative'
        for prediction in predictions
    ]

    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_batch)
        output_df.loc[valid_mask, 'negative_probability'] = probabilities[:, 0]
        output_df.loc[valid_mask, 'positive_probability'] = probabilities[:, 1]

    return output_df, valid_mask

# ============== Main App ==============
def main():
    # Header
    st.markdown('<p class="main-header">Twitter Sentiment Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyzing sentiment patterns in Twitter data using NLP & Machine Learning</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading and preprocessing data..."):
        df = load_and_preprocess_data()
    
    # Sidebar
    with st.sidebar:
        st.image("https://abs.twimg.com/icons/apple-touch-icon-192x192.png", width=80)
        st.title("Navigation")
        
        st.markdown("---")
        st.subheader("Dataset Info", anchor=False)
        st.metric("Total Tweets", f"{len(df):,}")
        st.metric("Positive", f"{(df['sentiment'] == 'Positive').sum():,}")
        st.metric("Negative", f"{(df['sentiment'] == 'Negative').sum():,}")
        
        st.markdown("---")
        st.subheader("Settings", anchor=False)
        top_n = st.slider("Top N words to display", 10, 30, 20)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        ":material/monitoring: Overview", 
        ":material/text_fields: Word Analysis", 
        ":material/smart_toy: Model Comparison", 
        ":material/functions: Math & ROC", 
        ":material/target: Live Prediction", 
        ":material/table_view: Data Explorer",
        ":material/upload_file: Batch Prediction"
    ])
    
    # ============== Tab 1: Overview ==============
    with tab1:
        st.header("Dataset Overview", anchor=False)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Tweets", f"{len(df):,}")
        with col2:
            st.metric("Positive", f"{(df['sentiment'] == 'Positive').sum():,}", 
                     f"{(df['sentiment'] == 'Positive').mean()*100:.1f}%")
        with col3:
            st.metric("Negative", f"{(df['sentiment'] == 'Negative').sum():,}",
                     f"{(df['sentiment'] == 'Negative').mean()*100:.1f}%")
        with col4:
            st.metric("Avg Words/Tweet", f"{df['doc_len'].mean():.1f}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sentiment Distribution", anchor=False)
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#2ecc71', '#e74c3c']
            sentiment_counts = df['sentiment'].value_counts()
            wedges, texts, autotexts = ax.pie(sentiment_counts.values, labels=sentiment_counts.index, 
                                               autopct='%1.1f%%', colors=colors, explode=(0.05, 0.05),
                                               shadow=True, startangle=90)
            ax.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
            plt.setp(autotexts, size=12, weight="bold", color="white")
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.subheader("Tweet Length Distribution", anchor=False)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(data=df, x='doc_len', hue='sentiment', kde=True, 
                        palette={'Positive': '#2ecc71', 'Negative': '#e74c3c'}, ax=ax)
            ax.set_xlabel('Number of Words', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Distribution of Tweet Lengths by Sentiment', fontsize=14, fontweight='bold')
            st.pyplot(fig)
            plt.close()
        
        # Character length distribution
        st.subheader("Character Count Distribution", anchor=False)
        fig, ax = plt.subplots(figsize=(12, 5))
        df['char_len'] = df['cleaned_text'].str.len()
        sns.histplot(data=df, x='char_len', hue='sentiment', kde=True,
                    palette={'Positive': '#2ecc71', 'Negative': '#e74c3c'}, ax=ax)
        ax.set_xlabel('Number of Characters', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Character Counts by Sentiment', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        plt.close()
    
    # ============== Tab 2: Word Analysis ==============
    with tab2:
        st.header("Word & N-gram Analysis", anchor=False)
        
        # Compute frequency distributions
        pos_tokens = [word for tokens in df[df['sentiment'] == 'Positive']['tokens'] for word in tokens]
        neg_tokens = [word for tokens in df[df['sentiment'] == 'Negative']['tokens'] for word in tokens]
        
        pos_fdist = FreqDist(pos_tokens)
        neg_fdist = FreqDist(neg_tokens)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Words in Positive Reviews", anchor=False)
            pos_df = pd.DataFrame(pos_fdist.most_common(top_n), columns=['Word', 'Frequency'])
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(data=pos_df, x='Frequency', y='Word', palette='Greens_r', ax=ax)
            ax.set_title(f'Top {top_n} Words in Positive Reviews', fontsize=14, fontweight='bold')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.subheader("Top Words in Negative Reviews", anchor=False)
            neg_df = pd.DataFrame(neg_fdist.most_common(top_n), columns=['Word', 'Frequency'])
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(data=neg_df, x='Frequency', y='Word', palette='Reds_r', ax=ax)
            ax.set_title(f'Top {top_n} Words in Negative Reviews', fontsize=14, fontweight='bold')
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        st.subheader("Bigram Analysis", anchor=False)
        
        # Bigram analysis
        pos_bigrams = [bg for bgs in df[df['sentiment'] == 'Positive']['bigrams'] for bg in bgs]
        neg_bigrams = [bg for bgs in df[df['sentiment'] == 'Negative']['bigrams'] for bg in bgs]
        
        pos_bg_fdist = FreqDist(pos_bigrams)
        neg_bg_fdist = FreqDist(neg_bigrams)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top Bigrams - Positive**")
            pos_bg_df = pd.DataFrame([(f"{bg[0]} {bg[1]}", freq) for bg, freq in pos_bg_fdist.most_common(15)], 
                                     columns=['Bigram', 'Frequency'])
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=pos_bg_df, x='Frequency', y='Bigram', palette='Greens_r', ax=ax)
            ax.set_title('Top 15 Bigrams in Positive Reviews', fontsize=12, fontweight='bold')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("**Top Bigrams - Negative**")
            neg_bg_df = pd.DataFrame([(f"{bg[0]} {bg[1]}", freq) for bg, freq in neg_bg_fdist.most_common(15)], 
                                     columns=['Bigram', 'Frequency'])
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=neg_bg_df, x='Frequency', y='Bigram', palette='Reds_r', ax=ax)
            ax.set_title('Top 15 Bigrams in Negative Reviews', fontsize=12, fontweight='bold')
            st.pyplot(fig)
            plt.close()
        
        # TF-IDF Analysis
        st.markdown("---")
        st.subheader("TF-IDF Top Words", anchor=False)
        
        def get_top_tfidf(texts, top_n=10):
            vectorizer = TfidfVectorizer(max_features=200, min_df=5, max_df=0.95, ngram_range=(1, 1), stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
            top_indices = mean_tfidf.argsort()[-top_n:][::-1]
            return pd.DataFrame({'Word': [feature_names[i] for i in top_indices], 
                                'TF-IDF Score': [mean_tfidf[i] for i in top_indices]})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**TF-IDF - Positive Reviews**")
            pos_tfidf = get_top_tfidf(df[df['sentiment'] == 'Positive']['cleaned_text'], 10)
            st.dataframe(pos_tfidf, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**TF-IDF - Negative Reviews**")
            neg_tfidf = get_top_tfidf(df[df['sentiment'] == 'Negative']['cleaned_text'], 10)
            st.dataframe(neg_tfidf, use_container_width=True, hide_index=True)
    
    # ============== Tab 3: Model Comparison ==============
    with tab3:
        st.header("Model Performance Comparison", anchor=False)
        
        with st.spinner("Training models... This may take a moment."):
            models = train_models(df)
        
        # Model metrics
        metrics_df = pd.DataFrame({
            'Model': list(models.keys()),
            'Accuracy': [m['accuracy'] for m in models.values()],
            'F1 Score': [m['f1'] for m in models.values()]
        }).sort_values('Accuracy', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Accuracy Comparison", anchor=False)
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12']
            bars = ax.barh(metrics_df['Model'], metrics_df['Accuracy'], color=colors)
            ax.set_xlabel('Accuracy', fontsize=12)
            min_acc = max(0, metrics_df['Accuracy'].min() - 0.05)
            ax.set_xlim(min_acc, 1.0)
            ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
            for bar, acc in zip(bars, metrics_df['Accuracy']):
                ax.text(acc + 0.005, bar.get_y() + bar.get_height()/2, f'{acc:.4f}', 
                       va='center', fontsize=11, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.subheader("F1 Score Comparison", anchor=False)
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(metrics_df['Model'], metrics_df['F1 Score'], color=colors)
            ax.set_xlabel('F1 Score', fontsize=12)
            min_f1 = max(0, metrics_df['F1 Score'].min() - 0.05)
            ax.set_xlim(min_f1, 1.0)
            ax.set_title('Model F1 Score Comparison', fontsize=14, fontweight='bold')
            for bar, f1 in zip(bars, metrics_df['F1 Score']):
                ax.text(f1 + 0.005, bar.get_y() + bar.get_height()/2, f'{f1:.4f}', 
                       va='center', fontsize=11, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        
        # Confusion matrices
        st.subheader("Confusion Matrices", anchor=False)
        
        selected_model = st.selectbox("Select a model to view confusion matrix:", list(models.keys()))
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            model_data = models[selected_model]
            cm = confusion_matrix(model_data['y_test'], model_data['y_pred'])
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                       xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], ax=ax,
                       annot_kws={'size': 16})
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('True Label', fontsize=12)
            ax.set_title(f'Confusion Matrix - {selected_model}', fontsize=14, fontweight='bold')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("### Classification Report")
            report = classification_report(model_data['y_test'], model_data['y_pred'], 
                                          target_names=['Negative', 'Positive'], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)
        
        # Performance summary table
        st.markdown("---")
        st.subheader("Performance Summary", anchor=False)
        st.dataframe(metrics_df.style.format({'Accuracy': '{:.4f}', 'F1 Score': '{:.4f}'})
                    .background_gradient(cmap='Greens', subset=['Accuracy', 'F1 Score']),
                    use_container_width=True, hide_index=True)
    
    # ============== Tab 4: Mathematical Formulations & ROC ==============
    with tab4:
        st.header("Mathematical Formulations & ROC Curves", anchor=False)
        
        with st.spinner("Training models..."):
            models = train_models(df)
        
        # Feature Extraction Methods
        st.subheader("Feature Extraction Methods", anchor=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(TFIDF_FORMULATION)
        with col2:
            st.markdown(BOW_FORMULATION)
        
        st.markdown("---")
        
        # Model Formulations
        st.subheader("Classification Model Formulations", anchor=False)
        
        model_tab1, model_tab2, model_tab3, model_tab4 = st.tabs([
            "Logistic Regression", "Random Forest", "SVM", "Naive Bayes"
        ])
        
        with model_tab1:
            st.markdown(LOGREG_FORMULATION)
        with model_tab2:
            st.markdown(RF_FORMULATION)
        with model_tab3:
            st.markdown(SVM_FORMULATION)
        with model_tab4:
            st.markdown(NB_FORMULATION)
        
        st.markdown("---")
        
        # ROC Curves
        st.subheader("ROC Curves", anchor=False)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12']
        
        for (name, model_data), color in zip(models.items(), colors):
            y_test = model_data['y_test']
            y_pred = model_data['y_pred']
            fpr, tpr, _ = roc_curve(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred)
            ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {auc_score:.3f})')
        
        # Diagonal line
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
        ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
        ax.set_title('ROC Curves - All Models', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # AUC Summary Table
        st.subheader("AUC Scores Summary", anchor=False)
        
        auc_data = []
        for name, model_data in models.items():
            y_test = model_data['y_test']
            y_pred = model_data['y_pred']
            auc_score = roc_auc_score(y_test, y_pred)
            auc_data.append({'Model': name, 'AUC Score': auc_score})
        
        auc_df = pd.DataFrame(auc_data).sort_values('AUC Score', ascending=False)
        st.dataframe(
            auc_df.style.format({'AUC Score': '{:.4f}'})
                  .background_gradient(cmap='Blues', subset=['AUC Score']),
            use_container_width=True, hide_index=True
        )
        
        # Model Summary Statistics
        st.markdown("---")
        st.subheader("Complete Model Summary", anchor=False)
        
        summary_data = []
        for name, model_data in models.items():
            y_test = model_data['y_test']
            y_pred = model_data['y_pred']
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            summary_data.append({
                'Model': name,
                'Accuracy': model_data['accuracy'],
                'F1 Score': model_data['f1'],
                'AUC': roc_auc_score(y_test, y_pred),
                'True Positives': tp,
                'True Negatives': tn,
                'False Positives': fp,
                'False Negatives': fn,
                'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0
            })
        
        summary_df = pd.DataFrame(summary_data).sort_values('F1 Score', ascending=False)
        st.dataframe(
            summary_df.style.format({
                'Accuracy': '{:.4f}', 'F1 Score': '{:.4f}', 'AUC': '{:.4f}',
                'Precision': '{:.4f}', 'Recall': '{:.4f}'
            }).background_gradient(cmap='Greens', subset=['Accuracy', 'F1 Score', 'AUC']),
            use_container_width=True, hide_index=True
        )
    
    # ============== Tab 5: Live Prediction ==============
    with tab5:
        st.header("Live Sentiment Prediction", anchor=False)
        st.markdown("Enter a tweet or review to predict its sentiment using our trained models.")
        
        with st.spinner("Loading models..."):
            models = train_models(df)
        
        user_input = st.text_area("Enter your text here:", height=150, 
                                  placeholder="Type a tweet or review to analyze...")
        
        model_choice = st.selectbox("Select Model:", list(models.keys()))
        
        if st.button("Predict Sentiment", type="primary", icon=":material/psychology:"):
            if user_input.strip():
                model_data = models[model_choice]
                vectorizer = model_data['vectorizer']
                model = model_data['model']
                
                # Preprocess with improved text handling
                cleaned = preprocess_for_prediction(user_input)
                
                # Show preprocessed text
                if cleaned != user_input.lower().strip():
                    st.info(f"Preprocessed text: *{cleaned}*")
                
                X_input = vectorizer.transform([cleaned])
                
                # Predict
                prediction = model.predict(X_input)[0]
                
                # Display result
                if prediction == 1:
                    st.success("### Positive Sentiment", icon=":material/thumb_up:")
                    st.balloons()
                else:
                    st.error("### Negative Sentiment", icon=":material/thumb_down:")
                
                # Confidence (if available)
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_input)[0]
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Negative Probability", f"{proba[0]*100:.1f}%")
                    with col2:
                        st.metric("Positive Probability", f"{proba[1]*100:.1f}%")
                    
                    # Confidence bar
                    fig, ax = plt.subplots(figsize=(10, 2))
                    ax.barh(['Sentiment'], [proba[0]], color='#e74c3c', label='Negative')
                    ax.barh(['Sentiment'], [proba[1]], left=[proba[0]], color='#2ecc71', label='Positive')
                    ax.set_xlim(0, 1)
                    ax.set_xlabel('Probability')
                    ax.legend(loc='upper right')
                    ax.set_title('Prediction Confidence')
                    st.pyplot(fig)
                    plt.close()
            else:
                st.warning("Please enter some text to analyze.")
    
    # ============== Tab 6: Data Explorer ==============
    with tab6:
        st.header("Data Explorer", anchor=False)
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            sentiment_filter = st.multiselect("Filter by Sentiment:", ['Positive', 'Negative'], 
                                              default=['Positive', 'Negative'])
        with col2:
            n_rows = st.slider("Number of rows to display:", 10, 100, 25)
        
        # Filter data
        filtered_df = df[df['sentiment'].isin(sentiment_filter)]
        
        # Display
        st.dataframe(
            filtered_df[['review_text', 'sentiment', 'doc_len']].head(n_rows).rename(
                columns={'review_text': 'Tweet', 'sentiment': 'Sentiment', 'doc_len': 'Word Count'}
            ),
            use_container_width=True,
            hide_index=True
        )
        
        # Download option
        csv = filtered_df[['review_text', 'sentiment']].to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name="sentiment_data.csv",
            mime="text/csv",
            icon=":material/download:"
        )

    # ============== Tab 7: Batch Prediction ==============
    with tab7:
        st.header("Batch Sentiment Prediction", anchor=False)
        st.markdown("Upload a CSV of tweets or reviews and score every non-empty text row.")

        with st.spinner("Loading models..."):
            models = train_models(df)

        batch_model_choice = st.selectbox(
            "Select Batch Model:",
            list(models.keys()),
            key="batch_model_choice"
        )
        uploaded_file = st.file_uploader("Upload CSV file:", type=["csv"])

        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)
            except Exception as exc:
                st.error(f"Could not read CSV file: {exc}")
                input_df = None

            if input_df is not None:
                if input_df.empty:
                    st.warning("The uploaded CSV has no rows.")
                else:
                    detected_column = detect_text_column(input_df.columns)
                    column_options = list(input_df.columns)
                    default_index = column_options.index(detected_column) if detected_column in column_options else 0
                    text_column = st.selectbox(
                        "Text column:",
                        column_options,
                        index=default_index
                    )

                    results_df, valid_mask = build_batch_predictions(
                        input_df,
                        text_column,
                        models[batch_model_choice]
                    )

                    st.metric("Rows Scored", int(valid_mask.sum()))
                    st.dataframe(results_df, use_container_width=True, hide_index=True)

                    csv_results = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv_results,
                        file_name="batch_sentiment_predictions.csv",
                        mime="text/csv",
                        icon=":material/download:"
                    )

if __name__ == "__main__":
    main()
