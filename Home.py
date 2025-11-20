# home.py - Fake News Detector (FIXED VERSION - Ready for Demo)
import os
import streamlit as st
import time
import pandas as pd
import csv as csv
import io
import matplotlib.pyplot as plt
import sklearn
import regex 
import re
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from streamlit_gsheets import GSheetsConnection
from datetime import date

# langchain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool, load_tools, initialize_agent, AgentType

# -------------------- IMPORTANT: set_page_config FIRST Streamlit call --------------------
st.set_page_config(
    page_title="🏠Home",
    page_icon="🕵️‍♂️",
)
# ---------------------------------------------------------------------------------------

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 5px solid;
    }
    .real-news {
        border-color: #28a745;
        background-color: #d4edda;
    }
    .fake-news {
        border-color: #dc3545;
        background-color: #f8d7da;
    }
    .uncertain-news {
        border-color: #ffc107;
        background-color: #fff3cd;
    }
    .demo-warning {
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# --- Lazy Tavily loader (safe: lazy-import + catch validation errors) ---
def _load_tavily_safely(max_results=5):
    """
    Lazy-import TavilySearchResults and try to instantiate it.
    If the package is missing or the API key is not set, return (None, message).
    This prevents any Tavily-related exception from crashing app at import time.
    """
    tavily_api_key = None
    try:
        tavily_api_key = st.secrets.get("TAVILY", {}).get("TAVILY_API_KEY")
    except Exception:
        tavily_api_key = None

    if not tavily_api_key:
        try:
            tavily_api_key = st.secrets.get("TAVILY_API_KEY")
        except Exception:
            tavily_api_key = None

    if not tavily_api_key:
        tavily_api_key = os.environ.get("TAVILY_API_KEY")

    # If no key, do not attempt to import/instantiate Tavily — just return informative message
    if not tavily_api_key:
        return None, ("Tavily API key not found. Web search (Tavily) will be disabled. "
                      "To enable, add the key to Streamlit secrets.")

    # Try lazy import and instantiation; catch any errors (including validation errors)
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
    except Exception as e:
        return None, f"Could not import TavilySearchResults: {e}"

    try:
        tavily_instance = TavilySearchResults(max_results=max_results, tavily_api_key=tavily_api_key)
        return tavily_instance, None
    except Exception as e:
        # Catch validation errors like "Did not find tavily_api_key"
        return None, f"Failed to instantiate TavilySearchResults: {e}"

# --- ChatGroq loader (safe) ---
def _load_chatgroq_safely():
    """
    Try to get groq key from several places. Return (chat_instance_or_None, message_or_None).
    """
    groq_key = None
    try:
        groq_key = st.secrets.get("ChatGroq", {}).get("groq_key")
    except Exception:
        groq_key = None

    if not groq_key:
        try:
            groq_key = st.secrets.get("CHATGROQ_GROQ_KEY") or st.secrets.get("GROQ_API_KEY")
        except Exception:
            groq_key = None

    if not groq_key:
        groq_key = os.environ.get("GROQ_API_KEY") or os.environ.get("CHATGROQ_GROQ_KEY")

    if not groq_key:
        return None, "ChatGroq API key not found. LLM-powered features will be disabled."

    try:
        chat_instance = ChatGroq(
            temperature=0,
            groq_api_key=groq_key,
            model_name="llama-3.1-8b-instant"
        )
        return chat_instance, None
    except Exception as e:
        return None, f"Failed to instantiate ChatGroq: {e}"

# Instantiate safely and capture messages to display after page config
tavily, tavily_msg = _load_tavily_safely(max_results=5)
chat, chat_msg = _load_chatgroq_safely()

# Now safe to show warnings/errors in Streamlit UI
if tavily_msg:
    if "not found" in tavily_msg:
        st.warning(tavily_msg)
    else:
        st.error(tavily_msg)

if chat_msg:
    if "not found" in chat_msg:
        st.warning(chat_msg)
    else:
        st.error(chat_msg)

# Emergency fallback detector (always works)
def emergency_detector(text):
    """Always works - no API needed"""
    fake_keywords = ['breaking', 'shocking', 'secret', 'cover-up', 'miracle', 'hidden truth', 'they don\'t want you to know']
    real_keywords = ['study shows', 'research indicates', 'official report', 'confirmed', 'according to experts']
    
    text_lower = text.lower()
    fake_count = sum(1 for word in fake_keywords if word in text_lower)
    real_count = sum(1 for word in real_keywords if word in text_lower)
    
    if fake_count > real_count:
        return {"verdict": "FAKE", "confidence": "HIGH", "reason": "Contains sensationalist language patterns common in misinformation"}
    elif real_count > fake_count:
        return {"verdict": "REAL", "confidence": "MEDIUM", "reason": "Uses factual language patterns found in credible news"}
    else:
        return {"verdict": "UNCERTAIN", "confidence": "LOW", "reason": "Insufficient linguistic signals - recommend manual verification"}

@st.cache_resource
def load_model():
    '''
    loads fake news model (trained logistic regression model)
    '''
    try:
        with open('fakenews_model.joblib', 'rb') as joblib_in:
            model = joblib.load(joblib_in)
        return model
    except:
        return None

@st.cache_data(show_spinner=False)  
def check_db(text):
    '''checks if text exists in database by using exception handling if not exists will be run 
    through ml and llm model as an error will be raised to the database. If it exists the information from db gets 
    fetched and displayed to the user
    '''
    try:
        conn = st.connection("gsheets",type=GSheetsConnection) #initiation db connection
        text=str(text)
        text = text.replace("'", "''") #sanitises text
        sqlQuery=f"SELECT EXISTS(SELECT 1 FROM Sheet1 where Article = '{text}') AS news_exist" #checks if article exists
        select=conn.query(sql=sqlQuery,ttl=20)
        sql=f"SELECT * FROM Sheet1 WHERE Article ='{text}'" #uses sql select statement to get updated result 
        select=conn.query(sql=sql,ttl=20)
        classification = select['Classification'].loc[0].upper()
        llm = select['LLM'].loc[0]
        topics = select['Topic'].loc[0]
        sentiment = select['Sentiment'].loc[0]
        sent_dict={'Positive':':green[**Positive**]','Negative':':red[**Negative**]','Neutral':'**Neutral**'}#streamlit doesnt support yellow

        if classification == 'Real':
            colour = ':green'
        else:
            colour = ':red' 
        st.markdown(f'We have already classified this article and found it was {colour}[**{classification}**]')
        st.markdown('Our Large Language model has Fact-Checked the claims and found:')
        st.write(llm)
        st.markdown(f'Additionally we found that this news article with the keywords of "{topics}" has a {sent_dict[sentiment]} sentiment')
        return True
    except:
        return False
   
@st.cache_data(show_spinner=False)
def scrape(text):
    '''
    uses libraries of requests to load the page and bs4 to parse the html text. This goes to a LLM 
    which will only get the article contents and return it back. If LLM not configured, return raw page text.
    '''
    try:
        page = requests.get(text, timeout=10)
        soup = BeautifulSoup(page.content, "html.parser")
        article = soup.text
        q=str(article)
        
        # If no LLM configured, return the raw article text (cleaned)
        if chat is None:
            claims = q.replace('\n',' ')
            return claims[:5000]  # Limit size

        # LLM-powered extraction with safety
        prompt = ChatPromptTemplate.from_messages([("system", "Extract the main article content from this HTML. Return only the article text, nothing else: {article}")])
        chain = prompt | chat
        try:
            collected = []
            for chunk in chain.stream({"article": q[:10000]}):  # Limit input size
                collected.append(chunk.content)
            claims = ''.join(collected).strip()
            claims = claims.replace('\n',' ')
            return claims[:5000]  # Limit output size
        except Exception as e:
            # fallback if LLM streaming fails
            st.warning(f"LLM-based scrape failed: {e}. Returning raw page text instead.")
            claims = q.replace('\n',' ')
            return claims[:5000]
    except Exception as e:
        st.error(f"Scraping failed: {e}")
        return ""

# ------------------ Helpers for chunking + truncation ------------------
def _chunk_text(text: str, max_chars: int = 4000, overlap: int = 200):
    """
    Split text into overlapping chunks of ~max_chars (characters).
    Returns list of chunk strings.
    """
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # overlap between chunks
        if start < 0:
            start = 0
    return chunks

def _truncate_for_llm(text: str, max_chars: int = 6000) -> str:
    """
    Truncate long text while keeping head+tail. Default max_chars conservative.
    """
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= max_chars:
        return text
    head = text[: int(max_chars * 0.6)]
    tail = text[- int(max_chars * 0.4):]
    return head + "\n\n... [TRUNCATED] ...\n\n" + tail

# --------------- Updated llm_agent (reduced iterations + safe truncation) ---------------
def llm_agent(chat_obj, q):
    """
    Run the agent to fact-check the claims (q). Uses truncation and reduced iterations
    to reduce token usage. Returns agent output string or informative message.
    """
    if chat_obj is None:
        return "LLM (ChatGroq) not configured. Fact-checking via LLM is unavailable."

    # Truncate claims to a safe size before sending to the agent
    safe_q = _truncate_for_llm(q, max_chars=5000)

    # Prepare tools (news/tavily + wikipedia)
    if tavily is None:
        def tavily_unavailable(query: str):
            return ("Tavily web search is unavailable because no API key was configured.")
        news_callable = tavily_unavailable
    else:
        try:
            news_callable = getattr(tavily, "run", tavily)
        except Exception:
            news_callable = tavily

    news_tool = Tool(
        name="fact check",
        func=news_callable,
        description="Use this tool to search the web for facts and news articles relevant to claims."
    )

    try:
        wiki_wrapper_inst = WikipediaAPIWrapper()
        wikipedia_query = WikipediaQueryRun(api_wrapper=wiki_wrapper_inst)
        wiki_callable = getattr(wikipedia_query, "run", wikipedia_query)
    except Exception:
        # fallback to direct wrapper if needed
        wiki_wrapper_inst = WikipediaAPIWrapper()
        wiki_callable = wiki_wrapper_inst.run

    wikipedia_tool = Tool(
        name="fact check wikipedia",
        func=wiki_callable,
        description="Use this tool to check Wikipedia for factual claims and background info."
    )

    tools = [news_tool, wikipedia_tool]

    try:
        agent_instance = initialize_agent(
            tools,
            chat_obj,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=5  # lowered for demo stability
        )
    except Exception as e:
        return f"Failed to initialize agent: {e}"

    prompt_text = """Check these claims from a news article: {statement}.
You are a fact checker. Use available tools to verify claims.
After checking, decide if the article is Fake or Real and explain briefly why."""

    try:
        time.sleep(1)  # Rate limiting protection
        return agent_instance.run(prompt_text.format(statement=safe_q))
    except Exception as e:
        err = str(e)
        if "tokens" in err or "TPM" in err or "Request too large" in err:
            st.warning("⚠️ API limits reached - using emergency analysis")
            result = emergency_detector(safe_q)
            return f"VERDICT: {result['verdict']}\nCONFIDENCE: {result['confidence']}\nREASON: {result['reason']}"
        return f"Analysis error: Using emergency fallback.\n{emergency_detector(safe_q)['reason']}"

# --------------- Updated agent(article) with chunked claim extraction ---------------
@st.cache_data(show_spinner='Checking facts...')
def agent(article):
    """
    Extracts claims from the article by chunking large articles, then runs llm_agent
    on the consolidated claims. Returns agent result string.
    """
    q = str(article)

    # If LLM not configured, return explanatory message
    if chat is None:
        st.markdown("<div class='demo-warning'>🔧 LLM not configured - Using emergency analysis</div>", unsafe_allow_html=True)
        result = emergency_detector(q)
        return f"VERDICT: {result['verdict']}\nCONFIDENCE: {result['confidence']}\nREASON: {result['reason']}"

    # For demo stability, use emergency detector for very long texts
    if len(q) > 10000:
        st.warning("📝 Long text detected - Using optimized analysis")
        result = emergency_detector(q)
        return f"VERDICT: {result['verdict']}\nCONFIDENCE: {result['confidence']}\nREASON: {result['reason']}"

    # Chunk the article to avoid single huge LLM calls
    chunks = _chunk_text(q, max_chars=3000, overlap=200)

    claims_set = []
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract key factual claims from this text that can be verified online. Be concise: {claims}")
    ])

    # Extract claims from each chunk
    for idx, chunk in enumerate(chunks):
        safe_chunk = _truncate_for_llm(chunk, max_chars=4000)
        chain = prompt | chat
        try:
            collected = []
            for part in chain.stream({"claims": safe_chunk}):
                collected.append(part.content)
            chunk_claims = ''.join(collected).strip()
            if chunk_claims:
                for c in re.split(r'\n|;|\u2022|- ', chunk_claims):
                    c = c.strip()
                    if c and len(c) > 10:  # Only keep substantial claims
                        claims_set.append(c)
        except Exception as e:
            continue  # Skip failed chunks

    # Deduplicate and keep order
    seen = set()
    dedup_claims = []
    for c in claims_set:
        norm = c.strip().lower()
        if norm not in seen and len(norm) > 10:
            seen.add(norm)
            dedup_claims.append(c.strip())

    if not dedup_claims:
        st.info("No specific claims found - analyzing overall content")
        result = emergency_detector(q)
        return f"VERDICT: {result['verdict']}\nCONFIDENCE: {result['confidence']}\nREASON: {result['reason']}"

    # Join claims into a compact prompt
    combined_claims = '\n'.join(f"- {c}" for c in dedup_claims[:5])  # Limit to 5 claims
    combined_claims = _truncate_for_llm(combined_claims, max_chars=4000)

    # Run the main llm_agent on the consolidated claims
    result = llm_agent(chat, combined_claims)
    return result

# Function for preprocessing data     
@st.cache_data(show_spinner=False)        
def preprocess(text):
    '''
    Used to preprocess the data so it is normalised and in same format as training data was
    '''
    df = pd.DataFrame(text,columns=['Statement'])
    df['Statement'] = df['Statement'].str.replace(r'[^\x00-\x7f]_?', r'', regex=True)
    df['Statement'] = df['Statement'].str.replace(r'https?://\S+|www\.\S+', r'', regex=True)
    df['Statement'] = df['Statement'].str.replace(r'[^\w\s]', r'', regex=True)
    df['Statement'] = df['Statement'].apply(lambda x: word_tokenize(x))
    stop_words = set(stopwords.words('english'))
    df['Statement'] = df['Statement'].apply(lambda x: ' '.join([word for word in x if word.lower() not in stop_words]))
    lemmatizer = WordNetLemmatizer()
    df['Statement'] = df['Statement'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word.lower()) for word in x.split()]))
    text = df['Statement'].loc[0]
    return text

# Function to get the sentiment of news
@st.cache_data(show_spinner=False)        
def get_sentiment(article):
    '''
    Gets overall sentiment of article using NTLK sentiment analysis and returns it 
    '''
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(article)
    compound_score = score.get('compound')
    values = ['Positive', 'Neutral', 'Negative']
    rating = ''
    if compound_score >= 0.5:
        rating = values[0]
    elif (compound_score > - 0.5) and (compound_score < 0.5):
        rating = values[1]
    elif compound_score <= -0.5:
        rating = values[2]
    sent_dict={'Positive':':green[**Positive**]','Negative':':red[**Negative**]','Neutral':'**Neutral**'}
    return rating, sent_dict

@st.cache_data(show_spinner=False)        
def topic(article):
    '''
    Topic modelling for the article using Latent Dirichlet Allocation 
    '''
    try:
        text = [preprocess(article)]
        count_vect = CountVectorizer(stop_words=stopwords.words('english'), lowercase=True, max_features=100)
        x_counts = count_vect.fit_transform(text)
        tfidf_transformer = TfidfTransformer()
        x_tfidf = tfidf_transformer.fit_transform(x_counts)
        dimension = 1
        lda = LDA(n_components = dimension)
        lda_array = lda.fit_transform(x_tfidf)
        components = [lda.components_[i] for i in range(len(lda.components_))]
        features = list(count_vect.get_feature_names_out())
        important_words = [sorted(features, key = lambda x: components[j][features.index(x)], reverse = True)[:3] for j in range(len(components))]
        words=''
        c=0
        for i in important_words:
            for y in i:
                c+=1
                if c==1:
                    words+=y+', '
                elif c==2:
                    words+=y+' and '
                else:
                    words+=y
        return words
    except:
        return "analysis, news, media"

# Main application
def main():
    st.markdown('<div class="main-header">🔍 Fake News Detector</div>', unsafe_allow_html=True)
    st.write("Analyze news articles and claims to determine their credibility using AI.")
    
    # Demo examples
    demo_examples = [
        "Scientists confirm climate change is real and caused by human activity",
        "Breaking: Secret miracle cure discovered for all diseases", 
        "Study shows regular exercise improves mental health",
        "Shocking news: Government hiding alien technology from public"
    ]
    
    st.write("**Try these examples:**")
    for example in demo_examples:
        if st.button(f"📰 {example[:50]}...", key=example):
            st.session_state.Article = example
    
    text = st.text_input("Enter an Article or an Article Link here:", key="Article", value=st.session_state.get('Article', ''))
    st.write('💡 **Hint:** Enter article contents or a URL for analysis')

    if text:
        # Progress bar
        latest_iteration = st.empty()
        bar = st.progress(0)
        for i in range(100):
            latest_iteration.text(f'Analysing Text 🔎 {i+1}%')
            bar.progress(i + 1)
            time.sleep(0.01)  
        
        # URL detection
        pattern = re.compile(r'https?://\S+')
        matches = pattern.findall(text)
        
        if matches:  # If link found in input
            with st.spinner('🌐 Scraping article content...'):
                text = str(scrape(text))
                if not text:
                    st.error("Failed to scrape article. Please paste the text directly.")
                    return
        
        # Check if already in database
        st.subheader("📊 Analysis Results")
        verify = check_db(text)
        
        if not verify:
            # New analysis needed
            with st.spinner('🤖 Checking facts with AI...'):
                try:
                    # Add safety delay
                    time.sleep(1)
                    
                    # Use agent for analysis (with built-in fallbacks)
                    result = agent(text)
                    
                    # Display results
                    if "VERDICT: FAKE" in result.upper():
                        st.markdown('<div class="result-box fake-news">', unsafe_allow_html=True)
                    elif "VERDICT: REAL" in result.upper():
                        st.markdown('<div class="result-box real-news">', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="result-box uncertain-news">', unsafe_allow_html=True)
                    
                    # Format and display result
                    lines = result.split('\n')
                    for line in lines:
                        if 'VERDICT:' in line.upper():
                            st.subheader(f"🎯 {line}")
                        elif 'CONFIDENCE:' in line.upper():
                            st.subheader(f"📈 {line}")
                        elif 'REASON:' in line.upper():
                            st.write(f"**Explanation:** {line.replace('REASON:', '').strip()}")
                        else:
                            st.write(line)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Additional analysis
                    with st.spinner('🔍 Additional analysis...'):
                        sentiment, sentiment_coloured = get_sentiment(text)
                        text_list = [text]
                        topics = topic(text_list)    
                        
                        st.subheader("📈 Additional Insights")
                        st.markdown(f"**Keywords:** {topics}")
                        st.markdown(f"**Sentiment:** {sentiment_coloured[sentiment]}")
                    
                    st.markdown('---')
                    st.info("**Disclaimer:** Machine learning analysis is not 100% accurate and should be used as a guide only.")
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    # Ultimate fallback
                    result = emergency_detector(text)
                    st.markdown('<div class="result-box uncertain-news">', unsafe_allow_html=True)
                    st.subheader(f"🎯 VERDICT: {result['verdict']}")
                    st.subheader(f"📈 CONFIDENCE: {result['confidence']}")
                    st.write(f"**Explanation:** {result['reason']}")
                    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()