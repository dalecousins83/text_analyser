import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
import spacy
from textblob import TextBlob
from gensim import corpora, models
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.tokenizers import Tokenizer

# Prompt user for input text
input_text = input('Paste the text to analyze here: ')

# Perform text preprocessing
text = input_text.lower() # Convert text to lowercase
text = nltk.sent_tokenize(text) # Split text into sentences
text = [nltk.word_tokenize(sent) for sent in text] # Split sentences into words
stop_words = set(nltk.corpus.stopwords.words('english'))
text = [[word for word in sent if word not in stop_words] for sent in text] # Remove stop words
lemmatizer = nltk.stem.WordNetLemmatizer()
text = [[lemmatizer.lemmatize(word) for word in sent] for sent in text] # Lemmatize words

# PERFORM SENTIMENT ANALYSIS
blob = TextBlob(input_text)
polarity = blob.sentiment.polarity
subjectivity = blob.sentiment.subjectivity
print(f"\nSentiment analysis results: polarity={polarity}, subjectivity={subjectivity}")
print(f"*Polarity scale is -1 (negative) to 1 (positive) and Subjectivity score is 0 (objective fact) to 1 (personal opinion)")

# PERFORM TOPIC MODELING
dictionary = corpora.Dictionary(text)
corpus = [dictionary.doc2bow(sent) for sent in text]
print(f"\ndictionary is ")
print(dictionary)

print(f"\ncorpus is ")
print(corpus)

lda = models.LdaModel(corpus, num_topics=10, id2word=dictionary)
print("\nTopic modeling results:\n")
for i, topic in lda.show_topics():
    print(f"Topic {i+1}: {topic}")

# PERFORM TEXT SUMMARIZATION
parser = PlaintextParser.from_string(input_text, Tokenizer('english'))
summarizer = LsaSummarizer()
summary = summarizer(parser.document, 10) # Generate a summary of sentences
print("\nText summarization results:\n")
for sentence in summary:
    print(sentence)

# PERFORM PLOT ANALYSIS
nlp = spacy.load('en_core_web_sm')
doc = nlp(input_text)
print("\nPlot analysis results:\n")
for ent in doc.ents:
    if ent.label_ == "PERSON":
        print(f"Character: {ent.text}, Entity Type: {ent.label_}")
    elif ent.label_ == "LOC":
        print(f"Setting: {ent.text}, Entity Type: {ent.label_}")
    #all other types
    else:
        print(f"Other: {ent.text}, Entity Type: {ent.label_}")
