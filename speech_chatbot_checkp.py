import rich.color
import streamlit as st
import speech_recognition as sr
import string
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

with open("C:\\Users\\Amarachi Uzochukwu\\Mark Twain.txt", 'r', encoding='utf-8') as f:
    data = f.read().replace('\n', ' ')
sentences = sent_tokenize(data)

def preprocess(sentence):
    # Tokenize the sentence into words
    words = word_tokenize(sentence)
    # Remove stopwords and punctuation
    words = [word.lower() for word in words if word.lower() not in stopwords.words('english') and word not in string.punctuation]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

# Preprocess each sentence in the text
corpus = [preprocess(sentence) for sentence in sentences]

# Define a function to find the most relevant sentence given a query
def get_most_relevant_sentence(query):
    # Preprocess the query
    query = preprocess(query)
    # Compute the similarity between the query and each sentence in the text
    max_similarity = 0
    most_relevant_sentence = ""
    for sentence in corpus:
        similarity = len(set(query).intersection(sentence)) / float(len(set(query).union(sentence)))
        if similarity > max_similarity:
            max_similarity = similarity
            most_relevant_sentence = " ".join(sentence)
    return most_relevant_sentence


def transcribe_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Speak now...")
        # listen for speech and store in audio_text variable
        audio_text = r.listen(source)
        st.info("Transcribing...")

        try:
            # using google speech recognition
            text = r.recognize_google(audio_text)
            return text
        except:
            return "Sorry, i did not get that."

def chatbot(question):
    # Find the most relevant sentence
    most_relevant_sentence = get_most_relevant_sentence(question)
    r = sr.recognizers
    if ('text_input'):
        return most_relevant_sentence
    else:
        r.recognize_google(mrelevant_sentence)

def main():
    st.title('ChatBot Speech Recognition App')

    user_options = ['Text', 'Speech']
    user = st.selectbox('Preferred Choice', user_options)
    r = sr.recognizers

    if st.button('start recording'):
        text = transcribe_speech(user)
        st.write('Transcription:', text)
    elif st.button('start recording'):
        speech = r.recognize_google(user)
        st.write('Transcription:', speech)

if __name__ == "__main__":
    main()




