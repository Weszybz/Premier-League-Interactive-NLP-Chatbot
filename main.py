import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

# Load Q&A pairs from the CSV file
def load_qa_pairs(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path)
    # Filter questions related to bonds and stocks
    filtered_data = data[data['Question'].str.contains(r'\bstocks\b|\bbonds\b', case=False, na=False)]
    # Create a dictionary of questions and answers
    qa_dict = dict(zip(filtered_data['Question'].str.lower(), filtered_data['Answer']))
    return qa_dict

# Initialize Q&A pairs from the CSV
qa_pairs = load_qa_pairs('COMP3074-CW1-Dataset.csv')

# 1. Intent Classifier with Preprocessing
training_data = [
    ("Hi", "greeting"),
    ("Hello", "greeting"),
    ("how are you", "small_talk"),
    ("What is my name?", "identity"),
    ("What is my name", "identity"),
    ("My name is John", "identity"),
    ("Tell me about stocks", "qa"),
    ("What are bonds?", "qa"),
    ("What are bonds", "qa"),
    ("What are stocks?", "qa"),
    ("What can you do?", "small_talk"),
    ("What are stocks and bonds?", "qa"),
    ("Bye", "farewell"),
    ("Goodbye", "farewell")
]

texts, labels = zip(*training_data)

# Preprocessing function to strip punctuation
def preprocess_input(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Preprocess training data
processed_texts = [preprocess_input(text) for text in texts]

# Pipeline for intent classification
intent_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', LogisticRegression())
])

# Train intent classifier
intent_pipeline.fit(processed_texts, labels)

# Generate TF-IDF vectors for Q&A pairs
questions = list(qa_pairs.keys())
tfidf_vectorizer = TfidfVectorizer().fit(questions)
question_vectors = tfidf_vectorizer.transform(questions)

# 3. Question Answering
def answer_query(query):
    query = preprocess_input(query)  # Preprocess the query
    query_vector = tfidf_vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, question_vectors)
    best_match = similarities.argmax()
    return qa_pairs[questions[best_match]]

# 4. Small Talk
def small_talk_response(user_input):
    responses = {
        "how are you": "I'm just a bot, but I'm doing great! How about you?",
        "hello": "Hello! What's your name?",
        "hi": "Hi there! What's your name?",
        "what can you do": "I can answer questions, remember your name, and engage in small talk!",
        "bye": "Have a great day!",
        "goodbye": "Goodbye! Take care."
    }
    user_input = preprocess_input(user_input)
    return responses.get(user_input, "I'm here to help!")

# 5. Chatbot Response Function
def chatbot_response(user_input):
    user_input = preprocess_input(user_input)
    intent = intent_pipeline.predict([user_input])[0]
    if intent == "greeting":
        return small_talk_response(user_input)
    elif intent == "identity":
        if "my name is" in user_input:
            name = user_input.split("is")[-1].strip()
            return set_user_name(name)
        else:
            return get_user_name()
    elif intent == "qa":
        return answer_query(user_input)
    elif intent == "small_talk":
        return small_talk_response(user_input)
    elif intent == "farewell":
        return small_talk_response(user_input)
    else:
        return "I'm not sure how to help with that."

# 2. Identity Management
user_name = None

def set_user_name(name):
    global user_name
    user_name = name
    return f"Hi {name}!"

def get_user_name():
    return f"Your name is {user_name}." if user_name else "I don't know your name yet."

# Example Conversation
if __name__ == "__main__":
    print("Chatbot: Hello! How can I help you?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "stop"]:
            print("Chatbot: Goodbye!")
            break
        print(f"Chatbot: {chatbot_response(user_input)}")
