import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Sample travel knowledge base (in a real scenario, this would be larger)
knowledge_base = {
    "faqs": [
        {
            "question": "What documents do I need for international travel?",
            "answer": "For international travel, you typically need a valid passport, visa (if required by the destination country), and sometimes proof of onward travel or vaccinations."
        },
        {
            "question": "How early should I arrive at the airport for an international flight?",
            "answer": "For international flights, it's recommended to arrive at least 3 hours before your scheduled departure time to allow for check-in, security, and immigration procedures."
        },
        {
            "question": "What is the best time to book flights for cheap prices?",
            "answer": "Generally, booking flights 6-8 weeks in advance for domestic and 3-5 months for international travel can yield better prices. Also, flying mid-week (Tuesday-Wednesday) is often cheaper."
        },
        {
            "question": "What should I do if my flight is canceled?",
            "answer": "If your flight is canceled, immediately contact the airline for rebooking options. You may be entitled to compensation or accommodation depending on the circumstances and local regulations."
        },
        {
            "question": "How can I avoid jet lag?",
            "answer": "To minimize jet lag: adjust your sleep schedule before traveling, stay hydrated, avoid alcohol and caffeine during the flight, and try to adapt to the local time zone as soon as possible."
        },
        {
            "question": "What are the baggage allowance rules?",
            "answer": "Baggage allowance varies by airline and ticket class. Typically, economy allows 1 carry-on (7-10kg) and 1 checked bag (20-23kg). Always check with your specific airline for exact limits."
        },
        {
            "question": "Do I need travel insurance?",
            "answer": "Travel insurance is highly recommended as it covers medical emergencies, trip cancellations, lost baggage, and other unforeseen circumstances during your travels."
        }
    ]
}

# Save knowledge base to a JSON file (for demonstration)
with open('travel_kb.json', 'w') as f:
    json.dump(knowledge_base, f)

# Load knowledge base from JSON file
with open('travel_kb.json', 'r') as f:
    kb = json.load(f)

# Preprocess text function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Prepare questions and answers
questions = [q['question'] for q in kb['faqs']]
answers = [q['answer'] for q in kb['faqs']]

# Preprocess all questions
processed_questions = [preprocess_text(q) for q in questions]

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(processed_questions)

def get_most_similar_answer(user_query):
    # Preprocess user query
    processed_query = preprocess_text(user_query)
    # Vectorize the query
    query_vector = vectorizer.transform([processed_query])
    # Compute similarity
    similarities = cosine_similarity(query_vector, question_vectors)
    # Get index of most similar question
    most_similar_idx = np.argmax(similarities)
    # Get similarity score
    similarity_score = similarities[0][most_similar_idx]
    
    # Threshold to determine if the match is good enough
    if similarity_score > 0.5:
        return answers[most_similar_idx], similarity_score
    else:
        return "I'm sorry, I don't have information about that. Please try asking another travel-related question.", similarity_score

# Simple CLI interface
def chat():
    print("Travel Assistant: Hi! I'm your travel assistant. Ask me any travel-related questions or type 'quit' to exit.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Travel Assistant: Goodbye! Safe travels!")
            break
        
        answer, score = get_most_similar_answer(user_input)
        print(f"\nTravel Assistant: {answer}")
        # Uncomment below to see the confidence score
        # print(f"(Confidence: {score:.2f})")

# Start the chat
if __name__ == "__main__":
    chat()