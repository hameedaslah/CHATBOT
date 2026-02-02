from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

questions = [
    "hi", "hello", "how are you", "what is your name", "bye",
    "goodbye", "who made you", "what can you do"
]

answers = [
    "Hello!", "Hi there Nigga!", "I'm doing great! Nigga", "Thanks for sharing",
    "Goodbye!", "I am a simple chatbot created using Python.",
    "I was created by God Suleimani.",
    "I can chat with you and answer simple questions nigga."
]

vectorizer = CountVectorizer().fit(questions)
question_vectors = vectorizer.transform(questions)

def chatbot_response(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, question_vectors).flatten()

    index = similarity.argmax()
    if similarity[index] == 0:
        return "Sorry, I don't understand."
    return answers[index]

print("Chatbot is ready! Type 'exit' to stop.")

while True:
    user_input = input("You: ").lower()
    if user_input == "exit":
        print("Chatbot: Goodbye!")
        break
    print("Chatbot:", chatbot_response(user_input))