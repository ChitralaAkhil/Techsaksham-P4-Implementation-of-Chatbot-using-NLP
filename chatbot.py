import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

file_path = os.path.abspath(r"C:\Users\chara\OneDrive\Documents\Python Scripts\intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
        
counter = 0

def main():
    global counter
    st.title("Education ChatBot")

    menu = ["Home", "History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome to the Education chatbot. ")
        st.write("Please type a message and press Enter to start the conversation.")

        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:

            user_input_str = str(user_input)

            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    elif choice == "History":
        st.header("History")
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  
            for row in csv_reader:
                st.text(f"User👨🏻‍💻: {row[0]}")
                st.text(f"Chatbot🤖: {row[1]}")
                st.text(f"Timestamp🕰️: {row[2]}")
                st.markdown("---")

    elif choice == "About":
        st.write("I’m an educational chatbot here to help with your studies! Whether you need help with homework, study tips, or explanations in subjects like math, science, history, and more, I'm here to assist. My goal is to make learning easier and more enjoyable by offering support and resources tailored to your needs.")
        st.subheader("Project Overview:")

        st.write()

        st.subheader("Dataset:")

        st.write()

        st.subheader("Streamlit Chatbot Interface:")

        st.write("The Educational Chatbot, built with Streamlit, offers an interactive interface where users can input text and receive personalized responses in a chat window. Powered by a trained model, it provides academic support, study tips, explanations, and helps with homework across various subjects to enhance learning and academic performance.")

        st.subheader("Conclusion:")

        st.write("In conclusion, the Educational Chatbot offers an engaging and personalized learning experience, making it easier for users to get academic support, study tips, and explanations across various subjects. With its interactive interface and smart responses, it ensures that help with homework, exam preparation, and learning is always just a chat away, making education more accessible and enjoyable.")

if __name__ == '__main__':
    main()