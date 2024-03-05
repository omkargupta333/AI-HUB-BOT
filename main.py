import os
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st
from streamlit_option_menu import option_menu
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import FAISS
import pygame
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns
import emojis 
import wordcloud
from gtts import gTTS
import streamlit as st
import speech_recognition as sr
from googletrans import LANGUAGES, Translator
from langchain.llms import GooglePalm
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from gemini_utility import (
    load_gemini_pro_model,
    gemini_pro_response,
    gemini_pro_vision_response,
    embeddings_model_response,
)
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

working_dir = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Your AI Bot",
    page_icon="🧠",
    layout="centered",
)

with st.sidebar:
    selected = option_menu(
        "Your AI Bot",
        [
            "ChatBot",
            "Image Captioning",
            "Embed text",
            "PDF Analysis",
            "WhatsApp Chat Analysis",
            "Real time Language Translator",
            "Ask me anything",
        ],
        menu_icon="robot",
        icons=[
            "chat-dots-fill",
            "image-fill",
            "textarea-t",
            "clipboard2-data-fill",
            "whatsapp",
            "megaphone",
            "patch-question-fill",
        ],
        default_index=0,
    )

# Function to translate roles between Gemini-Pro and Streamlit terminology
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role
    


# chatbot page
if selected == "ChatBot":
    model = load_gemini_pro_model()

    # Initialize chat session in Streamlit if not already present
    if "chat_session" not in st.session_state:  # Renamed for clarity
        st.session_state.chat_session = model.start_chat(history=[])

    # Display the chatbot's title on the page
    st.title("🤖 ChatBot")

    # Display the chat history
    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    # Input field for user's message
    user_prompt = st.chat_input("Ask Gemini-Pro...")  # Renamed for clarity
    if user_prompt:
        # Add user's message to chat and display it
        st.chat_message("user").markdown(user_prompt)

        # Send user's message to Gemini-Pro and get the response
        gemini_response = st.session_state.chat_session.send_message(
            user_prompt
        )  # Renamed for clarity

        # Display Gemini-Pro's response
        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)

# Image captioning page
if selected == "Image Captioning":

    st.title("📷 Snap Narrate")

    uploaded_image = st.file_uploader(
        "Upload an image...", type=["jpg", "jpeg", "png"]
    )

    user_prompt = st.text_input(
        "Enter your caption prompt:", "write a short caption for this image"
    )

    if st.button("Analysis"):
        if uploaded_image is not None:
            image = Image.open(uploaded_image)

            col1, col2 = st.columns(2)

            with col1:
                resized_img = image.resize((800, 500))
                st.image(resized_img)

            # get the caption of the image from the gemini-pro-vision LLM
            caption = gemini_pro_vision_response(user_prompt, image)

            with col2:
                st.info(caption)
        else:
            st.warning("Please upload an image before generating a caption.")

# text embedding model
if selected == "Embed text":

    st.title("🔡 Embed Text")

    # text box to enter prompt
    user_prompt = st.text_area(
        label="", placeholder="Enter the text to get embeddings"
    )

    if st.button("Get Response"):
        response = embeddings_model_response(user_prompt)
        st.markdown(response)


# PDF reader
if selected == "PDF Analysis":# Function definitions
    def get_pdf_text(pdf_docs):
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def get_text_chunks(text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        chunks = text_splitter.split_text(text)
        return chunks

    def get_vector_store(text_chunks):
        embeddings = GooglePalmEmbeddings()
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store

    def get_conversational_chain(vector_store):
        llm = GooglePalm()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
        return conversation_chain

    def user_input(user_question):
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chatHistory = response['chat_history']
        for i, message in enumerate(st.session_state.chatHistory):
            if i % 2 == 0:
                st.write("Human: ", message.content)
            else:
                st.write("Bot: ", message.content)
    models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]

    # Streamlit app
    st.title("🔍PDF READER")

    # Model selection
    selected_model = st.selectbox("Select a Generative Model", models)

    # Multiple PDF-based chatbot UI
    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
    if st.button("Process PDFs"):
        with st.spinner("Processing PDFs"):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            vector_store = get_vector_store(text_chunks)
            st.session_state.conversation = get_conversational_chain(vector_store)

    # User input for PDF-based chatbot
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)    
        
        
        
# Real Time Speech Translator
if selected == "Real time Language Translator":
    isTranslateOn = False

    translator = Translator() # Initialize the translator module.
    pygame.mixer.init()  # Initialize the mixer module.

     # Create a mapping between language names and language codes
    language_mapping = {name: code for code, name in LANGUAGES.items()}

    def get_language_code(language_name):
         return language_mapping.get(language_name, language_name)

    def translator_function(spoken_text, from_language, to_language):
         return translator.translate(spoken_text, src='{}'.format(from_language), dest='{}'.format(to_language))

    def text_to_voice(text_data, to_language):
         myobj = gTTS(text=text_data, lang='{}'.format(to_language), slow=False)
         myobj.save("cache_file.mp3")
         audio = pygame.mixer.Sound("cache_file.mp3")  # Load a sound.
         audio.play()
         os.remove("cache_file.mp3")

    def main_process(output_placeholder, from_language, to_language):
    
         global isTranslateOn
    
         while isTranslateOn:

             rec = sr.Recognizer()
             with sr.Microphone() as source:
                 output_placeholder.text("Listening...")
                 rec.pause_threshold = 1
                 audio = rec.listen(source, phrase_time_limit=10)
        
             try:
                output_placeholder.text("Processing...")
                spoken_text = rec.recognize_google(audio, language='{}'.format(from_language))
            
                output_placeholder.text("Translating...")
                translated_text = translator_function(spoken_text, from_language, to_language)

                text_to_voice(translated_text.text, to_language)
    
             except Exception as e:
                 print(e)

    # UI layout
    st.title("Language Translator")

    # Dropdowns for selecting languages
    from_language_name = st.selectbox("Select Source Language:", list(LANGUAGES.values()))
    to_language_name = st.selectbox("Select Target Language:", list(LANGUAGES.values()))

    # Convert language names to language codes
    from_language = get_language_code(from_language_name)
    to_language = get_language_code(to_language_name)

    # Button to trigger translation
    start_button = st.button("Start")
    stop_button = st.button("Stop")

# Check if "Start" button is clicked
    if start_button:
        if not isTranslateOn:
            isTranslateOn = True
            output_placeholder = st.empty()
            main_process(output_placeholder, from_language, to_language)

    # Check if "Stop" button is clicked
    if stop_button:
        isTranslateOn = False
        
        
        
# WhatsApp Chat Analysis
if selected =="WhatsApp Chat Analysis":
    st.info("Upload a WhatsApp chat file to analyze. Please make sure the file is in text format.")

# Sidebar
st.sidebar.title("Whatsapp Chat Analyzer")

# File uploader
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    # fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):

        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        # Monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Activity map
        st.title('Activity Map')
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # Finding the busiest users in the group (Group level)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # Most common words
        most_common_df = helper.most_common_words(selected_user, df)

        fig, ax = plt.subplots()

        ax.bar(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')

        st.title('Most common words')
        st.pyplot(fig)
        
        
        # Emoji Analysis
        
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("Emoji Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(emoji_df)
        with col2:
            if not emoji_df.empty:
                for index, row in emoji_df.head().iterrows():
                    emoji_name = row['Emoji']
                    emoji_count = row['Count']
            else:
                st.write("No emojis found for this user.")
        

        # Sentiment distribution
        if selected_user != 'Overall':
            sentiment_counts = df[df['user'] == selected_user]['sentiment'].apply(
                lambda x: 'positive' if x > 0 else ('neutral' if x == 0 else 'negative')).value_counts()
        else:
            sentiment_counts = df['sentiment'].apply(
                lambda x: 'positive' if x > 0 else ('neutral' if x == 0 else 'negative')).value_counts()

        sentiment_labels = sentiment_counts.index.tolist()
        sentiment_values = sentiment_counts.values.tolist()

        colors = ['green', 'orange', 'red']

        st.title("Sentiment Analysis")
        st.subheader("Sentiment Distribution")

        fig, ax = plt.subplots()
        ax.pie(sentiment_values, labels=sentiment_labels, colors=colors, autopct='%1.1f%%', startangle=90)

        # Draw a circle to create a donut chart
        centre_circle = plt.Circle((0, 0), 0.7, color='white', fc='white', linewidth=1.25)
        fig.gca().add_artist(centre_circle)

        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)
    

        
        
        
# text embedding model
if selected == "Ask me anything":

    st.title("❓ Ask me a question")

    # text box to enter prompt
    user_prompt = st.text_area(label="", placeholder="Ask me anything...")

    if st.button("Get Response"):
        response = gemini_pro_response(user_prompt)
        st.markdown(response)