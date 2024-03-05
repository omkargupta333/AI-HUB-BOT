# import os
# from PIL import Image
# from dotenv import load_dotenv
# import google.generativeai as genai
# import streamlit as st
# from streamlit_option_menu import option_menu
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import GooglePalmEmbeddings
# from langchain.vectorstores import FAISS
# import pygame
# from gtts import gTTS
# import streamlit as st
# import speech_recognition as sr
# from googletrans import LANGUAGES, Translator
# from langchain.llms import GooglePalm
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from gemini_utility import (
#     load_gemini_pro_model,
#     gemini_pro_response,
#     gemini_pro_vision_response,
#     embeddings_model_response,
# )
# load_dotenv()
# GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
# genai.configure(api_key=GOOGLE_API_KEY)

# working_dir = os.path.dirname(os.path.abspath(__file__))

# st.set_page_config(
#     page_title="Your AI Bot",
#     page_icon="🧠",
#     layout="centered",
# )

# with st.sidebar:
#     selected = option_menu(
#         "Your AI Bot",
#         [
#             "ChatBot",
#             "Image Captioning",
#             "Embed text",
#             "PDF Analysis",
#             "Chat Analysis",
#             "Real time Language Translator",
#             "Ask me anything",
#         ],
#         menu_icon="robot",
#         icons=[
#             "chat-dots-fill",
#             "image-fill",
#             "textarea-t",
#             "clipboard2-data-fill",
#             "whatsapp",
#             "megaphone",
#             "patch-question-fill",
#         ],
#         default_index=0,
#     )

# # Function to translate roles between Gemini-Pro and Streamlit terminology
# def translate_role_for_streamlit(user_role):
#     if user_role == "model":
#         return "assistant"
#     else:
#         return user_role


# # chatbot page
# if selected == "ChatBot":
#     model = load_gemini_pro_model()

#     # Initialize chat session in Streamlit if not already present
#     if "chat_session" not in st.session_state:  # Renamed for clarity
#         st.session_state.chat_session = model.start_chat(history=[])

#     # Display the chatbot's title on the page
#     st.title("🤖 ChatBot")

#     # Display the chat history
#     for message in st.session_state.chat_session.history:
#         with st.chat_message(translate_role_for_streamlit(message.role)):
#             st.markdown(message.parts[0].text)

#     # Input field for user's message
#     user_prompt = st.chat_input("Ask Gemini-Pro...")  # Renamed for clarity
#     if user_prompt:
#         # Add user's message to chat and display it
#         st.chat_message("user").markdown(user_prompt)

#         # Send user's message to Gemini-Pro and get the response
#         gemini_response = st.session_state.chat_session.send_message(
#             user_prompt
#         )  # Renamed for clarity

#         # Display Gemini-Pro's response
#         with st.chat_message("assistant"):
#             st.markdown(gemini_response.text)

# # Image captioning page
# if selected == "Image Captioning":

#     st.title("📷 Snap Narrate")

#     uploaded_image = st.file_uploader(
#         "Upload an image...", type=["jpg", "jpeg", "png"]
#     )

#     user_prompt = st.text_input(
#         "Enter your caption prompt:", "write a short caption for this image"
#     )

#     if st.button("Analysis"):
#         if uploaded_image is not None:
#             image = Image.open(uploaded_image)

#             col1, col2 = st.columns(2)

#             with col1:
#                 resized_img = image.resize((800, 500))
#                 st.image(resized_img)

#             # get the caption of the image from the gemini-pro-vision LLM
#             caption = gemini_pro_vision_response(user_prompt, image)

#             with col2:
#                 st.info(caption)
#         else:
#             st.warning("Please upload an image before generating a caption.")

# # text embedding model
# if selected == "Embed text":

#     st.title("🔡 Embed Text")

#     # text box to enter prompt
#     user_prompt = st.text_area(
#         label="", placeholder="Enter the text to get embeddings"
#     )

#     if st.button("Get Response"):
#         response = embeddings_model_response(user_prompt)
#         st.markdown(response)


# # PDF reader
# if selected == "PDF Analysis":# Function definitions
#     def get_pdf_text(pdf_docs):
#         text = ""
#         for pdf in pdf_docs:
#             pdf_reader = PdfReader(pdf)
#             for page in pdf_reader.pages:
#                 text += page.extract_text()
#         return text

#     def get_text_chunks(text):
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
#         chunks = text_splitter.split_text(text)
#         return chunks

#     def get_vector_store(text_chunks):
#         embeddings = GooglePalmEmbeddings()
#         vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#         return vector_store

#     def get_conversational_chain(vector_store):
#         llm = GooglePalm()
#         memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#         conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
#         return conversation_chain

#     def user_input(user_question):
#         response = st.session_state.conversation({'question': user_question})
#         st.session_state.chatHistory = response['chat_history']
#         for i, message in enumerate(st.session_state.chatHistory):
#             if i % 2 == 0:
#                 st.write("Human: ", message.content)
#             else:
#                 st.write("Bot: ", message.content)
#     models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]

#     # Streamlit app
#     st.title("🔍PDF READER")

#     # Model selection
#     selected_model = st.selectbox("Select a Generative Model", models)

#     # Multiple PDF-based chatbot UI
#     pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
#     if st.button("Process PDFs"):
#         with st.spinner("Processing PDFs"):
#             raw_text = get_pdf_text(pdf_docs)
#             text_chunks = get_text_chunks(raw_text)
#             vector_store = get_vector_store(text_chunks)
#             st.session_state.conversation = get_conversational_chain(vector_store)

#     # User input for PDF-based chatbot
#     user_question = st.text_input("Ask a Question from the PDF Files")
#     if user_question:
#         user_input(user_question)    
        
        
        
# # Real Time Speech Translator
# if selected == "Real time Language Translator":
#     isTranslateOn = False

#     translator = Translator() # Initialize the translator module.
#     pygame.mixer.init()  # Initialize the mixer module.

#      # Create a mapping between language names and language codes
#     language_mapping = {name: code for code, name in LANGUAGES.items()}

#     def get_language_code(language_name):
#          return language_mapping.get(language_name, language_name)

#     def translator_function(spoken_text, from_language, to_language):
#          return translator.translate(spoken_text, src='{}'.format(from_language), dest='{}'.format(to_language))

#     def text_to_voice(text_data, to_language):
#          myobj = gTTS(text=text_data, lang='{}'.format(to_language), slow=False)
#          myobj.save("cache_file.mp3")
#          audio = pygame.mixer.Sound("cache_file.mp3")  # Load a sound.
#          audio.play()
#          os.remove("cache_file.mp3")

#     def main_process(output_placeholder, from_language, to_language):
    
#          global isTranslateOn
    
#          while isTranslateOn:

#              rec = sr.Recognizer()
#              with sr.Microphone() as source:
#                  output_placeholder.text("Listening...")
#                  rec.pause_threshold = 1
#                  audio = rec.listen(source, phrase_time_limit=10)
        
#              try:
#                 output_placeholder.text("Processing...")
#                 spoken_text = rec.recognize_google(audio, language='{}'.format(from_language))
            
#                 output_placeholder.text("Translating...")
#                 translated_text = translator_function(spoken_text, from_language, to_language)

#                 text_to_voice(translated_text.text, to_language)
    
#              except Exception as e:
#                  print(e)

#     # UI layout
#     st.title("Language Translator")

#     # Dropdowns for selecting languages
#     from_language_name = st.selectbox("Select Source Language:", list(LANGUAGES.values()))
#     to_language_name = st.selectbox("Select Target Language:", list(LANGUAGES.values()))

#     # Convert language names to language codes
#     from_language = get_language_code(from_language_name)
#     to_language = get_language_code(to_language_name)

#     # Button to trigger translation
#     start_button = st.button("Start")
#     stop_button = st.button("Stop")

# # Check if "Start" button is clicked
#     if start_button:
#         if not isTranslateOn:
#             isTranslateOn = True
#             output_placeholder = st.empty()
#             main_process(output_placeholder, from_language, to_language)

#     # Check if "Stop" button is clicked
#     if stop_button:
#         isTranslateOn = False
        
        
        
# # text embedding model
# if selected == "Ask me anything":

#     st.title("❓ Ask me a question")

#     # text box to enter prompt
#     user_prompt = st.text_area(label="", placeholder="Ask me anything...")

#     if st.button("Get Response"):
#         response = gemini_pro_response(user_prompt)
#         st.markdown(response)