import os
import requests
import streamlit as st
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Function to extract text from a webpage
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract the main text from the page (can be refined based on specific websites)
        text = soup.get_text()
        return text.strip()
    except Exception as e:
        return str(e)

# Function to summarize text and prepare for Q&A
def summarize_and_answer(text):
    # Initialize Google LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=google_api_key
    )

    # Summarization
    messages_summary = [
        ("system", "You're a helpful assistant that summarizes websites."),
        ("human", text),
    ]
    summary = llm.invoke(messages_summary).content

    # Prepare for Q&A with memory
    memory = ConversationBufferMemory()
    conversation = ConversationChain(
        llm=llm,
        verbose=False,
        memory=memory
    )

    # Save the context for Q&A
    memory.save_context({"input": text}, {"output": ""})

    return summary, conversation

# Main function for the Streamlit app
def main():
    # Set up page config
    st.set_page_config(page_title="Website Summarizer and Q&A", layout="wide")

    # Add a title and description
    st.title("🌐 Website Summarizer and Q&A")
    st.write("Enter a website URL to summarize the content and ask questions about it.")

    # URL input field
    st.sidebar.header("Website Input")
    website_url = st.sidebar.text_input("Enter Website URL")

    if website_url:
        with st.spinner(f"Fetching and summarizing content from {website_url}..."):
            # Extract text from the website
            text = extract_text_from_url(website_url)

            if text:
                # Summarize the text and set up the Q&A conversation
                summary, conversation = summarize_and_answer(text)

                # Display the summary in a card format
                st.subheader("Website Summary:")
                st.markdown(
                    f"<div style='border: 1px solid #e0e0e0; border-radius: 5px; padding: 15px; background-color: #f9f9f9;'>{summary}</div>",
                    unsafe_allow_html=True
                )

                # Initialize session state for chat messages
                if "web_messages" not in st.session_state:
                    st.session_state.web_messages = []

                # Q&A section
                st.subheader("💬 Ask Questions About the Website")
                web_messages = st.container()
                with web_messages:
                    # Display chat messages
                    for message in st.session_state.web_messages:
                        if message['role'] == 'user':
                            st.chat_message("user").write(message['content'])
                        else:
                            st.chat_message("assistant").write(message['content'])

                    # User input for Q&A
                    if prompt := st.chat_input("Ask a question about the website"):
                        st.session_state.web_messages.append({"role": "user", "content": prompt})
                        st.chat_message("user").write(prompt)

                        with st.spinner("Generating response..."):
                            response = conversation.predict(input=prompt)
                            if response.strip() == "":
                                response = "The context is not provided on the website."

                        st.session_state.web_messages.append({"role": "assistant", "content": response})
                        st.chat_message("assistant").write(response)

if __name__ == "__main__":
    main()
