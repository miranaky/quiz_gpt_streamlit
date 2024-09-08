import json
import os
from operator import itemgetter

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import WikipediaRetriever
from langchain.schema.runnable import RunnableLambda
from langchain.text_splitter import CharacterTextSplitter

folders = [
    "./.cache",
    "./.cache/files",
]
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)


st.set_page_config(page_title="QuizGPT", page_icon="🧠")
st.title("QuizGPT")

function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}
openai_api_key = None


def load_llm(openai_api_key):
    return ChatOpenAI(
        temperature=0.1,
        model="gpt-4o-mini",
        api_key=openai_api_key,
    ).bind(
        function_call={
            "name": "create_quiz",
        },
        functions=[
            function,
        ],
    )


level_dict = {
    "begginers": "Create 10 questions that test basic understanding of the text. Each question should have 4 answers, three of them must be incorrect, and one should be correct. These questions should focus on simple facts or concepts.",
    "advanced": "Create 10 questions that test in-depth understanding of the text. Each question should have 4 answers, three of them must be incorrect, and one should be correct. These questions should require critical thinking or a deep grasp of the subject.",
}

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """ 
    You are a helpful assistant that is role playing as a teacher.

    Based ONLY on the following context make 10 (TEN) questions to test the user's knowledge about the text.
    
    {level}
        
    Context:{context}
    """,
        ),
    ]
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@st.cache_data(
    show_spinner="Embedding file...",
)
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = UnstructuredFileLoader(file_path)

    docs = loader.load_and_split(text_splitter=splitter)

    return docs


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriver = WikipediaRetriever(top_k_results=1)
    docs = retriver.get_relevant_documents(term)
    return docs


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic, level):
    chain = (
        {
            "context": itemgetter("context") | RunnableLambda(format_docs),
            "level": itemgetter("level"),
        }
        | prompt
        | llm
    )
    response = chain.invoke({"context": _docs, "level": level})
    json_data = response.additional_kwargs["function_call"]["arguments"]
    return json.loads(json_data)


with st.sidebar:
    docs = None
    topic = None

    st.write("github repo: https://github.com/miranaky/quiz_gpt_streamlit")
    openai_api_key = st.text_input("OpenAI API Key", type="password")

    choice = st.selectbox(
        "Choose what you want to use",
        (
            "Wikipedia",
            "File",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a file(.txt, .pdf, .docx)",
            type=["txt", "pdf", "docx"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)
    level = st.selectbox(
        "Choose the level of the questions", ("begginers", "advanced"), index=0
    )


if openai_api_key is None or not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.

    Enter your OpenAI API Key in the sidebar. 

    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    llm = load_llm(openai_api_key)
    response = run_quiz_chain(
        docs,
        topic=topic if topic else file.name,
        level=level_dict[level],
    )
    correct = 0
    with st.form("questions_form"):
        for idx, question in enumerate(response["questions"], start=1):
            st.write(f"{idx}.", question["question"])
            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
                correct += 1
            elif value is not None:
                st.error("Wrong!")
        button = st.form_submit_button()

    if button:
        st.write(f"Your score is {correct}/10")
        if correct > 0 and correct != 10:
            st.error("Try again!")
        elif correct == 10:
            st.balloons()
            st.success("Congratulations! You got all the answers right!")
