import streamlit as st
import LLMClass as llm
import json
import os
import html 
from langchain_chroma import Chroma
import chromadb

CHAT_FILE = "chat_history.json"
# # --- Load or Initialize Chat History ---
def load_chat_history():
    if os.path.exists(CHAT_FILE):
        with open(CHAT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_chat_history(history):
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# --- Session State for In-Memory Chat ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()


st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://storage.googleapis.com/kagglesdsdata/datasets/7125881/11380797/image%20%281%29.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20250416%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250416T182726Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=7d255d30ae43d0ff894b3b5392f29b1dd184ae1b0c996dff5ffda5e64da7c270e7d3f8558af472d4a95b284127665e1639a6d85e26cbcc856434c8eeaf8bf91f738ee270afe9ce822b17876765a71c78e68f59faf1444259f84ce227cb7aae8ec07a3d0dc0dc13b57e0984c85dda25f3f825e5a3c49af460a5580e42322a4c127b7344b05833920c305d29ccc5ce25d3bc4077ba35e81f0a5f99ed71e5a2fb2336d41b1fa704321cc37edbc791921659ca50922f49271344fc449006becd40d08a038c5e46f6312314fb0a2943535e917df6b9770ff199052bc039ad2c33c4c9c1b108212e2561919a3a94c7c6083ec0244c363c7e532e101e29f0d58dc35823");
        background-repeat: no-repeat;
        background-position: top center;
        background-size: cover;
        
        /* Width x Height */
    }}
    </style>
    """,
    unsafe_allow_html=True
)

if "load_db" not in st.session_state:
    st.session_state.load_db = False


GOOGLE_API_KEY = "AIzaSyD9QQd80bEbw4ZIjd5Q5RhZZ1Vf3EMDVKY"
ASSEMBLY_AI_API = "a780dffafa21462aa994f73cda689715"
# #https://img.freepik.com/free-photo/vivid-colored-transparent-autumn-leaf_23-2148239739.jpg?ga=GA1.1.1564546954.1730728324&semt=ais_hybrid&w=740
with st.sidebar:

    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-image: url('https://img.freepik.com/free-photo/vivid-colored-transparent-autumn-leaf_23-2148239739.jpg?ga=GA1.1.1564546954.1730728324&semt=ais_hybrid&w=740');
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }
    </style>
    """, unsafe_allow_html=True)

    file_path = st.text_input("file Path : ")
    kind_of_input = st.selectbox("Type : ", ["PDF file", "Youtube Video", "Voice"])
    store_button = st.button("Save")
    if kind_of_input == "PDF file":
        kind_of_input = "pdf"
    elif kind_of_input == "Youtube Video":
        kind_of_input = "vid"
    else:
        kind_of_input = "voc"
    if store_button and file_path:
        try:
            llm_obj = llm.LangChainLLMByGoogle(file_path=file_path, api_key=GOOGLE_API_KEY, assembly_api=ASSEMBLY_AI_API, kind=kind_of_input)
            llm_obj.add(file_path, kind=kind_of_input)
            st.session_state.load_db = True
            st.success("Done succesfully!")
            
            
            
        except Exception as e:
            st.error(f"File path is wrong {e}")
    else:
        if store_button:
            st.error("Enter Valid file path")

query_asked = st.text_input("Powered by Gemini","")
Send_button = st.button("Send")

if st.session_state.load_db and Send_button:

    llm_obj = llm.LangChainLLMByGoogle(file_path=None, api_key=GOOGLE_API_KEY, assembly_api=ASSEMBLY_AI_API, kind="pdf")
    llm_obj.load_chromadb(persist="chroma")
    ai_answer = llm_obj.invoke(query_asked)

    
    st.session_state.chat_history.append({"role": "user", "message": query_asked})
    st.session_state.chat_history.append({"role": "ai", "message": ai_answer})
    save_chat_history(st.session_state.chat_history)
elif Send_button:
    st.error("No stored data yet!!")
    
for msg in st.session_state.chat_history:
    is_user = msg["role"] == "user"
    alignment = "flex-end" if is_user else "flex-start"
    bg_color = "#1C2833" if is_user else "#424949"

    safe_msg = html.escape(msg["message"]).replace("\n", "<br>")

    st.markdown(
        f"""
        <div style='display: flex; justify-content: {alignment}; margin-bottom: 10px;'>
            <div style='background-color: {bg_color}; padding: 10px 15px; border-radius: 15px; max-width: 70%;'>
                {safe_msg}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# #C:\Users\Mohsen\Downloads\gulati20_interspeech.pdf
#https://www.poetryoutloud.org/wp-content/uploads/sites/2/2019/07/01-Track-01.mp3
# llm_obj = llm.LangChainLLMByGoogle(file_path=None, api_key=GOOGLE_API_KEY, assembly_api=ASSEMBLY_AI_API, kind="pdf")
# llm_obj.load_chromadb(persist="chroma")
# chromadb = Chroma(persist_directory='chroma', collection_name="Paper_summary", embedding_function=llm.EmbeddingFunction())
# ai_answer = chromadb.similarity_search("What did poetry said about silence sea", k=1)
# print(ai_answer[0].page_content)