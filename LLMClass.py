
import warnings
from unstructured.partition.pdf import partition_pdf
import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from google.api_core import retry
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from google.api_core import retry
from chromadb import EmbeddingFunction
from google import genai
from google.genai import types
from typing import List
from uuid import uuid4
from langchain_chroma import Chroma
from langchain_core.documents import Document
import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from google.api_core import retry
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
import yt_dlp
import assemblyai as aai
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
import enum
from dotenv import load_dotenv


warnings.filterwarnings('ignore')


load_dotenv()
GOOGLE_API_KEY = your_api
ASSEMBLY_AI_API = your_api
# @title
prompts= """ You will be provided with a query and relevant documents retrieved from a database. Your task is to answer the query creatively and insightfully based on the information within the documents.  Do not copy directly from the documents. Synthesize the information and present it in a novel and engaging way.

User Query: {query}

Relevant Documents:
{documents}

Few-Shot Examples:

Example 1:
User Query: What are the benefits of using solar energy?

Relevant Documents:
Document 1: Solar energy is a renewable resource... reduces reliance on fossil fuels... environmentally friendly.
Document 2: Solar panels can reduce electricity bills... require minimal maintenance.

Response: Harnessing the sun's power offers a plethora of advantages. Not only is it a clean and sustainable energy source, reducing our carbon footprint and dependence on finite fossil fuels, but it can also lead to significant long-term savings by slashing electricity costs.  Furthermore, solar panels are remarkably low-maintenance, allowing you to enjoy clean energy with minimal effort.

Example 2:
User Query: How does artificial intelligence impact the job market?

Relevant Documents:
Document 1: AI automates repetitive tasks... potential job displacement in certain sectors.
Document 2: AI creates new job opportunities... data scientists, AI specialists.

Response: The rise of artificial intelligence presents a complex and evolving impact on the job market. While concerns about job displacement due to automation are valid, particularly in sectors reliant on repetitive tasks, AI is simultaneously creating exciting new opportunities.  The demand for skilled professionals in fields like data science and AI specialization is rapidly growing, reshaping the employment landscape.

Example 3:
User Query: What are the health benefits of regular exercise?

Relevant Documents:
Document 1: Exercise improves cardiovascular health... reduces risk of heart disease.
Document 2: Regular physical activity strengthens muscles and bones... improves mood and reduces stress.

Response: Engaging in regular exercise is a cornerstone of a healthy lifestyle.  From bolstering your heart and reducing the risk of cardiovascular diseases to strengthening your musculoskeletal system, the benefits are undeniable.  Moreover, exercise acts as a natural mood enhancer, combating stress and promoting overall well-being.


Your Response:"""



prompts2 = """You will be provided with the following:
Documents: {documents2}
Query: {query}

Instructions:
1. Carefully analyze the provided documents and understand their content.
2. Based on the information in the documents, answer the query in a comprehensive and informative way.
3. Do not simply copy and paste text from the documents.  Synthesize the information and present it in your own words.
4. Ensure your response is fluent, readable, and easy to understand.

Few-Shot Examples:

Example 1:
Documents:
Document 1: "The capital of France is Paris."
Document 2: "Paris is known for its iconic Eiffel Tower."
Query: "What is the capital of France known for?"
Response: "Paris, the capital of France, is renowned for its iconic Eiffel Tower."

Example 2:
Documents:
Document 1: "Albert Einstein developed the theory of relativity."
Document 2: "The theory of relativity revolutionized our understanding of space and time."
Query: "Who developed the theory of relativity and what was its impact?"
Response: "Albert Einstein developed the theory of relativity, which revolutionized our understanding of space and time."

Example 3:
Documents:
Document 1: "Photosynthesis is the process by which plants convert light energy into chemical energy."
Document 2: "Chlorophyll is essential for photosynthesis."
Query: "What is the role of chlorophyll in photosynthesis?"
Response: "Chlorophyll plays a crucial role in photosynthesis, the process by which plants convert light energy into chemical energy."


Your response to the query:"""

evaluation_prompt = """ You will be provided with:
Query: {query}
Answer A: {answer_a}
Answer B: {answer_b}

Instructions:
1. Carefully analyze the query and both answers (A and B).
2. Determine which answer is a more relevant and comprehensive response to the query.
3. If answer A is better, output "A" followed by a brief explanation of why it is better.
4. If answer B is better, output "B" followed by a brief explanation of why it is better.
5. If both answers are equally good or essentially identical, output "SAME" followed by a brief explanation.

Example:
Query: What are the benefits of regular exercise?
Answer A: Regular exercise has numerous benefits, including improved cardiovascular health, weight management, stronger bones and muscles, and stress reduction.
Answer B: Exercise is good.

Output: A. Answer A provides a more comprehensive list of benefits, while Answer B is too vague.


Your Output:"""

prompt_text = """
You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.

Respond only with the summary, no additionnal comment.
Do not start your message by saying "Here is a summary" or anything like that.
Just give the summary as it is.

Table or text chunk: {element}

"""




def return_prompt():
  return prompts, prompts2



class Enum_answers(enum.Enum):
  answer1 = "A"
  answer2 = "B"
  query = "SAME"


client = genai.Client(api_key=GOOGLE_API_KEY)
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})


class EmbeddingFunction(EmbeddingFunction):


  @retry.Retry(predicate=is_retriable)
  def embed_query(self, text : str) -> List[float]:
    task = "retrieval_query"

    response = client.models.embed_content(
        model="text-embedding-004",
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task
        )
    )

    return [e.values for e in response.embeddings][0]


  @retry.Retry(predicate=is_retriable)
  def embed_documents(self, text : List[str]) -> List[List[float]]:
    task = "retrieval_document"

    response = client.models.embed_content(
        model="text-embedding-004",
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task
        )
    )


    return [e.values for e in response.embeddings]

class LangChainLLMByGoogle:


  def __init__(self, file_path, api_key, kind='pdf', assembly_api=None):
    self.file_path = file_path
    self.api_key = api_key
    self.assembly_api = assembly_api
    self.google = GoogleGenerativeAI(model='gemini-1.5-flash', google_api_key=self.api_key)
    self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",google_api_key=self.api_key)
    persistent_client = chromadb.PersistentClient()

    self.chroma = Chroma(client=persistent_client, collection_name='Paper_summary', embedding_function=EmbeddingFunction())
    # self.add(file_path, kind)

    #-------------------------------------------------Model Creation Part-----------------------------------------------------
                                       #--------------------Flas Model-------------------------
    prompt_template_flash = ChatPromptTemplate.from_template(prompts)
    chroma_runnable = RunnableLambda(lambda x: self.chroma.similarity_search(x, k=4))
    prompt_inputs_flash = RunnableParallel({
        "documents": chroma_runnable,
        "query": RunnablePassthrough()
    })

    google_model_flash = ChatGoogleGenerativeAI(model='gemini-2.0-flash', api_key=api_key)

    self.rag_chain_flash = prompt_inputs_flash | prompt_template_flash | google_model_flash | StrOutputParser()
                                        #--------------------Pro Model-------------------------
    prompt_template_pro = ChatPromptTemplate.from_template(prompts2)
    chroma_runnable_pro = RunnableLambda(lambda x: self.chroma.similarity_search(x, k=4))
    prompt_inputs_pro = RunnableParallel({
        "documents2": chroma_runnable_pro,
        "query": RunnablePassthrough()
    })

    google_model_pro = ChatGoogleGenerativeAI(model='gemini-1.5-pro', api_key=api_key)

    self.rag_chain_pro = prompt_inputs_pro | prompt_template_pro | google_model_pro | StrOutputParser()


  def invoke(self, query, flash=True):
    if flash:
      return self.rag_chain_flash.invoke(query)
    else:
      return self.rag_chain_pro.invoke(query)


  def add(self, file_path, kind='pdf'):
    self.file_path = file_path
    if kind=="pdf":
      self.__Parse_pdf__()
    elif kind=="vid":
      if self.assembly_api is None : raise ValueError("Fill the Assembly API first!!")
      video_Url = self.file_path

      ydl_opts = {
          'format': 'm4a/bestaudio/best',  # The best audio version in m4a format
          'outtmpl': 'context.m4a',
          "cookiesfrombrowser": ("chrome",),
          'postprocessors': [{  # Extract audio using ffmpeg
              'key': 'FFmpegExtractAudio',
              'preferredcodec': 'm4a',
          }]
      }


      with yt_dlp.YoutubeDL(ydl_opts) as ydl:
          error_code = ydl.download(video_Url)
      self.__Parse_video_or_audio__("./context.m4a")
    elif kind =="voc":
      self.__Parse_video_or_audio__(file_path)
    else:
      raise ValueError("Invalid kind")

  def load_chromadb(self, persist):
     self.chroma = Chroma(persist_directory=persist, collection_name='Paper_summary', embedding_function=EmbeddingFunction())
     

  def evaluate(self, query):
    answer_1, answer_2, Query = self.__generate_output_via_models__(query)
    target = self.__final_score__(Query, answer_1, answer_2)
    if target[0] == "SAME":
      print("both are same")
      print(target[1])
      return
    else:
      if target[0] == "A":
        name = "gemini 1.5 Pro"
      else:
        name = "gemini 2.0 flash"
      print(f"High evaluation is for {name}")
      print(target[1])
    return


  @retry.Retry(predicate=is_retriable)
  def __final_score__(self, query, answerA, answerB):
    chat = client.chats.create(model='gemini-2.0-flash')

    # Generate the full text response.
    response = chat.send_message(
        message=evaluation_prompt.format(
            query=query,
            answer_a=answerA,
            answer_b=answerB)
    )
    verbose_eval = response.text

    structured_output_config = types.GenerateContentConfig(
        response_mime_type="text/x.enum",
        response_schema=Enum_answers,
    )
    response = chat.send_message(
        message="Convert the final score.",
        config=structured_output_config,
    )
    structured_eval = response.parsed

    return structured_eval.value, verbose_eval


  def __return_prompt__(self):
    return prompts, prompts2

  def __generate_output_via_models__(self, query):
    google_model_pro = ChatGoogleGenerativeAI(model='gemini-1.5-pro', api_key=GOOGLE_API_KEY)
    rag_chain_pro_short_step =  google_model_pro | StrOutputParser()

    google_model_flash = ChatGoogleGenerativeAI(model='gemini-2.0-flash', api_key=GOOGLE_API_KEY)
    rag_chain_flash_short_step =  google_model_flash | StrOutputParser()

    prompt1, prompt2 = self.__return_prompt__()
    similarity = self.chroma.similarity_search(query, k=4)

    prompt1 = prompt1.format(documents=similarity, query=query)
    prompt2 = prompt2.format(documents2=similarity, query=query)

    answer1 = rag_chain_pro_short_step.invoke(prompt1)
    answer2 = rag_chain_flash_short_step.invoke(prompt2)
    return answer1, answer2, [query, similarity]



  def __Parse_video_or_audio__(self, path):
    aai.settings.api_key = self.assembly_api
    transcriber = aai.Transcriber()

    transcript = transcriber.transcribe(path)

    split_text = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    video_chunks = split_text.create_documents([transcript.text])

    video_ids = [str(uuid4) for _ in video_chunks]
    video_docs = [
        Document(page_content=chunk.page_content, metadata={"video_id": video_ids[i]}) for i, chunk in enumerate(video_chunks)
    ]

    self.chroma.add_documents(documents=video_docs, ids=video_ids)









  def __Parse_pdf__(self):
#-------------------------------------------------Summaries Part-----------------------------------------------------

    chunks = self.__Extract_PDF__()
    Tables, Texts, Images = self.__extract_from_chunk__(chunks)



    texts_summarize = self.__summaries_text__(Texts)
  # tables_html = [table.metadata.text_as_html for table in Tables]
  # table_summarize = __summaries_text__(tables_html)
    image_summaries = self.__summaries_images_fn__(Images)

#-------------------------------------------------Storing Part-----------------------------------------------------


    id_key = "paper_id"
    doc_ids = [str(uuid4()) for _ in Texts]
    summary_texts = [
        Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(texts_summarize)
    ]
    self.chroma.add_documents(documents=summary_texts, ids=doc_ids)


    # table_ids = [str(uuid4()) for _ in Tables]
    # summary_tables = [
    #   Document(page_content=summary, metadata={id_key: table_ids[i]}) for i, summary in enumerate(table_summarize)
    # ]
    # self.chroma.add_documents(documents=summary_tables, ids=table_ids)

    img_ids = [str(uuid4()) for _ in Images]
    summary_img = [
        Document(page_content=summary, metadata={id_key: img_ids[i]}) for i, summary in enumerate(image_summaries)
    ]
    self.chroma.add_documents(documents=summary_img, ids=img_ids)








  def __Extract_PDF__(self):
    st.write("Extracting")
    extracted_data = partition_pdf(
    filename=self.file_path,       # format : .../x.pdf

    strategy="hi_res",# recommnd for images

    infer_table_structure=True,
    extract_images_in_pdf=True,#Extract tables too
    langueges=['eng'], #Only on English Papers
    extract_image_block_types=["Image"],
    extract_image_block_to_payload=True,    # [optional] Store images in base64 format
    chunking_strategy="by_title",
    max_characters=10000,                  # defaults to 500
    combine_text_under_n_chars=2000,       # defaults to 0
    new_after_n_chars=6000,# optional - only works when ``extract_image_block_to_payload=False``
    )

    return extracted_data




  def __extract_from_chunk__(self, pdf_chunks):
    tables = []
    texts = []
    images = []
    for chunk in pdf_chunks:
        if "Table" in str(type(chunk)):
            tables.append(chunk)

        if "CompositeElement" in str(type((chunk))):
            texts.append(chunk)
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                  if "Image" in str(type(el)):
                      images.append(el.metadata.image_base64)
    return tables, texts, images






  @retry.Retry(predicate=is_retriable)
  def __summaries_text__(self, chunks):

    template = ChatPromptTemplate.from_template(prompt_text)
    summarize_chain = {"element": lambda x: x} | template | self.google  | StrOutputParser()
    chunks_summarize = summarize_chain.batch(chunks)
    return chunks_summarize


  @retry.Retry(predicate=is_retriable)
  def __summaries_images_fn__(self, images_base64):
      summaries = []

      for image_b64 in images_base64:
          img_url = f"data:image/jpeg;base64,{image_b64}"

          msg = HumanMessage(
              content=[
                  {
                      "type": "text",
                      "text": (
                          "Describe the image in detail. For context, "
                          "the image is part of a research paper explaining the transformers architecture. "
                          "Be specific about graphs, such as bar plots."
                      ),
                  },
                  {
                      "type": "image_url",
                      "image_url": {"url": img_url}
                  }
              ]
          )

          response = self.llm.invoke([msg])
          parser = StrOutputParser()
          summaries.append(parser.invoke(response))

      return summaries

#C:\Users\Mohsen\Downloads\gulati20_interspeech.pdf
#https://www.poetryoutloud.org/wp-content/uploads/sites/2/2019/07/01-Track-01.mp3
# obj = LangChainLLMByGoogle(file_path="C:\\Users\\Mohsen\\Downloads\\gulati20_interspeech.pdf", kind="pdf", api_key=GOOGLE_API_KEY, assembly_api=ASSEMBLY_AI_API)
# answer = obj.invoke("What is architecture of model")
# print(answer)
