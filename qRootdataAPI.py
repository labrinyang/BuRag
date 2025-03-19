from fastapi import FastAPI, HTTPException
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import chromadb
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.query_engine import RetrieverQueryEngine

# 加载环境变量
load_dotenv()

router = APIRouter()
# 定义请求体参数模型
class QueryRequest(BaseModel):
    user_query: str
    top_n: int
    collection_name: str
    temperature: float

# 创建一个全局的 Chroma 客户端（建议保持长连接，避免重复初始化）
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def build_query_engine(collection_name: str, top_n: int, temperature: float):
    """
    根据传入的参数构建查询引擎：
    1. 根据 temperature 参数初始化 LLM 模型
    2. 根据 collection_name 参数获取或创建 Chroma 集合
    3. 构造索引、检索器以及查询引擎，并设置 LLMRerank 的 top_n 参数
    """
    # 1. 初始化 LLM 模型（temperature 参数动态设置）
    llm_instance = OpenAI(
        model="gpt-4o",
        temperature=temperature,  # 使用 API 传入的温度
        max_tokens=4096,
        api_key=os.getenv('OPENAI_API_KEY'),
        api_base=os.getenv('OPENAI_API_BASE', "https://chatapi.nloli.xyz/v1")
    )
    Settings.llm = llm_instance

    # 2. 初始化 embedding 模型
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-large",
        dimensions=1536,
        api_key=os.getenv('OPENAI_API_KEY'),
        api_base=os.getenv('OPENAI_API_BASE', "https://chatapi.nloli.xyz/v1")
    )
    Settings.embed_model = embed_model

    # 3. 根据传入的 collection_name 初始化 Chroma 集合
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    chroma_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # 4. 从 Chroma 加载索引
    index = VectorStoreIndex.from_vector_store(vector_store=chroma_store)

    # 5. 构造检索器（此处 similarity_top_k 固定为 20，可根据需要调整）
    retriever = index.as_retriever(similarity_top_k=20)

    # 6. 定义文本问答模板（注意其中的占位符 {context_str} 与 {query_str} 会在查询时被填充）
    template_str = """
You are a professional cryptocurrency industry analyst specializing in providing users with detailed information queries about specific projects. Below is an excerpt from the database as a reference:

{context_str}

Based on the above information, please answer the user's question. Your response must strictly follow the JSON format:

User Question:
{query_str}

Important Notes:

1. If the database contains detailed information relevant to the user’s question, provide all related content as comprehensively as possible without omissions.
2. If the database does not contain relevant information, return NaN in the response.
3. Answers must be strictly based on the provided data and must not include any unverified speculation or assumptions.
4. The response must be in English.
5. ONLY VALID JSON IS ALLOWED as an answer. No explanation or other text is allowed.

Your response format:
{"query_str": "User question", "response": "Detailed answer"}
Your detailed response:
    """
    prompt_template = PromptTemplate(template_str)

    # 7. 构造查询引擎，传入 LLMRerank 的 top_n 参数
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[LLMRerank(top_n=top_n)],
        text_qa_template=prompt_template
    )

    return query_engine

@router.post("/query_rootdata_data")
async def query_api(request: QueryRequest):
    """
    接收 POST 请求，参数见 QueryRequest 定义
    返回 JSON 格式结果：
    {"result": <llm 返回的结果>}
    """
    try:
        # 构建查询引擎（根据传入的 collection_name, top_n 和 temperature 参数）
        query_engine = build_query_engine(request.collection_name, request.top_n, request.temperature)
        # 执行查询
        response = query_engine.query(request.user_query)
        # 假定返回结果在 response.response 属性中
        return {"result": response.response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
