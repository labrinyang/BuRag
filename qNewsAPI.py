from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
from datetime import datetime
from dotenv import load_dotenv
import chromadb
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.core.query_engine import RetrieverQueryEngine

# 加载环境变量
load_dotenv()

router = APIRouter()  # 创建 APIRouter 实例

# 定义请求体参数模型
class DateQueryRequest(BaseModel):
    user_query: str
    top_n: int
    collection_name: str
    temperature: float
    start_date: str  # 格式要求："YYYY-MM-DD HH:MM:SS"
    end_date: str    # 格式要求："YYYY-MM-DD HH:MM:SS"

# 全局初始化 Chroma 客户端
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def convert_date_to_timestamp(date_str: str) -> float:
    """
    解析 'YYYY-MM-DD HH:MM:SS' 格式的日期字符串，并转换为 Unix 时间戳
    """
    return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").timestamp()

def get_date_filtered_retriever(index, start_date=None, end_date=None, top_k=20):
    """
    根据日期范围构造过滤器，并返回支持时间过滤的检索器
    """
    filters = []
    if start_date:
        filters.append(MetadataFilter(
            key="date",
            value=convert_date_to_timestamp(start_date),
            operator=FilterOperator.GTE
        ))
    if end_date:
        filters.append(MetadataFilter(
            key="date",
            value=convert_date_to_timestamp(end_date),
            operator=FilterOperator.LTE
        ))
    metadata_filters = MetadataFilters(filters=filters)
    retriever = index.as_retriever(
        similarity_top_k=top_k,
        filters=metadata_filters
    )
    return retriever

def build_date_filtered_query_engine(collection_name: str, top_n: int, temperature: float, start_date: str, end_date: str):
    """
    构造带日期过滤的查询引擎
    """
    # 初始化 LLM 模型
    llm_instance = OpenAI(
        model="gpt-4o",
        temperature=temperature,
        max_tokens=4096,
        api_key=os.getenv('OPENAI_API_KEY'),
        api_base=os.getenv('OPENAI_API_BASE', "https://chatapi.nloli.xyz/v1")
    )
    Settings.llm = llm_instance

    # 初始化 embedding 模型
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-large",
        dimensions=1536,
        api_key=os.getenv('OPENAI_API_KEY'),
        api_base=os.getenv('OPENAI_API_BASE', "https://chatapi.nloli.xyz/v1")
    )
    Settings.embed_model = embed_model

    # 根据 collection_name 获取或创建 Chroma 集合
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    chroma_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # 从 Chroma 加载索引
    index = VectorStoreIndex.from_vector_store(vector_store=chroma_store)

    # 构造带日期过滤的检索器
    retriever = get_date_filtered_retriever(index, start_date=start_date, end_date=end_date, top_k=20)

    # 定义查询模板
    prompt_template = PromptTemplate("""
You are an experienced cryptocurrency industry analyst. Below is an article excerpt:

{context_str}

Based on the above information, please answer the user's question. Your response must strictly follow the JSON format:

User Question:
{query_str}

Important Reminders:

1. Strictly base your answer on the provided information, avoiding any unverified speculation or fabrication.
2. If the content involves time-sensitive information, be sure to specify the exact date to enhance accuracy and credibility.
3. Provide a comprehensive, detailed, and in-depth analysis to fully meet the needs of professional users.
4. Prefer new perspectives and thorough arguments, offering insights that the user may not have considered.
5. Strongly prefer clear reasoning and detailed data support.
6. If the provided information does not contain relevant details to answer the user’s question, return NaN in the response.
7. The response must be in English.
8. ONLY VALID JSON IS ALLOWED as an answer. No explanation or other text is allowed.

Your response format:
{"query_str": "User question", "response": "Detailed answer"}
Your detailed response:
""")
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[LLMRerank(top_n=top_n)],
        text_qa_template=prompt_template
    )
    return query_engine

@router.post("/query_news_date")
async def query_date_api(request: DateQueryRequest):
    """
    带日期过滤的查询接口
    """
    try:
        query_engine = build_date_filtered_query_engine(
            collection_name=request.collection_name,
            top_n=request.top_n,
            temperature=request.temperature,
            start_date=request.start_date,
            end_date=request.end_date
        )
        response = query_engine.query(request.user_query)
        return {"result": response.response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
