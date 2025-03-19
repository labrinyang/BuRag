# 1.从requirements.txt安装依赖

```
pip install --upgrade pip
pip install -r requirements.txt
```

# 2.运行

```
uvicorn main:app --reload
```

在`http://127.0.0.1:8000`可访问api，在`http://127.0.0.1:8000/docs`可查看文档


# 3.api

第一个api：/crypto /query_rootdata_data
参数示例：
```
{
    "user_query": "what is PNP",
    "top_n": 3,
    "collection_name": "rootdata_crypto_collection",
    "temperature": 0.1
}```

这三个参数保持默认设置："top_n": 3,
    "collection_name": "rootdata_crypto_collection",
    "temperature": 0.1

第二个api：/cryoto/query_news_date
参数示例：
```
{
    "user_query": "特朗普发行NFT事件始末？",
    "top_n": 6,
    "collection_name": "news_crypto_collection",
    "temperature": 0.1,
    "start_date": "2022-01-01 00:00:00",
    "end_date": "2025-03-01 23:59:59"
}
```

这三个参数保持默认设置：    "top_n": 6,
    "collection_name": "news_crypto_collection",
    "temperature": 0.1,

时间参数格式为"YYYY-MM-DD HH:MM:SS"
