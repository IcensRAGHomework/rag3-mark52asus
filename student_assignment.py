import datetime
import chromadb
import traceback
import pandas as pd
import time

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"

def generate_hw01():
    # 讀取 CSV 檔案
    file_path = "./COA_OpenData.csv"
    df = pd.read_csv(file_path)
    
    # 轉換日期為時間戳格式（秒）
    df['date'] = pd.to_datetime(df['CreateDate']).astype('int64') // 10**9
    
    # 初始化 ChromaDB 客戶端（使用 SQLite 作為儲存）
    client = chromadb.PersistentClient(path=dbpath)
    
    # 刪除舊 Collection，確保使用 OpenAI Embedding (1536 維度)
    try:
        client.delete_collection("TRAVEL")
    except Exception as e:
        print("Collection does not exist or could not be deleted.")
    
    # 設定 OpenAI Embedding Function
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=gpt_emb_config['api_key'],
        api_base=gpt_emb_config['api_base'],
        api_type=gpt_emb_config['openai_type'],
        api_version=gpt_emb_config['api_version'],
        deployment_id=gpt_emb_config['deployment_name']
    )
    
    # 建立新的 Collection，確保使用 1536 維度
    collection = client.create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    
    # 準備要插入的數據
    for index, row in df.iterrows():
        metadata = {
            "file_name": "COA_OpenData.csv",
            "name": row['Name'],
            "type": row['Type'],
            "address": row['Address'],
            "tel": row['Tel'],
            "city": row['City'],
            "town": row['Town'],
            "date": row['date']
        }
        
        document_text = row['HostWords'] if pd.notna(row['HostWords']) else ""
        
        collection.add(
            ids=[str(index)],
            documents=[document_text],
            metadatas=[metadata]
        )
    return collection

def generate_hw02(question, city=[], store_type=[], start_date=None, end_date=None):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=gpt_emb_config['api_key'],
        api_base=gpt_emb_config['api_base'],
        api_type=gpt_emb_config['openai_type'],
        api_version=gpt_emb_config['api_version'],
        deployment_id=gpt_emb_config['deployment_name']
    )
    
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    
    # 查詢資料（使用 query() 方法）
    results = collection.query(
        query_texts=[question],
        n_results=40 #40, 才會有8筆
    )
    
    # 確保結果中有距離資訊
    if not results["distances"] or len(results["distances"]) == 0:
        return []
    
    # **解開巢狀列表**
    ids_list = results["ids"][0] if results["ids"] else []
    distances_list = results["distances"][0] if results["distances"] else []
    metadatas_list = results["metadatas"][0] if results["metadatas"] else []
    for i in range(len(ids_list)):  # 確保正確迭代
        metadata = metadatas_list[i]
        score = distances_list[i]
        similarity = 1 - score  # 轉換為相似度
        name = metadata.get("name", "Unknown")
        store_city = metadata.get("city", "Unknown")
        store_type_value = metadata.get("type", "Unknown")        
        print(f"Before Matching Store: {name}, City: {store_city}, Type: {store_type_value}, Start Date: {start_date}, End Date: {end_date}, Similarity: {similarity}")  # Debug: 印出名稱、城市、類型、起始/結束日期和相似度

    # 篩選條件
    filtered_results = []
    for i in range(len(ids_list)):  # 確保正確迭代
        metadata = metadatas_list[i]
        score = distances_list[i]

        similarity = 1 - score  # 轉換為相似度

        if similarity >= 0.70:  # 修正為正確相似度比較
            if city and metadata.get("city", "") not in city:
                continue
            if store_type and metadata.get("type", "") not in store_type:
                continue
            if start_date and end_date:
                if not (start_date.timestamp() <= metadata.get("date", 0) <= end_date.timestamp()):
                    continue
            name = metadata.get("name", "Unknown")
            store_city = metadata.get("city", "Unknown")
            store_type_value = metadata.get("type", "Unknown")            
            print(f"Matching Store: {name}, City: {store_city}, Type: {store_type_value}, Start Date: {start_date}, End Date: {end_date}, Similarity: {similarity}")  # Debug: 印出名稱、城市、類型、起始/結束日期和相似度
            filtered_results.append((metadata.get("name", "Unknown"), similarity))
    
    # 依據相似度排序（由高到低）
    filtered_results.sort(key=lambda x: x[1], reverse=True)
    
    # 回傳店家名稱列表
    return [name for name, _ in filtered_results]


def generate_hw03(question, store_name, new_store_name, city, store_type):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=gpt_emb_config['api_key'],
        api_base=gpt_emb_config['api_base'],
        api_type=gpt_emb_config['openai_type'],
        api_version=gpt_emb_config['api_version'],
        deployment_id=gpt_emb_config['deployment_name']
    )
    
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    
    # 更新店家名稱
    results = collection.get()
    for i, metadata in enumerate(results["metadatas"]):
        if metadata["name"] == store_name:
            metadata["new_store_name"] = new_store_name
            collection.update(
                ids=[results["ids"][i]],
                metadatas=[metadata]
            )
    
    # 查詢資料
    results = collection.query(
        query_texts=[question],
        n_results=60
    )
    
    # 確保結果中有距離資訊
    if not results["distances"] or len(results["distances"]) == 0:
        return []
    
    # **解開巢狀列表**
    ids_list = results["ids"][0] if results["ids"] else []
    distances_list = results["distances"][0] if results["distances"] else []
    metadatas_list = results["metadatas"][0] if results["metadatas"] else []

    for i in range(len(ids_list)):  # 確保正確迭代
        metadata = metadatas_list[i]
        score = distances_list[i]
        similarity = 1 - score  # 轉換為相似度
        name = metadata.get("new_store_name", metadata.get("name", "Unknown"))
        print(f"Before Matching Store: {name}, City: {metadata.get('city', 'Unknown')}, Type: {metadata.get('type', 'Unknown')}, Similarity: {similarity}")

    # 篩選條件
    filtered_results = []
    for i in range(len(ids_list)):  # 確保正確迭代
        metadata = metadatas_list[i]
        score = distances_list[i]

        similarity = 1 - score  # 轉換為相似度

        if similarity >= 0.80:  # 修正為正確相似度比較
            if city and metadata.get("city", "") not in city:
                continue
            if store_type and metadata.get("type", "") not in store_type:
                continue
            name = metadata.get("new_store_name", metadata.get("name", "Unknown"))
            print(f"Matching Store: {name}, City: {metadata.get('city', 'Unknown')}, Type: {metadata.get('type', 'Unknown')}, Similarity: {similarity}")
            filtered_results.append((name, similarity))
    
    # 依據相似度排序（由高到低）
    filtered_results.sort(key=lambda x: x[1], reverse=True)
    
    # 回傳店家名稱列表
    return [name for name, _ in filtered_results]   

def demo(question):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    
    return collection

if __name__ == '__main__':
    #可以在程式後，印出collection內容
    collection = generate_hw01()
    # 取得所有記錄（限制前 5 筆以便查看）
    results = collection.get()
    # 印出資料
    for i in range(min(5, len(results["ids"]))):
        print(f"ID: {results['ids'][i]}")
        print(f"Document: {results['documents'][i]}")
        print(f"Metadata: {results['metadatas'][i]}")
        print("-" * 40)
    #question = '我想要找有關茶餐點的店家'
    #city = ["宜蘭縣", "新北市"]
    #store_type = ["美食"]
    #start_date = datetime.datetime(2024, 4, 1)
    #end_date  = datetime.datetime(2024, 5, 1)
    #result = generate_hw02(question, city, store_type, start_date, end_date)
    #print(result)
    #question = '我想要找南投縣的田媽媽餐廳，招牌是蕎麥麵'
    #store_name = "耄饕客棧"
    #new_store_name = "田媽媽（耄饕客棧）"
    #city = ["南投縣"]
    #store_type = ["美食"]
    #result = generate_hw03(question, store_name, new_store_name, city, store_type)
    #print(result)

    
 
   
