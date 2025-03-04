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
    
    # 建立或獲取 Collection
    collection = client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"}
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
def generate_hw02(question, city, store_type, start_date, end_date):
    pass
    
def generate_hw03(question, store_name, new_store_name, city, store_type):
    pass
    
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
	collection = generate_hw01()
 
   
