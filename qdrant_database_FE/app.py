from fastapi import FastAPI, status 
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from typing import Union, List
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from uuid import uuid4
# import torch
# from fastapi.middleware.cors import CORSMiddleware

import os
# import random
import datetime
from logging_config import setup_database_logging, get_database_logger

# Setup logging
logger = setup_database_logging()

tags_metadata = [
    {
        "name": "Colection",
        "description": "APIs for collection"
    },
    {
        "name":"Snapshot",
        "description": "APIs for snapshot"
    },
    {
        "name":"Point",
        "description": "APIs for point"
    }
]
# from qdrant_db import *
# docs_url=None, redoc_url=None
app = FastAPI(openapi_tags=tags_metadata)

logger.info("Database API service starting...")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int( os.getenv("QDRANT_PORT", "6333"))
THRESHOLD_PASS = 0.54
THRESHOLD_SEARCH = 0.54

logger.info(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
client = AsyncQdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=300)

class CreateCollection(BaseModel):
    collection_name: str = None
    
class InsertPoint(BaseModel):
    collection_name: Union[str, None] = ""
    vector_embedding: Union[List[int], List[float]] = None
    id: Union[str, None] = ""
    name: Union[str, None] = ""
    store_id: Union[str, None] = ""
    is_update_id: Union[bool, None] = False

class SearchPoint(BaseModel):
    collection_name: Union[str, None] = ""
    vector_embedding: Union[List[int], List[float]] = None
    store_id: Union[str, None] = ""

class DeletePoint(BaseModel):
    collection_name: Union[str, None] = ""
    id: Union[str, None] = ""
    
class RecoverSnapshot(BaseModel):
    collection_name: Union[str, None] = ""
    snapshot_name: Union[str, None] = ""

async def _check_exist(collection_name):
    collections = await client.get_collections()
    return collection_name in [c.name for c in collections.collections]

async def _get_points(collection_name, id):
    try:
        res = await client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="id",
                        match=models.MatchValue(value=id),
                    )
                ]
            )
        )
        if len(res[0])==0:
            return None
        return res[0]
    except Exception as e:
        return None


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/get_collections", tags=["Colection"])
async def get_collections():
    collections = await client.get_collections()
    return {
        'status': status.HTTP_200_OK,
        'collections': [c.name for c in collections.collections]
    }
    
@app.post("/create_collection", tags=["Colection"])
async def create_collection(data:CreateCollection):
    collection_name= data.collection_name
    if collection_name is None:
        return JSONResponse(status_code=404, content={"message": "Collection name not found!"})
    if not await _check_exist(collection_name):
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=4096, distance=Distance.COSINE, on_disk=True),
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=200,
            ),
            quantization_config=models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(always_ram=True)
            )
        )
        await client.create_snapshot(collection_name=collection_name)
        return JSONResponse(status_code=201, content={"message": "Collection created"})
    else:
        return JSONResponse(status_code=200, content={"message": "Collection existed"})
    
@app.get('/create_snapshot/{collection_name}', tags=["Snapshot"])
async def create_snapshot(collection_name):
    try:
        snapshots = await client.list_snapshots(collection_name=f"{collection_name}")
        for snapshot in snapshots:
            await client.delete_snapshot(
                collection_name=collection_name, snapshot_name=snapshot.name
            )
        snapshot_info = await client.create_snapshot(collection_name=collection_name)
        return snapshot_info
    except Exception as e:
        return str(e)

@app.get('/all_snapshots/{collection_name}', tags=["Snapshot"])
async def all_snapshots(collection_name):
    snapshots = await client.list_snapshots(collection_name=collection_name)
    return snapshots

@app.get('/recover_snapshot_local/{collection_name}', tags=["Snapshot"])
async def recover_snapshot_local(collection_name):
    try:
        if collection_name == "Employees" or collection_name == "Customers":
            snapshots = await client.list_snapshots(collection_name=collection_name)
            if len(snapshots) == 0:
                snapshot = await client.create_snapshot(collection_name=collection_name)
                snapshots = await client.list_snapshots(collection_name=collection_name)
            snapshot = snapshots[0]
            await client.recover_snapshot(
                collection_name=collection_name, location=f"file:///qdrant/snapshots/{collection_name}/{snapshot.name}"
            )
            return JSONResponse(status_code=200, content={"message": "Recover snapshot successfully", 
                    "Path": f"file:///qdrant/snapshots/{collection_name}/{snapshot.name}"})
        else:
            return JSONResponse(
                status_code=404, 
                content={
                    "message": "Collection name not found or invalid!"}
                )
    except Exception as e:
        return JSONResponse(status_code=404, content={"message": str(e)})
    
@app.post('/recover_snapshot', tags=["Snapshot"])
async def recover_snapshot(data: RecoverSnapshot):
    try:
        collection_name, path_snapshot = data.collection_name, data.snapshot_name
        print(data)
        if collection_name.split("_")[1] not in ["Employees", "Customers"]:
            return JSONResponse(status_code=404, content={"message": "Collection name not found or invalid!"})

        if not await _check_exist(collection_name):
            await client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=4096, distance=Distance.COSINE, on_disk=True),
                hnsw_config=models.HnswConfigDiff(
                    m=16,
                    ef_construct=200, 
                ),
                quantization_config=models.BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(always_ram=True)
                )
            )
        await client.recover_snapshot(
            collection_name=collection_name, location=f"file:///qdrant/snapshots/{path_snapshot}"
        )
        return JSONResponse(status_code=200, content={"message": "Recover snapshot successfully"})
    except Exception as e:
        return JSONResponse(status_code=404, content={"message": str(e)})

# @app.get('/delete_snapshot')
# async def delete_snapshot():
#     snapshots = client.list_snapshots(collection_name="Employees")
#     for snapshot in snapshots:
#         client.delete_snapshot(
#             collection_name="Employees", snapshot_name=f"{snapshot.name}"
#         )
#     return JSONResponse(status_code=200, content={"message": "Delete snapshot successfully"})

@app.post("/insert_point", tags=["Point"])
async def insert_point(data:InsertPoint):
    collection_name = data.collection_name
    embedding = data.vector_embedding
    id = data.id
    name = data.name
    store_id = data.store_id
    is_update_id = data.is_update_id
    time_created = datetime.datetime.now().strftime("%Y/%m/%d")
    
    if collection_name == "" or collection_name is None:
        return JSONResponse(status_code=404, content={"message": "Collection name not found or invalid!"})
    
    if not await _check_exist(collection_name):
        return JSONResponse(status_code=404, content={"message": "Collection name not found or invalid!"})
    
    if embedding is None or len(embedding) != 4096:
        return JSONResponse(status_code=404, content={"message": "Embedding not found or invalid!"})
    
    if id is None or id == "":
        return JSONResponse(status_code=404, content={"message": "ID not found or invalid!"})
    if name is None or id == "":
        return JSONResponse(status_code=404, content={"message": "Name not found or invalid!"})
    
    _id = str(uuid4())
    payload = {
        'id': id,
        'name': name,
        "store_id": store_id,
        'time_created': time_created
    }
    
    if is_update_id:
        p = await _get_points(collection_name, id)
        if p is not None:
            _id = p[0].id
            payload = p[0].payload
        else:
            return JSONResponse(status_code=404, content={"message": "ID not found!"})
    try:
        point = PointStruct(id=_id,
                            vector=embedding,
                            payload=payload
                )
        await client.upsert(collection_name=collection_name, points=[point])
        
        return JSONResponse(status_code=201, content={"message": "Point inserted"})
    except Exception as e:
        return JSONResponse(status_code=404, content={"message": str(e)})
    

@app.post("/search_point", tags=["Point"])
async def search_point(data: SearchPoint):
    collection_name = data.collection_name
    vector_embedding = data.vector_embedding
    store_id = data.store_id
    if collection_name is None or collection_name == "":
        return JSONResponse(status_code=404, content={"message": "Collection name not found or invalid!"})
    
    if not await _check_exist(collection_name):
        return JSONResponse(status_code=404, content={"message": "Collection name not found or invalid!"})
    
    # if vector_embedding is None or len(vector_embedding) != 128 or not all(isinstance(i, (int, float)) for i in vector_embedding) or not all(-1 <= i <= 1 for i in vector_embedding):
    if vector_embedding is None or len(vector_embedding) != 4096:
        return JSONResponse(status_code=404, content={"message": "Embedding not found or invalid!"})
    
    try:
        result = await client.search(
            collection_name=collection_name, 
            query_vector=vector_embedding, 
            limit=5, 
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="store_id",
                        match=models.MatchValue(value=store_id)
                    )
                ]
            ),
            search_params=models.SearchParams(
                hnsw_ef=128,
                exact=True,
                quantization=models.QuantizationSearchParams(
                    rescore=True,       
                    oversampling=2.0  
                )
            ),
            timeout=30
        )
        print([(i.score, i.payload) for i in result])

        data = [(i.score, i.payload) for i in result if i.score >= THRESHOLD_PASS]
        
        print("Data search: ", data)
        
        if len(data) == 0:
            return JSONResponse(status_code=200, content={"message": "Point not found", "data": []})
        
        data_dict = {}
        for i in result:
            if i.score >= THRESHOLD_PASS:
                if i.payload['id'] in data_dict:
                    data_dict[i.payload['id']] += 1
                else:
                    data_dict[i.payload['id']] = 1
        
        print(data_dict)
        
        if len(set(data_dict.values())) == 1 and len(data_dict) > 1:
            print("All values are the same")
            data_result = [(i.score, i.payload) for i in result if i.score >= THRESHOLD_SEARCH]
            
            if len(data_result) == 0:
                return JSONResponse(status_code=200, content={"message": "Point not found", "data": []})
                
            i_max = max(data_result, key=lambda x: x[0])
            score = i_max[0]
            payload = i_max[1]
            return JSONResponse(status_code=200, content={"message": "Point found", "data": [(score,payload)]})
        
        id_max = max(data_dict, key=data_dict.get)
        
        scores = []
        for i in result:
            if i.payload['id'] == id_max:
                scores.append(i.score)
        
        score = max(scores)
        
        payload = None
        for i in result:
            if i.payload['id'] == id_max:
                payload = i.payload
                break
        
        return JSONResponse(status_code=200, content={"message": "Point found", "data": [(score, payload)]})
    except Exception as e:
        return JSONResponse(status_code=404, content={"message": str(e)})

@app.delete("/delete_collection", tags=["Colection"])
async def delete_collection(data: CreateCollection):
    collection_name = data.collection_name
    if collection_name is None:
        return JSONResponse(status_code=404, content={"message": "Collection name not found!"})
    if not await _check_exist(collection_name):
        return JSONResponse(status_code=404, content={"message": "Collection name not found!"})
    try:
        await client.delete_collection(collection_name)
        try:
            snapshots = await client.list_snapshots(collection_name=f"{collection_name}")
            for snapshot in snapshots:
                await client.delete_snapshot(
                    collection_name=collection_name, snapshot_name=snapshot.name
                )
        except Exception as e:
            print(f"Error deleting snapshot: {str(e)}")
        return JSONResponse(status_code=200, content={"message": "Collection deleted"})
    except Exception as e:
        return JSONResponse(status_code=404, content={"message": str(e)})

@app.delete("/delete_point", tags=["Point"])
async def delete_point(data: DeletePoint):
    collection_name = data.collection_name
    id = data.id
    if collection_name is None or collection_name == "":
        return JSONResponse(status_code=404, content={"message": "Collection name not found or invalid!"})
    
    if not await _check_exist(collection_name):
        return JSONResponse(status_code=404, content={"message": "Collection name not found or invalid!"})
    
    if id is None or id == "":
        return JSONResponse(status_code=404, content={"message": "ID not found or invalid!"})
    
    try:
        await client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="id",
                            match=models.MatchValue(value=id),
                        ),
                    ],
                )
            )
        )
        return JSONResponse(status_code=200, content={"message": "Point deleted"})
    except Exception as e:
        return JSONResponse(status_code=404, content={"message": str(e)})

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for database API."""
    try:
        # Test connection to Qdrant
        collections = await client.get_collections()
        
        return {
            "status": "healthy",
            "message": "Database API is running",
            "qdrant_connection": "connected",
            "collections_count": len(collections.collections),
            "qdrant_host": QDRANT_HOST,
            "qdrant_port": QDRANT_PORT
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Database connection failed: {str(e)}",
                "qdrant_connection": "disconnected",
                "qdrant_host": QDRANT_HOST,
                "qdrant_port": QDRANT_PORT
            }
        )

# if __name__ == "__main__":
#    import uvicorn
#    uvicorn.run(app, host="0.0.0.0", port=7005) # chạy ứng dụng với uvicorn, host là ip và port là cổng
