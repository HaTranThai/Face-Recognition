from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from uuid import uuid4
import pandas as pd

class QDrantDB:
    def __init__(self, path:str='qdrant_db', collection_name:str='face_128', embedding_size:int=4096, distance=Distance.COSINE):
        self.path = path
        self.client = QdrantClient(path=path)
        self.embedding_size = embedding_size
        self.distance = distance
        self.collection_name = collection_name
        self.all_collection_name = [c.name for c in self.client.get_collections().collections]
        if len(self.all_collection_name)==0:
            self.create_collection()

    def _get_point(self, customer_id=None):
        if customer_id is None:
            return None
        try:
            res = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="id",
                            match=models.MatchValue(value=customer_id),
                        )
                    ]
                )
            )
            if len(res[0])==0:
                return None
            return res[0]
        except Exception as e:
            return e
        
    def create_collection(self, collection_name:str=None):
        try:
            if collection_name:
                self.collection_name = collection_name
            if self.collection_name in self.all_collection_name:
                self.delete_collection()
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embedding_size, distance=self.distance),
            )
            self.all_collection_name = [c.name for c in self.client.get_collections().collections]
            return True
        except Exception as e:
            return e

    def delete_collection(self, collection_name:str=None):
        try:
            if collection_name:
                self.collection_name = collection_name
            self.client.delete_collection(self.collection_name)
            self.all_collection_name = [c.name for c in self.client.get_collections().collections]
            return True
        except Exception as e:
            return e
    
    def insert_face(self, vector:list=None, payload:dict={'name': 'test_name', 'id': 'test_id'}, id:str=None):
        if vector==None:
            return False
        _id = str(uuid4())
        if id is not None:
            p = self._get_point(id)
            if len(p)!=0:
                _id = p[0].id
                payload = p[0].payload if payload is None else payload
            else:
                return 'id not found!'
        try:
            point = PointStruct(id=_id,
                                vector=vector,
                                payload=payload
                    )
            self.client.upsert(collection_name=self.collection_name, points=[point])
            return True
        except Exception as e:
            return e

    def delete_point(self, id:str=None):
        if not id:
            return False
        try:
            self.client.delete(
                collection_name=self.collection_name,
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
            return True
        except Exception as e:
            return e

    def search(self, vector:list=None, limit:int=1, thresh:float=0.95):
        if vector==None:
            return []
        try:
            results = self.client.search(collection_name=self.collection_name, query_vector=vector, limit=limit)
            return [(i.score, i.payload) for i in results if i.score >= thresh]
        except:
            return []
        
    def save_data(self, path_csv='data.csv'):
        data = self.client.search(collection_name=self.collection_name, query_vector=list(map(float, range(self.embedding_size))), with_vectors=True)
        out = []
        for i in data:
            res = i.payload
            res['vector'] = i.vector
            out.append(res)
        df = pd.DataFrame(out)
        df.to_csv(path_csv, index=False)
        return True
        
    def close(self,):
        self.client.close()

if __name__ == "__main__":
    qd = QDrantDB()
    qd.close()