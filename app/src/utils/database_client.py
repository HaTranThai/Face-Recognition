"""Database client utilities for calling qdrant_database_FE API."""
import logging
from typing import List, Optional, Dict, Any
import httpx
import asyncio
from config.logging import get_database_logger

logger = get_database_logger()


class DatabaseClient:
    """HTTP client for qdrant_database_FE API operations."""
    
    def __init__(self, api_host: str, api_port: int):
        """Initialize the database client."""
        self.api_host = api_host
        self.api_port = api_port
        self.base_url = f"http://{api_host}:{api_port}"
        self.timeout = httpx.Timeout(30.0)
    
    async def get_collections(self) -> List[str]:
        """
        Get list of all collections.
        
        Returns:
            List of collection names
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/get_collections")
                response.raise_for_status()
                
                result = response.json()
                if result.get("status") == 200:
                    return result.get("collections", [])
                else:
                    logger.error(f"Failed to get collections: {result}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting collections: {str(e)}")
            return []
    
    async def create_collection(self, collection_name: str) -> bool:
        """
        Create a new collection.
        
        Args:
            collection_name: Name of the collection to create
            
        Returns:
            bool: True if created successfully or already exists
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                payload = {"collection_name": collection_name}
                response = await client.post(
                    f"{self.base_url}/create_collection", 
                    json=payload
                )
                response.raise_for_status()
                
                # Both 200 (existed) and 201 (created) are success cases
                return response.status_code in [200, 201]
                
        except Exception as e:
            logger.error(f"Error creating collection '{collection_name}': {str(e)}")
            return False
    
    async def ensure_collection_exists(self, collection_name: str, vector_size: int = 4096) -> bool:
        """
        Ensure that a collection exists, create if it doesn't.
        
        Args:
            collection_name: Name of the collection
            vector_size: Size of the embedding vectors (ignored, using API default of 4096)
            
        Returns:
            bool: True if collection exists or was created successfully
        """
        try:
            collections = await self.get_collections()
            if collection_name in collections:
                logger.info(f"Collection '{collection_name}' already exists")
                return True
            
            # Create collection if it doesn't exist
            logger.info(f"Creating collection '{collection_name}'")
            success = await self.create_collection(collection_name)
            if success:
                logger.info(f"Collection '{collection_name}' created successfully")
            return success
            
        except Exception as e:
            logger.error(f"Error ensuring collection '{collection_name}' exists: {str(e)}")
            return False
    
    async def ensure_collections_exist(self, store_id: str) -> bool:
        """
        Ensure both Customer and Employee collections exist for a store.
        
        Args:
            store_id: Store identifier
            
        Returns:
            bool: True if both collections exist or were created successfully
        """
        try:
            customer_collection = f"{store_id}_Customers"
            employee_collection = f"{store_id}_Employees"
            
            # Create both collections
            customer_success = await self.ensure_collection_exists(customer_collection)
            employee_success = await self.ensure_collection_exists(employee_collection)
            
            return customer_success and employee_success
            
        except Exception as e:
            logger.error(f"Error ensuring collections exist for store {store_id}: {str(e)}")
            return False
    
    async def insert_point(
        self, 
        collection_name: str, 
        vector_embedding: List[float], 
        id: str, 
        name: str, 
        store_id: str, 
        is_update_id: bool = False
    ) -> bool:
        """
        Insert a point into the collection.
        
        Args:
            collection_name: Name of the collection
            vector_embedding: Face embedding vector (must be 4096 dimensions)
            id: Unique identifier for the point
            name: Name associated with the point
            store_id: Store ID for filtering
            is_update_id: Whether to update existing point with same ID
            
        Returns:
            bool: True if inserted successfully
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                payload = {
                    "collection_name": collection_name,
                    "vector_embedding": vector_embedding,
                    "id": id,
                    "name": name,
                    "store_id": store_id,
                    "is_update_id": is_update_id
                }
                
                response = await client.post(
                    f"{self.base_url}/insert_point", 
                    json=payload
                )
                response.raise_for_status()
                
                return response.status_code == 201
                
        except Exception as e:
            logger.error(f"Error inserting point {id}: {str(e)}")
            return False
    
    async def store_embedding(
        self, 
        collection_name: str, 
        point_id: str, 
        embedding: List[float], 
        payload: dict
    ) -> bool:
        """
        Store an embedding vector in the specified collection.
        
        Args:
            collection_name: Name of the collection
            point_id: Unique identifier for the point
            embedding: Face embedding vector
            payload: Metadata to store with the embedding (should contain id, name, store_id)
            
        Returns:
            bool: True if stored successfully
        """
        try:
            # Extract required fields from payload
            id_value = payload.get("id", point_id)
            name = payload.get("name", "")
            store_id = payload.get("store_id", "")
            
            return await self.insert_point(
                collection_name=collection_name,
                vector_embedding=embedding,
                id=id_value,
                name=name,
                store_id=store_id,
                is_update_id=False
            )
            
        except Exception as e:
            logger.error(f"Error storing embedding {point_id}: {str(e)}")
            return False
    
    async def search_point(
        self, 
        collection_name: str, 
        vector_embedding: List[float], 
        store_id: str
    ) -> List[Dict[str, Any]]:
        """
        Search for similar points in the collection.
        
        Args:
            collection_name: Name of the collection to search
            vector_embedding: Query face embedding vector
            store_id: Store ID for filtering
            
        Returns:
            List of matching results with metadata
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                payload = {
                    "collection_name": collection_name,
                    "vector_embedding": vector_embedding,
                    "store_id": store_id
                }
                
                response = await client.post(
                    f"{self.base_url}/search_point", 
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                if result.get("message") == "Point found":
                    # Convert API response format to expected format
                    search_results = []
                    for score, payload_data in result.get("data", []):
                        search_results.append({
                            'id': payload_data.get('id'),
                            'score': score,
                            'payload': payload_data
                        })
                    return search_results
                else:
                    logger.info(f"No points found in {collection_name}")
                    return []
                
        except Exception as e:
            logger.error(f"Error searching points: {str(e)}")
            return []
    
    async def search_similar_faces(
        self, 
        collection_name: str, 
        query_embedding: List[float], 
        limit: int = 5,
        score_threshold: float = 0.5,
        store_id: str = ""
    ) -> List[dict]:
        """
        Search for similar faces in the collection.
        
        Args:
            collection_name: Name of the collection to search
            query_embedding: Query face embedding vector
            limit: Maximum number of results to return (ignored by API)
            score_threshold: Minimum similarity score threshold (handled by API)
            store_id: Store ID for filtering
            
        Returns:
            List of matching results with metadata
        """
        try:
            results = await self.search_point(
                collection_name=collection_name,
                vector_embedding=query_embedding,
                store_id=store_id
            )
            
            logger.info(f"Found {len(results)} similar faces in {collection_name}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar faces: {str(e)}")
            return []
    
    async def delete_point(self, collection_name: str, id: str) -> bool:
        """
        Delete a point by ID from the collection.
        
        Args:
            collection_name: Name of the collection
            id: ID of the point to delete (from payload.id, not point UUID)
            
        Returns:
            bool: True if deleted successfully
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                payload = {
                    "collection_name": f"{collection_name}",
                    "id": id
                }
                logger.info(f"Preparing to delete point {id} from {collection_name}")
                
                
                response = await client.request(
                    method="DELETE",
                    url=f"{self.base_url}/delete_point", 
                    json=payload
                )
                
                # response = await client.post(
                #     f"{self.base_url}/delete_point", 
                #     json=payload
                # )
                
                logger.info(f"Deleting point {id} from {collection_name}")
                logger.debug(f"Delete response: {response.text}")
                
                response.raise_for_status()
                
                return response.status_code == 200
                
        except Exception as e:
            logger.error(f"Error deleting point {id}: {str(e)}")
            return False
    
    async def delete_by_id(self, collection_name: str, point_id: str) -> bool:
        """
        Delete a point by ID from the collection.
        
        Args:
            collection_name: Name of the collection
            point_id: ID of the point to delete
            
        Returns:
            bool: True if deleted successfully
        """
        return await self.delete_point(collection_name, point_id)
    
    async def create_snapshot(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Create a snapshot of the collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Snapshot information or None if failed
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/create_snapshot/{collection_name}"
                )
                response.raise_for_status()
                
                return response.json()
                
        except Exception as e:
            logger.error(f"Error creating snapshot for {collection_name}: {str(e)}")
            return None
    
    async def recover_snapshot(self, collection_name: str, snapshot_name: str) -> bool:
        """
        Recover a collection from a snapshot.
        
        Args:
            collection_name: Name of the collection
            snapshot_name: Name of the snapshot file
            
        Returns:
            bool: True if recovered successfully
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                payload = {
                    "collection_name": collection_name,
                    "snapshot_name": snapshot_name
                }
                
                response = await client.post(
                    f"{self.base_url}/recover_snapshot", 
                    json=payload
                )
                response.raise_for_status()
                
                return response.status_code == 200
                
        except Exception as e:
            logger.error(f"Error recovering snapshot {snapshot_name}: {str(e)}")
            return False, str(e)
    
    async def recover_snapshot_local(self, collection_name: str) -> bool:
        """
        Recover a collection from the latest local snapshot.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            bool: True if recovered successfully
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/recover_snapshot_local/{collection_name}"
                )
                response.raise_for_status()
                
                return response.status_code == 200
                
        except Exception as e:
            logger.error(f"Error recovering local snapshot for {collection_name}: {str(e)}")
            return False
    
    async def get_collection_info(self, collection_name: str) -> Optional[dict]:
        """
        Get information about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Collection information or None if not found
        """
        try:
            collections = await self.get_collections()
            if collection_name in collections:
                return {
                    'name': collection_name,
                    'exists': True
                }
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting collection info for {collection_name}: {str(e)}")
            return None
