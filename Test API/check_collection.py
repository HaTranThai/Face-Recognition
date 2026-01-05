"""
Script để kiểm tra số lượng points trong collection
"""
import requests

QDRANT_DB_URL = "http://localhost:7005"

def check_collection_points(collection_name: str):
    """Kiểm tra collection có bao nhiêu points"""
    print(chillf"\n{'='*60}")
    print(f"CHECKING COLLECTION: {collection_name}")
    print(f"{'='*60}")
    
    try:
        # Gọi trực tiếp Qdrant API để count points
        response = requests.get(
            f"http://localhost:6333/collections/{collection_name}",
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            result = data.get('result', {})
            points_count = result.get('points_count', 0)
            vectors_count = result.get('vectors_count', 0)
            status = result.get('status', 'unknown')
            
            print(f"\n✅ Collection Status: {status}")
            print(f"📊 Points Count: {points_count:,}")
            print(f"🔢 Vectors Count: {vectors_count:,}")
            
            if points_count == 0:
                print(f"\n⚠️  WARNING: Collection is EMPTY!")
                print(f"   This is why face recognition is failing.")
                print(f"   You need to recover from a valid snapshot.")
            else:
                print(f"\n✅ Collection has data!")
                
        elif response.status_code == 404:
            print(f"\n❌ Collection NOT FOUND!")
            print(f"   Collection '{collection_name}' does not exist.")
        else:
            print(f"\n❌ Error: Status code {response.status_code}")
            print(f"   Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Cannot connect to Qdrant at localhost:6333")
        print(f"   Is Qdrant running?")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    # Kiểm tra cả 2 collections
    check_collection_points("TEST_Employees")
    # check_collection_points("TMP_Customers")
