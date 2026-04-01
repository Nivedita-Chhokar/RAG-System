import chromadb
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

print(f"Connecting to ChromaDB at: {DB_PATH}")
client = chromadb.PersistentClient(path=DB_PATH)

# The app uses the collection name 'investment_book'
try:
    collection = client.get_collection("investment_book")
    count = collection.count()
    print(f"\n✅ Found collection 'investment_book' with {count} total chunks.")
    
    if count > 0:
        print("\n--- Displaying the first 2 chunks for Backend Verification ---")
        results = collection.get(limit=2, include=["documents", "metadatas", "embeddings"])
        
        for i in range(len(results["ids"])):
            print(f"\n🔹 Chunk ID: {results['ids'][i]}")
            print(f"🔹 Metadata: {results['metadatas'][i]}")
            print(f"🔹 Text Content:\n   {results['documents'][i][:200]}...\n")
            
            embedding = results['embeddings'][i]
            # Format a preview of the embedding list to show it contains the vector floats
            preview = ", ".join(f"{val:.6f}" for val in embedding[:10])
            print(f"🔹 Vector Embedding (Length: {len(embedding)} dimensions):\n   [{preview}, ...]")
            
except Exception as e:
    print(f"\n❌ Error accessing collection: {e}")
    # List collections to see what exactly exists
    collections = client.list_collections()
    print(f"Available collections: {[c.name for c in collections]}")
