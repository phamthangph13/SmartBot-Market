import os
import json
import warnings
import chromadb
from sentence_transformers import SentenceTransformer
from google.generativeai import GenerativeModel
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict, Any
import streamlit as st

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*flash attention.*")
warnings.filterwarnings("ignore", message=".*oneDNN.*")

# Environment variables to suppress warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Load environment variables
load_dotenv(override=True)

# Fallback: Set API key directly if not found in environment
if not os.getenv('GEMINI_API_KEY'):
    os.environ['GEMINI_API_KEY'] = 'AIzaSyDWlcRSqEEcTzcaDsuKQlParf60gjgboFU'

class RAGAgent:
    def __init__(self):
        """Initialize the RAG Agent with Gemini and ChromaDB"""
        # Configure Gemini API
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=self.api_key)
        
        # Try to use the latest model, fallback to alternatives if needed
        try:
            self.model = GenerativeModel('gemini-1.5-flash')
            print("Using model: gemini-1.5-flash")
        except Exception as e:
            print(f"Failed to load gemini-1.5-flash: {e}")
            try:
                self.model = GenerativeModel('gemini-1.5-pro')
                print("Using model: gemini-1.5-pro")
            except Exception as e2:
                print(f"Failed to load gemini-1.5-pro: {e2}")
                # List available models for debugging
                try:
                    models = genai.list_models()
                    print("Available models:")
                    for model in models:
                        if 'generateContent' in model.supported_generation_methods:
                            print(f"  - {model.name}")
                except Exception as e3:
                    print(f"Could not list models: {e3}")
                raise ValueError("No suitable Gemini model found")
        
        # Initialize sentence transformer for embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB with telemetry disabled
        self.chroma_client = chromadb.Client(settings=chromadb.Settings(anonymized_telemetry=False))
        self.collection = self.chroma_client.create_collection(
            name="policy_documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Load and process documents
        self.load_documents()
    
    def load_documents(self):
        """Load and process all JSON documents from Data folder"""
        data_folder = "Data"
        if not os.path.exists(data_folder):
            raise FileNotFoundError(f"Data folder not found: {data_folder}")
        
        documents = []
        metadatas = []
        ids = []
        
        for filename in os.listdir(data_folder):
            if filename.endswith('.json'):
                file_path = os.path.join(data_folder, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract text content from JSON structure
                text_content = self.extract_text_from_json(data)
                
                # Split into chunks for better retrieval
                chunks = self.split_text_into_chunks(text_content, chunk_size=500)
                
                for i, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadatas.append({
                        "source": filename,
                        "title": data.get('title', ''),
                        "chunk_id": i
                    })
                    ids.append(f"{filename}_{i}")
        
        # Generate embeddings and add to ChromaDB
        if documents:
            embeddings = self.embedding_model.encode(documents).tolist()
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
    
    def extract_text_from_json(self, data: Dict[str, Any]) -> str:
        """Extract readable text from JSON structure"""
        text_parts = []
        
        # Add title and description
        if 'title' in data:
            text_parts.append(f"Tiêu đề: {data['title']}")
        if 'description' in data:
            text_parts.append(f"Mô tả: {data['description']}")
        if 'introduction' in data:
            text_parts.append(f"Giới thiệu: {data['introduction']}")
        
        # Process sections
        if 'sections' in data:
            for section in data['sections']:
                if 'title' in section:
                    text_parts.append(f"\n{section['title']}")
                
                if 'content' in section:
                    if isinstance(section['content'], list):
                        for item in section['content']:
                            if isinstance(item, str):
                                text_parts.append(item)
                            else:
                                text_parts.append(str(item))
                    else:
                        text_parts.append(str(section['content']))
                
                if 'subsections' in section:
                    for sub_key, sub_value in section['subsections'].items():
                        text_parts.append(f"\n{sub_key}:")
                        if isinstance(sub_value, list):
                            for item in sub_value:
                                if isinstance(item, str):
                                    text_parts.append(f"- {item}")
                                else:
                                    text_parts.append(f"- {str(item)}")
                        elif isinstance(sub_value, dict):
                            for k, v in sub_value.items():
                                if isinstance(v, list):
                                    text_parts.append(f"  {k}:")
                                    for item in v:
                                        text_parts.append(f"    - {str(item)}")
                                else:
                                    text_parts.append(f"  {k}: {str(v)}")
                        else:
                            text_parts.append(f"- {str(sub_value)}")
                
                # Handle other fields that might contain complex structures
                for key, value in section.items():
                    if key not in ['title', 'content', 'subsections']:
                        if isinstance(value, dict):
                            text_parts.append(f"\n{key}:")
                            for k, v in value.items():
                                if isinstance(v, list):
                                    text_parts.append(f"  {k}:")
                                    for item in v:
                                        if isinstance(item, dict):
                                            for item_k, item_v in item.items():
                                                text_parts.append(f"    {item_k}: {str(item_v)}")
                                        else:
                                            text_parts.append(f"    - {str(item)}")
                                elif isinstance(v, dict):
                                    text_parts.append(f"  {k}:")
                                    for item_k, item_v in v.items():
                                        text_parts.append(f"    {item_k}: {str(item_v)}")
                                else:
                                    text_parts.append(f"  {k}: {str(v)}")
                        elif isinstance(value, list):
                            text_parts.append(f"\n{key}:")
                            for item in value:
                                if isinstance(item, dict):
                                    for item_k, item_v in item.items():
                                        text_parts.append(f"  {item_k}: {str(item_v)}")
                                else:
                                    text_parts.append(f"- {str(item)}")
                        else:
                            text_parts.append(f"{key}: {str(value)}")
        
        return "\n".join(text_parts)
    
    def split_text_into_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into smaller chunks for better retrieval"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def retrieve_relevant_documents(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from ChromaDB"""
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        retrieved_docs = []
        for i in range(len(results['documents'][0])):
            retrieved_docs.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return retrieved_docs
    
    def generate_response(self, query: str) -> str:
        """Generate response using RAG pipeline"""
        # Retrieve relevant documents
        relevant_docs = self.retrieve_relevant_documents(query)
        
        if not relevant_docs:
            return "Xin lỗi, tôi không tìm thấy thông tin liên quan đến câu hỏi của bạn trong cơ sở dữ liệu."
        
        # Prepare context from retrieved documents
        context = "\n\n".join([
            f"Tài liệu: {doc['metadata']['source']}\n{doc['content']}"
            for doc in relevant_docs
        ])
        
        # Create prompt for Gemini
        prompt = f"""
Bạn là một trợ lý AI chuyên về chính sách và điều khoản của ZuneF.Com. 
Hãy trả lời câu hỏi của người dùng dựa trên thông tin được cung cấp bên dưới.

THÔNG TIN THAM KHẢO:
{context}

CÂU HỎI: {query}

HƯỚNG DẪN:
- Chỉ trả lời dựa trên thông tin có trong tài liệu tham khảo
- Nếu không có thông tin liên quan, hãy nói rõ "Tôi không tìm thấy thông tin về vấn đề này trong cơ sở dữ liệu"
- Trả lời bằng tiếng Việt, rõ ràng và dễ hiểu
- Nếu có thể, hãy trích dẫn nguồn tài liệu

TRẢ LỜI:
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Xin lỗi, có lỗi xảy ra khi xử lý câu hỏi: {str(e)}"

def main():
    """Main function for Streamlit interface"""
    st.set_page_config(
        page_title="ZuneF.Com Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 ZuneF.Com Chatbot")
    st.markdown("Hỏi đáp về chính sách, điều khoản và thông tin của ZuneF.Com")
    
    # Initialize RAG agent
    if 'agent' not in st.session_state:
        with st.spinner("Đang khởi tạo chatbot..."):
            try:
                st.session_state.agent = RAGAgent()
                st.success("Chatbot đã sẵn sàng!")
            except Exception as e:
                st.error(f"Lỗi khởi tạo: {str(e)}")
                return
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Đang tìm kiếm thông tin..."):
                response = st.session_state.agent.generate_response(prompt)
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
