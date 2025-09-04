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
        
        # Prefer Gemini 2.x Flash-Lite models, with sensible fallbacks
        try:
            self.model = GenerativeModel('gemini-2.5-flash-lite')
            print("Using model: gemini-2.5-flash-lite")
        except Exception as e_25lite:
            print(f"Failed to load gemini-2.5-flash-lite: {e_25lite}")
            try:
                self.model = GenerativeModel('gemini-2.0-flash-lite')
                print("Using model: gemini-2.0-flash-lite")
            except Exception as e_20lite:
                print(f"Failed to load gemini-2.0-flash-lite: {e_20lite}")
                try:
                    self.model = GenerativeModel('gemini-2.0-flash')
                    print("Using model: gemini-2.0-flash")
                except Exception as e_20flash:
                    print(f"Failed to load gemini-2.0-flash: {e_20flash}")
                    try:
                        self.model = GenerativeModel('gemini-1.5-pro')
                        print("Using model: gemini-1.5-pro")
                    except Exception as e_pro:
                        print(f"Failed to load gemini-1.5-pro: {e_pro}")
                        try:
                            self.model = GenerativeModel('gemini-1.5-flash')
                            print("Using model: gemini-1.5-flash")
                        except Exception as e_flash:
                            print(f"Failed to load gemini-1.5-flash: {e_flash}")
                            # List available models for debugging
                            try:
                                models = genai.list_models()
                                print("Available models:")
                                for model in models:
                                    if 'generateContent' in model.supported_generation_methods:
                                        print(f"  - {model.name}")
                            except Exception as e_list:
                                print(f"Could not list models: {e_list}")
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
                
                # Extract text content from JSON structure (robust for dicts/lists/primitives)
                text_content = self.extract_text_from_json(data, source_name=filename)
                
                # Split into chunks for better retrieval
                chunks = self.split_text_into_chunks(text_content, chunk_size=500)
                
                for i, chunk in enumerate(chunks):
                    # Include filename context directly in chunk for better grounding
                    documents.append(f"Nguồn: {filename}\n{chunk}")
                    # Safely infer a title if present
                    inferred_title = ''
                    if isinstance(data, dict):
                        inferred_title = data.get('title') or data.get('name') or ''
                    metadatas.append({
                        "source": filename,
                        "title": inferred_title,
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
    
    # --------------------------
    # Structured data utilities
    # --------------------------
    def _load_json_file(self, filename: str) -> Any:
        file_path = os.path.join("Data", filename)
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None

    def list_categories(self) -> str:
        data = self._load_json_file('categories.json')
        if not isinstance(data, list) or not data:
            return "Tôi không tìm thấy danh mục nào trong cơ sở dữ liệu."
        lines = ["Danh mục sản phẩm hiện có (nguồn: categories.json):"]
        for item in data:
            if not isinstance(item, dict):
                continue
            name = item.get('name') or ''
            desc = item.get('description') or ''
            slug = item.get('slug') or ''
            if name:
                bullet = f"- {name}"
                details = []
                if desc:
                    details.append(desc)
                if slug:
                    details.append(f"slug: {slug}")
                if details:
                    bullet += f" — {'; '.join(details)}"
                lines.append(bullet)
        return "\n".join(lines)

    def list_accounts(self) -> str:
        data = self._load_json_file('accounts.json')
        if not isinstance(data, list) or not data:
            return "Tôi không tìm thấy danh sách tài khoản trong cơ sở dữ liệu."
        lines = ["Danh sách tài khoản dịch vụ AI (nguồn: accounts.json):"]
        for item in data:
            if not isinstance(item, dict):
                continue
            if (str(item.get('productType', '')).lower() != 'account'):
                continue
            name = item.get('name') or ''
            price = item.get('price')
            duration = item.get('duration') or ''
            category = item.get('category')
            category_text = ", ".join(category) if isinstance(category, list) else (category or '')
            parts = []
            if name:
                parts.append(name)
            if price is not None:
                parts.append(f"Giá: {price}")
            if duration:
                parts.append(f"Thời hạn: {duration}")
            if category_text:
                parts.append(f"Danh mục: {category_text}")
            if parts:
                lines.append("- " + " — ".join(parts))
        if len(lines) == 1:
            return "Tôi không tìm thấy mục tài khoản phù hợp trong cơ sở dữ liệu."
        return "\n".join(lines)

    def list_sourcecodes(self) -> str:
        data = self._load_json_file('sourcecodes.json')
        if not isinstance(data, list) or not data:
            return "Tôi không tìm thấy danh sách sản phẩm mã nguồn trong cơ sở dữ liệu."
        lines = ["Danh sách sản phẩm mã nguồn (nguồn: sourcecodes.json):"]
        for item in data:
            if not isinstance(item, dict):
                continue
            name = item.get('name') or ''
            price = item.get('price')
            discounted = item.get('discountedPrice')
            tags = item.get('tags')
            tags_text = ", ".join(tags) if isinstance(tags, list) else (tags or '')
            categories = item.get('category')
            if isinstance(categories, list):
                cat_names = []
                for c in categories:
                    if isinstance(c, dict) and c.get('name'):
                        cat_names.append(str(c.get('name')))
                    else:
                        cat_names.append(str(c))
                category_text = ", ".join(cat_names)
            else:
                category_text = str(categories) if categories is not None else ''
            parts = []
            if name:
                parts.append(name)
            if price is not None:
                parts.append(f"Giá: {price}")
            if discounted is not None:
                parts.append(f"Giá KM: {discounted}")
            if category_text:
                parts.append(f"Danh mục: {category_text}")
            if tags_text:
                parts.append(f"Tags: {tags_text}")
            if parts:
                lines.append("- " + " — ".join(parts))
        if len(lines) == 1:
            return "Tôi không tìm thấy mục mã nguồn trong cơ sở dữ liệu."
        return "\n".join(lines)
    
    def extract_text_from_json(self, data: Any, source_name: str = "") -> str:
        """Recursively extract readable text from any JSON structure (dict, list, primitives)."""
        lines: List[str] = []

        def is_primitive(value: Any) -> bool:
            return isinstance(value, (str, int, float, bool)) or value is None

        def normalize_primitive(value: Any) -> str:
            if value is None:
                return "null"
            return str(value)

        def flatten(node: Any, path_parts: List[str]):
            # If the node is a primitive, record it with its path
            if is_primitive(node):
                path_display = " / ".join(path_parts) if path_parts else "root"
                value_text = normalize_primitive(node)
                if value_text.strip() != "":
                    lines.append(f"{path_display}: {value_text}")
                return

            # If the node is a list, recurse into each element
            if isinstance(node, list):
                for index, item in enumerate(node):
                    flatten(item, path_parts + [str(index)])
                return

            # If the node is a dict, optionally add a concise product summary then recurse into each key
            if isinstance(node, dict):
                # Heuristic: product-like object
                keys = set(node.keys())
                has_name = 'name' in keys
                has_type = 'productType' in keys
                has_price = 'price' in keys
                has_category = 'category' in keys
                # Heuristic: identify source code objects
                lower_source = source_name.lower() if source_name else ""
                looks_like_source_code = (
                    'sourcecode' in lower_source or
                    'sourcecodes' in lower_source or
                    'source_code' in lower_source or
                    'sourceCodeFile' in keys
                )
                if has_name or has_type or has_price or has_category:
                    name_text = str(node.get('name', '')).strip()
                    type_text = str(node.get('productType', '')).strip()
                    category_field = node.get('category')
                    if isinstance(category_field, list):
                        category_text = ", ".join([str(c) for c in category_field])
                    else:
                        category_text = str(category_field) if category_field is not None else ''
                    price_value = node.get('price')
                    price_text = '' if price_value is None else str(price_value)
                    # Only emit summary if it looks meaningful
                    if name_text or type_text or category_text or price_text:
                        summary_parts = []
                        if name_text:
                            summary_parts.append(f"Tên: {name_text}")
                        if type_text:
                            # Map common types to Vietnamese-friendly labels
                            vi_type = 'tài khoản' if type_text.lower() == 'account' else type_text
                            summary_parts.append(f"Loại: {vi_type}")
                        elif looks_like_source_code:
                            summary_parts.append("Loại: mã nguồn (source code)")
                        if category_text:
                            summary_parts.append(f"Danh mục: {category_text}")
                        if price_text:
                            summary_parts.append(f"Giá: {price_text}")
                        if summary_parts:
                            lines.append("Sản phẩm: " + "; ".join(summary_parts))
                            if looks_like_source_code:
                                # Add common synonyms/keywords to aid retrieval
                                lines.append("Từ khóa: mã nguồn; source code; codebase; dự án; template")
                for key, value in node.items():
                    flatten(value, path_parts + [str(key)])
                return

            # Fallback for unexpected types
            lines.append(f"{' / '.join(path_parts) if path_parts else 'root'}: {str(node)}")

        flatten(data, [])
        return "\n".join(lines)
    
    def split_text_into_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into smaller chunks for better retrieval"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def retrieve_relevant_documents(self, query: str, n_results: int = 8) -> List[Dict[str, Any]]:
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
        # Intent-aware shortcuts for structured listings
        q = query.strip().lower()
        if any(kw in q for kw in ["liệt kê danh mục", "danh mục sản phẩm", "danh mục", "categories", "list categories"]):
            return self.list_categories()
        if any(kw in q for kw in ["liệt kê tài khoản", "danh sách tài khoản", "tài khoản dịch vụ ai", "accounts", "list accounts"]):
            return self.list_accounts()
        if any(kw in q for kw in ["liệt kê mã nguồn", "danh sách mã nguồn", "source code", "sourcecode", "list source codes"]):
            return self.list_sourcecodes()
 
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
Bạn là một trợ lý AI của ZuneF.Com, hỗ trợ trả lời về chính sách, điều khoản, danh mục, sản phẩm (tài khoản, source code) và các thông tin liên quan khác. 
Hãy trả lời câu hỏi của người dùng dựa trên thông tin được cung cấp bên dưới.

THÔNG TIN THAM KHẢO:
{context}

CÂU HỎI: {query}

HƯỚNG DẪN:
- Chỉ trả lời dựa trên thông tin có trong tài liệu tham khảo
- Nếu không có thông tin liên quan, hãy nói rõ "Tôi không tìm thấy thông tin về vấn đề này trong cơ sở dữ liệu"
- Trả lời bằng tiếng Việt, rõ ràng và dễ hiểu
- Nếu có thể, hãy trích dẫn nguồn tài liệu (tên file)

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
