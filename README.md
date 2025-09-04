# ZuneF.Com Chatbot với RAG Pipeline

## Mô tả
Chatbot thông minh sử dụng Google Gemini kết hợp với RAG (Retrieval-Augmented Generation) pipeline để trả lời câu hỏi về chính sách, điều khoản và thông tin của ZuneF.Com.

## Kiến trúc
```
User ↔ Chatbot ↔ RAG Pipeline ↔ Vector Store ↔ Data Folder
```

## Tính năng
- 🤖 Sử dụng Google Gemini Pro cho việc tạo phản hồi
- 🔍 RAG pipeline với ChromaDB vector store
- 📚 Truy xuất thông tin từ các tài liệu JSON trong folder Data
- 💬 Giao diện web thân thiện với Streamlit
- 🚫 Chỉ trả lời khi có dữ liệu liên quan

## Cài đặt

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 2. Cấu hình API Key
Tạo file `.env` với nội dung:
```
GEMINI_API_KEY=AIzaSyDWlcRSqEEcTzcaDsuKQlParf60gjgboFU
```

### 3. Chạy chatbot
```bash
python run_chatbot.py
```

Hoặc chạy trực tiếp:
```bash
streamlit run agent.py
```

## Cấu trúc dự án
```
SmartBot-Market/
├── agent.py                 # Main chatbot application
├── run_chatbot.py          # Launcher script
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables
├── README.md              # Documentation
└── Data/                  # JSON documents
    ├── privacy_policy.json
    ├── terms_and_use.json
    ├── Refund_policy.json
    └── Cookies_Policy.json
```

## Cách hoạt động

1. **Tải dữ liệu**: Hệ thống đọc tất cả file JSON trong folder Data
2. **Xử lý văn bản**: Chuyển đổi JSON thành văn bản có thể đọc được
3. **Tạo embeddings**: Sử dụng SentenceTransformer để tạo vector embeddings
4. **Lưu trữ vector**: Lưu trữ trong ChromaDB vector database
5. **Truy xuất**: Khi có câu hỏi, tìm kiếm các đoạn văn bản liên quan
6. **Tạo phản hồi**: Sử dụng Gemini để tạo câu trả lời dựa trên ngữ cảnh

## Sử dụng

1. Mở trình duyệt và truy cập `http://localhost:8501`
2. Nhập câu hỏi về chính sách, điều khoản của ZuneF.Com
3. Chatbot sẽ tìm kiếm thông tin liên quan và trả lời

## Ví dụ câu hỏi
- "Chính sách hoàn tiền như thế nào?"
- "Điều kiện sử dụng tài khoản là gì?"
- "Cookies được sử dụng để làm gì?"
- "Thông tin cá nhân được bảo vệ ra sao?"

## Lưu ý
- Chatbot chỉ trả lời dựa trên dữ liệu có trong folder Data
- Nếu không tìm thấy thông tin liên quan, chatbot sẽ thông báo không thể trả lời
- Cần có kết nối internet để sử dụng Gemini API