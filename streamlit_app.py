import asyncio

# Đảm bảo rằng có vòng lặp sự kiện (Event Loop) khi sử dụng asyncio trong Streamlit
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
import os
import streamlit as st
# === Giao diện Streamlit ===
st.set_page_config(
    page_title="Tra cứu Điều lệ Đoàn",
    page_icon="https://quanlydoanvien.doanthanhnien.vn/favicon.ico",
    layout="wide"  # <== Bật chế độ wide mode
)
st.title("📘 Tra cứu Điều lệ Đoàn TNCS Hồ Chí Minh")

from dotenv import load_dotenv
# Tránh lỗi lazy loader của PyTorch khi dùng với Streamlit
os.environ["PYTORCH_NO_LAZY_LOADER"] = "1"

from llama_index.core import Settings, PromptTemplate
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage

# Load biến môi trường
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
model_name = "models/gemini-2.0-flash"

if not os.environ.get("GOOGLE_API_KEY"):
    st.error("❌ Chưa có GOOGLE_API_KEY trong .env! Hãy tạo file .env và thêm khóa.")
    st.stop()
# Cấu hình thư mục
DATA_DIR = "data_dieu_le_doan"
PERSIST_DIR = "doan_index_storage"
TTL = 24 * 60 * 60

# === Dùng cache để tránh reload model mỗi lần Streamlit refresh ===
@st.cache_data(ttl=TTL, show_spinner="Đang khởi tạo")
def load_embed_model_gemini():
    from llama_index.llms.google_genai import GoogleGenAI
    return GoogleGenAI(model=model_name, api_key=os.environ["GOOGLE_API_KEY"])
@st.cache_data(ttl=TTL, show_spinner="Đang khởi tạo")
def load_embed_model():
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    return HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

embed_model = load_embed_model()
llm = load_embed_model_gemini()

Settings.llm = llm
Settings.embed_model = embed_model

# Khởi tạo hoặc nạp chỉ mục
@st.cache_data(ttl=TTL, show_spinner="Đang khởi tạo")
def setup_index():
    if os.path.exists(os.path.join(PERSIST_DIR, "docstore.json")):
        print("🔁 Đang tải lại chỉ mục từ bộ nhớ...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        return load_index_from_storage(storage_context)
    else:
        print("🛠️ Đang tạo mới chỉ mục...")
        reader = SimpleDirectoryReader(input_dir=DATA_DIR)
        documents = reader.load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        return index

# === TẠO PROMPT TÙY CHỈNH ===
QA_PROMPT_TMPL = (
    "Bạn là một trợ lý hiểu biết về Điều lệ Đoàn TNCS Hồ Chí Minh. "
    "Trả lời các câu hỏi dưới đây một cách chi tiết, chính xác và sử dụng ngôn ngữ pháp lý. "
    "Nếu có thể, hãy trích dẫn cụ thể các điều, khoản trong Điều lệ Đoàn. "
    "Nếu câu hỏi vượt quá phạm vi tài liệu, bạn có thể tìm kiếm thêm từ các nguồn uy tín. "
    "Câu hỏi: {query_str}\n"
    "Tài liệu tham khảo:\n{context_str}\n\n"
    "Trả lời:"
)
qa_prompt = PromptTemplate(QA_PROMPT_TMPL)

index = setup_index()
query_engine = index.as_query_engine(similarity_top_k=4)
query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt})

# Hỏi Gemini tự do (fallback khi dữ liệu không đủ)
def ask_gemini_directly(question):
    try:
        return llm.complete(f"{question} Trong trường hợp này, bạn có thể tìm kiếm trên web để cung cấp thêm thông tin.").text
    except Exception as e:
        return f"❌ Đã xảy ra lỗi khi hỏi Gemini: {e}"

query = st.text_input("Nhập câu hỏi:", placeholder="Ví dụ: Quyền của đoàn viên là gì?", key="query_input")
submit = st.button("🧠 Trả lời") or (query and st.session_state.query_input)
if submit:
    with st.spinner("🔍 Đang tìm câu trả lời từ Điều lệ..."):
        response = query_engine.query(query)
        answer = response.response.strip()

        if len(answer) < 30:
            st.markdown("🌐 **Không đủ dữ liệu nội bộ, đang hỏi Gemini với tìm kiếm mở rộng...**")
            fallback = ask_gemini_directly(query)
            st.markdown(fallback)
        else:
            st.markdown("✅ **Trả lời từ Điều lệ Đoàn:**")
            st.markdown(answer)
