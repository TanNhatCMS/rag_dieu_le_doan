# import asyncio

# Đảm bảo rằng có vòng lặp sự kiện (Event Loop) khi sử dụng asyncio trong Streamlit
# try:
#     loop = asyncio.get_running_loop()
# except RuntimeError:
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
# === Giao diện Streamlit ===
st.set_page_config(
    page_title="Tra cứu Điều lệ Đoàn",
    page_icon="https://quanlydoanvien.doanthanhnien.vn/favicon.ico",
    layout="wide"
)

# Tránh lỗi lazy loader của PyTorch khi dùng với Streamlit
os.environ["PYTORCH_NO_LAZY_LOADER"] = "1"

from llama_index.core import Settings, PromptTemplate
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage

# Cấu hình thư mục
DATA_DIR = "data"
PERSIST_DIR = "index_storage"
TTL = 24 * 60 * 60
# Replicate Credentials
with st.sidebar:
    st.title("📘 Tra cứu Điều lệ Đoàn TNCS Hồ Chí Minh")
    st.write('Chatbot hỗ trợ bởi Gemini.')
    if 'google' in st.secrets:
        st.success('Khóa API đã được cung cấp!', icon='✅')
        google_api_key = st.secrets["google"]["api_key"]
    else:
        google_api_key = st.text_input('Nhập Google API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Vui lòng nhập thông tin đăng nhập của bạn!', icon='⚠️')
        else:
            st.success('Tiến hành nhập tin nhắn nhắc nhở của bạn!', icon='👉')
    os.environ['GOOGLE_API_KEY'] = google_api_key
    st.subheader('Mô hình và thông số')
    selected_model = st.sidebar.selectbox('Chọn mẫu Gemini', ['Gemini 2.0 Flash', 'Gemini 2.0 Flash (Image Generation) Experimental'], key='selected_model')
    if selected_model == 'Gemini 2.0 Flash':
        model_name = 'gemini-2.0-flash'
    elif selected_model == 'Gemini 2.0 Flash (Image Generation) Experimental':
        model_name = 'gemini-2.0-flash-exp-image-generation'
# === Tải model từ HuggingFace Hub về local ===
def download_model_to_local(repo_id: str, local_dir: str = "models") -> None:
    from huggingface_hub import snapshot_download
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)

# === Dùng cache để tránh reload model mỗi lần Streamlit refresh ===
@st.cache_data(ttl=TTL, show_spinner="Đang khởi tạo GoogleGenAI")
def load_embed_model_gemini():
    from llama_index.llms.google_genai import GoogleGenAI
    return GoogleGenAI(model=model_name, api_key=google_api_key)

@st.cache_data(ttl=TTL, show_spinner="Đang khởi tạo Embedding")
def load_embed_model():
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    download_model_to_local(repo_id="sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
    # from google.genai.types import EmbedContentConfig
    # return GoogleGenAIEmbedding(
    #     model_name="text-embedding-004",
    #     api_key=google_api_key
    # )

embed_model = load_embed_model()
llm = load_embed_model_gemini()

Settings.llm = llm
Settings.embed_model = embed_model

# Khởi tạo hoặc nạp chỉ mục
@st.cache_data(ttl=TTL, show_spinner="Đang khởi tạo dữ liệu")
def setup_index():
    if os.path.exists(os.path.join(PERSIST_DIR, "docstore.json")):
        print("🔁 Đang tải lại chỉ mục từ bộ nhớ...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        return load_index_from_storage(storage_context)
    else:
        print("🛠️ Đang tạo mới chỉ mục...")
        reader = SimpleDirectoryReader(input_dir=DATA_DIR)
        documents = reader.load_data()
        vsi = VectorStoreIndex.from_documents(documents)
        vsi.storage_context.persist(persist_dir=PERSIST_DIR)
        return vsi

# === TẠO PROMPT TÙY CHỈNH ===
QA_PROMPT_TMPL = (
    "Bạn là một trợ lý hiểu biết về Điều lệ Đoàn TNCS Hồ Chí Minh. "
    "Trả lời các câu hỏi dưới đây một cách chi tiết, chính xác và sử dụng ngôn ngữ pháp lý. "
    "Nếu có thể, hãy trích dẫn cụ thể các điều, khoản trong Điều lệ Đoàn. "
    "Nếu câu hỏi vượt quá phạm vi tài liệu, bạn có thể tìm kiếm thêm từ các nguồn uy tín. "
    "\nCâu hỏi: {query_str}\n"
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

# === Chat UI ===
st.header("💬 Trò chuyện với AI Điều lệ Đoàn")

# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Chào bạn! Tôi có thể hỗ trợ gì về Điều lệ Đoàn TNCS Hồ Chí Minh?"}
    ]

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Chào bạn! Tôi có thể hỗ trợ gì về Điều lệ Đoàn TNCS Hồ Chí Minh?"}]
st.sidebar.button('Xóa lịch sử trò chuyện', on_click=clear_chat_history)

# Nhận đầu vào từ người dùng
if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Đang tìm câu trả lời..."):
            response = query_engine.query(prompt)
            answer = response.response.strip()
            if len(answer) < 30:
                fallback = ask_gemini_directly(prompt)
                st.markdown("🌐 **Không đủ dữ liệu nội bộ, đang hỏi Gemini với tìm kiếm mở rộng...**")
                st.markdown(fallback)
                st.session_state.messages.append({"role": "assistant", "content": fallback})
            else:
                st.markdown("✅ **Trả lời từ Điều lệ Đoàn:**")
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})