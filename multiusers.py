"""멀티유저 RAG 챗봇 — user 테이블 로그인 + Supabase 세션/벡터 DB."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import secrets
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import Client, create_client

# ──────────────────────────────────────────────
# 경로 및 환경 변수
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
LOGO_PATH = PROJECT_ROOT / "logo.png"

load_dotenv(dotenv_path=ENV_PATH)

MODEL_NAME = "gpt-4o-mini"
VECTOR_TABLE = "vector_documents"
VECTOR_QUERY = "match_vector_documents"
USER_TABLE = "user"
EMBED_BATCH_SIZE = 10
PBKDF2_ITERATIONS = 120_000

ANSWER_FORMAT_INSTRUCTION = """
답변은 반드시 헤딩(# ## ###)을 사용하여 구조화하세요.
주요 주제는 # (H1)로, 세부 내용은 ## (H2)로, 구체적 설명은 ### (H3)로 구분하세요.
답변은 서술형으로 작성하되 존대말을 사용하세요.
완전한 문장으로 서술하세요.
구분선(---, ===, ___) 사용 금지.
취소선(~~텍스트~~) 사용 금지.
참조 표시나 출처 문구 사용 금지.
"""

RAG_SYSTEM_PROMPT = (
    "너는 매우 친절한 선생님이야. 답변은 매우 쉽게 중학생 레벨에서 이해할 수 있도록 해줘. "
    "그러나 내용은 생략하는 것 없이 모두 답을 해줘. 모르면 모른다고 답해줘. 말투는 존대말 한글로 해줘. "
    + ANSWER_FORMAT_INSTRUCTION
)

DIRECT_LLM_SYSTEM_PROMPT = (
    "당신은 친절하고 유능한 AI 어시스턴트입니다. "
    + ANSWER_FORMAT_INSTRUCTION
)

FOLLOW_UP_SYSTEM_PROMPT = (
    "사용자와 AI의 대화를 바탕으로, 사용자가 이어서 물어볼 만한 질문 3개를 생성하세요. "
    "각 질문은 한 줄로 작성하고, 번호 없이 질문만 줄바꿈으로 구분하세요. "
    "질문만 출력하고 다른 설명은 하지 마세요."
)

TITLE_SYSTEM_PROMPT = (
    "대화의 첫 질문과 답변을 보고 세션 제목을 한글로 아주 짧게 만들어 주세요. "
    "최대 20자, 따옴표나 설명 없이 제목만 출력하세요."
)


# ──────────────────────────────────────────────
# 로깅
# ──────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"multiusers_{datetime.now().strftime('%Y%m%d')}.log"

    logger = logging.getLogger("multiusers_chatbot")
    logger.setLevel(logging.WARNING)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    for noisy_logger in (
        "httpx",
        "httpcore",
        "urllib3",
        "openai",
        "langchain",
        "langchain_openai",
        "supabase",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    return logger


LOGGER = setup_logging()


# ──────────────────────────────────────────────
# 설정 로드 (st.secrets > .env)
# ──────────────────────────────────────────────
def get_config_value(key: str) -> str:
    """우선순위: st.secrets → 환경변수(.env)."""
    try:
        if key in st.secrets:
            value = st.secrets[key]
            if value is not None and str(value).strip():
                return str(value).strip()
    except Exception:
        pass
    return (os.getenv(key) or "").strip()


def load_api_keys() -> dict[str, str]:
    return {
        "OPENAI_API_KEY": get_config_value("OPENAI_API_KEY"),
        "SUPABASE_URL": get_config_value("SUPABASE_URL"),
        "SUPABASE_ANON_KEY": get_config_value("SUPABASE_ANON_KEY"),
    }


def missing_keys_message(keys: dict[str, str] | None = None) -> str | None:
    keys = keys or load_api_keys()
    missing = [name for name, value in keys.items() if not value]
    if not missing:
        return None
    return (
        "다음 키가 설정되지 않았습니다: "
        + ", ".join(missing)
        + f"\nStreamlit Cloud는 Secrets, 로컬은 `{ENV_PATH}` 을 확인해 주세요."
    )


# ──────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────
def remove_separators(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"~~(.+?)~~", r"\1", text, flags=re.DOTALL)
    cleaned = re.sub(r"^[\-\=_]{3,}\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def format_memory_context(memory: list[dict[str, str]], limit: int = 50) -> str:
    recent = memory[-limit:]
    lines: list[str] = []
    for item in recent:
        role = "사용자" if item["role"] == "user" else "어시스턴트"
        lines.append(f"{role}: {item['content']}")
    return "\n".join(lines)


def append_follow_up_section(answer: str, follow_up_questions: list[str]) -> str:
    section_lines = ["### 💡 다음에 물어볼 수 있는 질문들"]
    for idx, question in enumerate(follow_up_questions[:3], start=1):
        section_lines.append(f"{idx}. {question.strip()}")
    return f"{answer.rstrip()}\n\n" + "\n".join(section_lines)


def parse_follow_up_questions(raw_text: str) -> list[str]:
    questions: list[str] = []
    for line in raw_text.splitlines():
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
        if cleaned:
            questions.append(cleaned)
    return questions[:3]


def hash_password(password: str, salt: str | None = None) -> str:
    """PBKDF2-HMAC-SHA256 해시. 형식: salt$hexdigest (평문 저장 금지)."""
    if salt is None:
        salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PBKDF2_ITERATIONS,
    )
    return f"{salt}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, _ = stored_hash.split("$", 1)
    except ValueError:
        return False
    return hmac.compare_digest(hash_password(password, salt), stored_hash)


def current_user_id() -> str | None:
    return st.session_state.get("user_id")


def require_user_id() -> str:
    user_id = current_user_id()
    if not user_id:
        raise RuntimeError("로그인이 필요합니다.")
    return user_id


# ──────────────────────────────────────────────
# Supabase / LLM
# ──────────────────────────────────────────────
@st.cache_resource
def get_supabase_client(supabase_url: str, supabase_anon_key: str) -> Client | None:
    if not supabase_url or not supabase_anon_key:
        return None
    return create_client(supabase_url, supabase_anon_key)


def get_llm(openai_api_key: str, temperature: float = 0.7) -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=temperature,
        openai_api_key=openai_api_key,
    )


def get_embeddings(openai_api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(openai_api_key=openai_api_key)


def get_vector_store(client: Client, openai_api_key: str) -> SupabaseVectorStore:
    return SupabaseVectorStore(
        client=client,
        embedding=get_embeddings(openai_api_key),
        table_name=VECTOR_TABLE,
        query_name=VECTOR_QUERY,
    )


# ──────────────────────────────────────────────
# 사용자 인증 (user 테이블)
# ──────────────────────────────────────────────
def find_user_by_login_id(client: Client, login_id: str) -> dict[str, Any] | None:
    response = (
        client.table(USER_TABLE)
        .select("id, login_id, password_hash, created_at")
        .eq("login_id", login_id.strip())
        .limit(1)
        .execute()
    )
    rows = response.data or []
    return rows[0] if rows else None


def register_user(client: Client, login_id: str, password: str) -> tuple[bool, str]:
    login_id = login_id.strip()
    if not login_id or not password:
        return False, "아이디와 비밀번호를 모두 입력해 주세요."
    if len(login_id) < 3:
        return False, "아이디는 3자 이상이어야 합니다."
    if len(password) < 4:
        return False, "비밀번호는 4자 이상이어야 합니다."

    if find_user_by_login_id(client, login_id):
        return False, "이미 사용 중인 아이디입니다."

    payload = {
        "login_id": login_id,
        "password_hash": hash_password(password),
    }
    try:
        response = client.table(USER_TABLE).insert(payload).execute()
        if not response.data:
            return False, "회원가입에 실패했습니다."
        return True, "회원가입이 완료되었습니다. 로그인해 주세요."
    except Exception as exc:
        LOGGER.error("회원가입 실패: %s", exc)
        return False, f"회원가입 중 오류: {exc}"


def login_user(client: Client, login_id: str, password: str) -> tuple[bool, str]:
    login_id = login_id.strip()
    if not login_id or not password:
        return False, "아이디와 비밀번호를 모두 입력해 주세요."

    user = find_user_by_login_id(client, login_id)
    if not user:
        return False, "아이디 또는 비밀번호가 올바르지 않습니다."
    if not verify_password(password, user.get("password_hash") or ""):
        return False, "아이디 또는 비밀번호가 올바르지 않습니다."

    st.session_state.logged_in = True
    st.session_state.user_id = str(user["id"])
    st.session_state.login_id = str(user["login_id"])
    reset_chat_screen()
    return True, "로그인 성공"


def logout_user() -> None:
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.login_id = None
    reset_chat_screen(clear_session_list=True)


# ──────────────────────────────────────────────
# 세션 DB CRUD (항상 user_id 필터)
# ──────────────────────────────────────────────
def fetch_sessions(client: Client, user_id: str) -> list[dict[str, Any]]:
    response = (
        client.table("chat_sessions")
        .select("id, title, processed_files, created_at, updated_at, user_id")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .execute()
    )
    return list(response.data or [])


def create_session(
    client: Client,
    user_id: str,
    title: str = "새 세션",
    processed_files: list[str] | None = None,
) -> str:
    payload = {
        "user_id": user_id,
        "title": title,
        "processed_files": processed_files or [],
    }
    response = client.table("chat_sessions").insert(payload).execute()
    if not response.data:
        raise RuntimeError("세션 생성에 실패했습니다.")
    return str(response.data[0]["id"])


def update_session_meta(
    client: Client,
    user_id: str,
    session_id: str,
    title: str | None = None,
    processed_files: list[str] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if title is not None:
        payload["title"] = title
    if processed_files is not None:
        payload["processed_files"] = processed_files
    (
        client.table("chat_sessions")
        .update(payload)
        .eq("id", session_id)
        .eq("user_id", user_id)
        .execute()
    )


def delete_session(client: Client, user_id: str, session_id: str) -> None:
    (
        client.table("chat_sessions")
        .delete()
        .eq("id", session_id)
        .eq("user_id", user_id)
        .execute()
    )


def replace_messages(
    client: Client,
    user_id: str,
    session_id: str,
    chat_history: list[dict[str, str]],
) -> None:
    (
        client.table("chat_messages")
        .delete()
        .eq("session_id", session_id)
        .eq("user_id", user_id)
        .execute()
    )
    if not chat_history:
        return
    rows = [
        {
            "session_id": session_id,
            "user_id": user_id,
            "role": msg["role"],
            "content": msg["content"],
        }
        for msg in chat_history
    ]
    client.table("chat_messages").insert(rows).execute()


def load_messages(client: Client, user_id: str, session_id: str) -> list[dict[str, str]]:
    response = (
        client.table("chat_messages")
        .select("role, content, created_at")
        .eq("session_id", session_id)
        .eq("user_id", user_id)
        .order("created_at")
        .execute()
    )
    return [
        {"role": row["role"], "content": row["content"]}
        for row in (response.data or [])
    ]


def load_session_files(client: Client, user_id: str, session_id: str) -> list[str]:
    response = (
        client.table("chat_sessions")
        .select("processed_files")
        .eq("id", session_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    if not response.data:
        return []
    files = response.data[0].get("processed_files") or []
    return list(files)


def session_belongs_to_user(client: Client, user_id: str, session_id: str) -> bool:
    response = (
        client.table("chat_sessions")
        .select("id")
        .eq("id", session_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    return bool(response.data)


def list_vector_file_names(client: Client, session_id: str | None = None) -> list[str]:
    query = client.table(VECTOR_TABLE).select("file_name")
    if session_id:
        query = query.eq("session_id", session_id)
    response = query.execute()
    names = sorted({row["file_name"] for row in (response.data or []) if row.get("file_name")})
    return names


def generate_session_title(openai_api_key: str, user_query: str, answer: str) -> str:
    try:
        llm = get_llm(openai_api_key, temperature=0.3)
        messages = [
            SystemMessage(content=TITLE_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"첫 질문:\n{user_query}\n\n"
                    f"첫 답변:\n{answer[:800]}\n\n"
                    "세션 제목을 만들어 주세요."
                )
            ),
        ]
        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        title = remove_separators(str(content)).splitlines()[0].strip().strip("\"'`")
        return title[:20] or "저장된 세션"
    except Exception as exc:
        LOGGER.warning("세션 제목 생성 실패: %s", exc)
        return (user_query[:20] or "저장된 세션").strip()


def ensure_current_session(client: Client, user_id: str) -> str:
    if st.session_state.current_session_id:
        return st.session_state.current_session_id
    session_id = create_session(
        client,
        user_id=user_id,
        title="새 세션",
        processed_files=st.session_state.processed_files,
    )
    st.session_state.current_session_id = session_id
    return session_id


def autosave_session(client: Client, user_id: str) -> None:
    try:
        session_id = ensure_current_session(client, user_id)
        replace_messages(client, user_id, session_id, st.session_state.chat_history)
        update_session_meta(
            client,
            user_id,
            session_id,
            processed_files=st.session_state.processed_files,
        )
        st.session_state.sessions = fetch_sessions(client, user_id)
    except Exception as exc:
        LOGGER.error("자동 저장 실패: %s", exc)
        st.session_state.last_error = f"자동 저장 실패: {exc}"


def insert_session_snapshot(client: Client, user_id: str, openai_api_key: str) -> str | None:
    if not st.session_state.chat_history and not st.session_state.processed_files:
        st.warning("저장할 대화 또는 파일이 없습니다.")
        return None

    first_user = next(
        (m["content"] for m in st.session_state.chat_history if m["role"] == "user"),
        "새 세션",
    )
    first_assistant = next(
        (m["content"] for m in st.session_state.chat_history if m["role"] == "assistant"),
        "",
    )
    title = generate_session_title(openai_api_key, first_user, first_assistant)

    new_session_id = create_session(
        client,
        user_id=user_id,
        title=title,
        processed_files=st.session_state.processed_files,
    )
    replace_messages(client, user_id, new_session_id, st.session_state.chat_history)

    source_id = st.session_state.current_session_id
    if source_id and session_belongs_to_user(client, user_id, source_id):
        copy_vector_documents(client, source_id, new_session_id)

    st.session_state.current_session_id = new_session_id
    st.session_state.sessions = fetch_sessions(client, user_id)
    st.session_state.selected_session_label = f"{title} ({new_session_id[:8]})"
    return new_session_id


def copy_vector_documents(client: Client, source_session_id: str, target_session_id: str) -> None:
    response = (
        client.table(VECTOR_TABLE)
        .select("content, metadata, embedding, file_name")
        .eq("session_id", source_session_id)
        .execute()
    )
    rows = response.data or []
    if not rows:
        return

    payload = []
    for row in rows:
        file_name = row.get("file_name")
        if not file_name:
            continue
        metadata = dict(row.get("metadata") or {})
        metadata["session_id"] = target_session_id
        metadata["file_name"] = file_name
        payload.append(
            {
                "id": str(uuid.uuid4()),
                "content": row.get("content"),
                "metadata": metadata,
                "embedding": row.get("embedding"),
                "file_name": file_name,
                "session_id": target_session_id,
            }
        )

    for start in range(0, len(payload), EMBED_BATCH_SIZE):
        batch = payload[start : start + EMBED_BATCH_SIZE]
        client.table(VECTOR_TABLE).insert(batch).execute()


def apply_loaded_session(client: Client, user_id: str, session_id: str) -> None:
    if not session_belongs_to_user(client, user_id, session_id):
        raise PermissionError("다른 사용자의 세션에는 접근할 수 없습니다.")

    messages = load_messages(client, user_id, session_id)
    files = load_session_files(client, user_id, session_id)
    st.session_state.current_session_id = session_id
    st.session_state.chat_history = messages
    st.session_state.conversation_memory = messages[-50:]
    st.session_state.processed_files = files
    st.session_state.vector_ready = bool(files)
    st.session_state.has_vector_store = len(list_vector_file_names(client, session_id)) > 0


# ──────────────────────────────────────────────
# PDF → Supabase 직접 저장
# ──────────────────────────────────────────────
def process_pdf_files_to_supabase(
    client: Client,
    uploaded_files: list[Any],
    session_id: str,
    openai_api_key: str,
) -> tuple[list[str], str | None]:
    if not openai_api_key:
        return [], "OPENAI_API_KEY가 설정되지 않았습니다."

    embeddings = get_embeddings(openai_api_key)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    processed_names: list[str] = []

    for uploaded_file in uploaded_files:
        tmp_path = None
        file_name = uploaded_file.name
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            if not docs:
                return [], f"'{file_name}'에서 텍스트를 추출하지 못했습니다."

            chunks = splitter.split_documents(docs)
            texts = [c.page_content for c in chunks]
            vectors = embeddings.embed_documents(texts)

            rows: list[dict[str, Any]] = []
            for chunk, vector in zip(chunks, vectors):
                metadata = dict(chunk.metadata or {})
                metadata["file_name"] = file_name
                metadata["session_id"] = session_id
                rows.append(
                    {
                        "id": str(uuid.uuid4()),
                        "content": chunk.page_content,
                        "metadata": metadata,
                        "embedding": vector,
                        "file_name": file_name,
                        "session_id": session_id,
                    }
                )

            for start in range(0, len(rows), EMBED_BATCH_SIZE):
                batch = rows[start : start + EMBED_BATCH_SIZE]
                client.table(VECTOR_TABLE).insert(batch).execute()

            processed_names.append(file_name)
        except Exception as exc:
            LOGGER.error("PDF 처리 실패 (%s): %s", file_name, exc)
            return [], f"'{file_name}' 파일 처리 중 오류가 발생했습니다: {exc}"
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    try:
        get_vector_store(client, openai_api_key)
    except Exception as exc:
        LOGGER.warning("SupabaseVectorStore 초기화 경고: %s", exc)

    return processed_names, None


def retrieve_documents(
    client: Client,
    query: str,
    session_id: str,
    openai_api_key: str,
    k: int = 10,
) -> list[Document]:
    embeddings = get_embeddings(openai_api_key)
    query_embedding = embeddings.embed_query(query)

    try:
        response = client.rpc(
            VECTOR_QUERY,
            {
                "query_embedding": query_embedding,
                "match_count": k,
                "filter_session_id": session_id,
            },
        ).execute()
        docs: list[Document] = []
        for row in response.data or []:
            metadata = dict(row.get("metadata") or {})
            if row.get("file_name"):
                metadata["file_name"] = row["file_name"]
            if row.get("session_id"):
                metadata["session_id"] = row["session_id"]
            docs.append(
                Document(
                    page_content=row.get("content") or "",
                    metadata=metadata,
                )
            )
        return docs
    except Exception as exc:
        LOGGER.warning("RPC 검색 실패, 대체 검색 사용: %s", exc)

    try:
        response = (
            client.table(VECTOR_TABLE)
            .select("content, metadata, file_name, session_id, embedding")
            .eq("session_id", session_id)
            .execute()
        )
        rows = response.data or []
        if not rows:
            return []

        scored: list[tuple[float, Document]] = []
        for row in rows:
            emb = row.get("embedding")
            if not emb:
                continue
            if isinstance(emb, str):
                emb = [float(x) for x in emb.strip("[]").split(",") if x.strip()]
            dot = sum(a * b for a, b in zip(query_embedding, emb))
            norm_q = sum(a * a for a in query_embedding) ** 0.5
            norm_e = sum(b * b for b in emb) ** 0.5
            if norm_q == 0 or norm_e == 0:
                continue
            similarity = dot / (norm_q * norm_e)
            metadata = dict(row.get("metadata") or {})
            if row.get("file_name"):
                metadata["file_name"] = row["file_name"]
            metadata["session_id"] = session_id
            scored.append(
                (
                    similarity,
                    Document(
                        page_content=row.get("content") or "",
                        metadata=metadata,
                    ),
                )
            )
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:k]]
    except Exception as exc:
        LOGGER.error("대체 검색 실패: %s", exc)
        return []


# ──────────────────────────────────────────────
# 답변 생성 (stream)
# ──────────────────────────────────────────────
def generate_follow_up_questions(llm: Any, user_query: str, answer: str) -> list[str]:
    try:
        messages = [
            SystemMessage(content=FOLLOW_UP_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"사용자 질문:\n{user_query}\n\n"
                    f"AI 답변:\n{answer}\n\n"
                    "위 대화를 바탕으로 후속 질문 3개를 생성하세요."
                )
            ),
        ]
        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        questions = parse_follow_up_questions(str(content))
        while len(questions) < 3:
            questions.append("이 주제에 대해 더 자세히 설명해 주실 수 있나요?")
        return questions[:3]
    except Exception as exc:
        LOGGER.warning("후속 질문 생성 실패: %s", exc)
        return [
            "이 내용을 더 쉽게 설명해 주실 수 있나요?",
            "관련된 다른 주제도 알려 주실 수 있나요?",
            "실생활에서 어떻게 활용할 수 있나요?",
        ]


def stream_llm_response(llm: Any, messages: list[Any], placeholder: Any) -> str:
    full_response = ""
    for chunk in llm.stream(messages):
        piece = chunk.content if hasattr(chunk, "content") else str(chunk)
        if isinstance(piece, list):
            piece = "".join(
                getattr(p, "text", str(p)) if not isinstance(p, str) else p for p in piece
            )
        if piece:
            full_response += piece
            placeholder.markdown(remove_separators(full_response))
    return remove_separators(full_response)


def generate_direct_llm_answer(
    llm: Any,
    user_query: str,
    conversation_memory: list[dict[str, str]],
    placeholder: Any,
) -> str:
    memory_context = format_memory_context(conversation_memory)
    messages = [
        SystemMessage(content=DIRECT_LLM_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"이전 대화:\n{memory_context}\n\n"
                f"현재 질문:\n{user_query}"
            )
        ),
    ]
    answer = stream_llm_response(llm, messages, placeholder)
    follow_up = generate_follow_up_questions(llm, user_query, answer)
    final_answer = append_follow_up_section(answer, follow_up)
    placeholder.markdown(final_answer)
    return final_answer


def generate_rag_answer(
    client: Client,
    llm: Any,
    user_query: str,
    session_id: str,
    conversation_memory: list[dict[str, str]],
    placeholder: Any,
    openai_api_key: str,
) -> str:
    docs = retrieve_documents(client, user_query, session_id, openai_api_key, k=10)
    context = "\n\n".join(doc.page_content for doc in docs) if docs else "(관련 문서 없음)"
    memory_context = format_memory_context(conversation_memory)

    messages = [
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"이전 대화:\n{memory_context}\n\n"
                f"참고 문서:\n{context}\n\n"
                f"질문:\n{user_query}"
            )
        ),
    ]
    answer = stream_llm_response(llm, messages, placeholder)
    follow_up = generate_follow_up_questions(llm, user_query, answer)
    final_answer = append_follow_up_section(answer, follow_up)
    placeholder.markdown(final_answer)
    return final_answer


def update_conversation_memory(
    user_query: str,
    assistant_answer: str,
    conversation_memory: list[dict[str, str]],
) -> None:
    conversation_memory.append({"role": "user", "content": user_query})
    conversation_memory.append({"role": "assistant", "content": assistant_answer})
    if len(conversation_memory) > 50:
        del conversation_memory[:-50]


# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────
def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        h1 { color: #ff69b4 !important; font-size: 1.9rem !important; }
        h2 { color: #ffd700 !important; font-size: 1.6rem !important; }
        h3 { color: #1f77b4 !important; font-size: 1.35rem !important; }

        div[data-testid="stChatMessage"] {
            border-radius: 12px;
            padding: 0.5rem 0.75rem;
            margin-bottom: 0.75rem;
            font-size: 1.25rem !important;
            line-height: 1.7 !important;
        }

        div[data-testid="stChatMessage"] p,
        div[data-testid="stChatMessage"] li,
        div[data-testid="stChatMessage"] span,
        div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] {
            font-size: 1.25rem !important;
            line-height: 1.7 !important;
        }

        div[data-testid="stChatInput"] textarea {
            font-size: 1.2rem !important;
        }

        div.stButton > button {
            background-color: #ff69b4 !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
        }

        div.stButton > button:hover {
            background-color: #ff85c1 !important;
            color: white !important;
        }

        .ena-header-title {
            text-align: center !important;
            font-size: 2.4rem !important;
            line-height: 1.1 !important;
            font-weight: 700 !important;
            margin: 0.5rem 0 1rem 0 !important;
        }

        .ena-header-title .ena-blue {
            color: #1f77b4 !important;
            font-size: 2.4rem !important;
        }

        .ena-header-title .ena-gold {
            color: #ffd700 !important;
            font-size: 2.4rem !important;
        }

        .ena-auth-box {
            max-width: 420px;
            margin: 1.5rem auto;
            padding: 1.25rem 1.5rem;
            border-radius: 12px;
            background: rgba(31, 119, 180, 0.06);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    left_col, center_col, right_col = st.columns([1, 2, 1])

    with left_col:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=180)
        else:
            st.markdown("## 📚")

    with center_col:
        st.markdown(
            """
            <div class="ena-header-title">
                <span class="ena-blue">ENA</span>
                <span class="ena-gold">RAG 챗봇</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right_col:
        st.empty()


def init_session_state() -> None:
    defaults: dict[str, Any] = {
        "logged_in": False,
        "user_id": None,
        "login_id": None,
        "auth_mode": "로그인",
        "chat_history": [],
        "conversation_memory": [],
        "processed_files": [],
        "current_session_id": None,
        "sessions": [],
        "selected_session_label": "— 새 세션 —",
        "prev_selected_session_label": "— 새 세션 —",
        "has_vector_store": False,
        "vector_ready": False,
        "show_vectordb": False,
        "vectordb_files": [],
        "last_error": None,
        "initialized": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def session_label(session: dict[str, Any]) -> str:
    return f"{session['title']} ({str(session['id'])[:8]})"


def resolve_session_id_from_label(label: str) -> str | None:
    if not label or label.startswith("—"):
        return None
    for session in st.session_state.sessions:
        if session_label(session) == label:
            return str(session["id"])
    return None


def reset_chat_screen(*, clear_session_list: bool = False) -> None:
    st.session_state.chat_history = []
    st.session_state.conversation_memory = []
    st.session_state.processed_files = []
    st.session_state.current_session_id = None
    if clear_session_list:
        st.session_state.sessions = []
    st.session_state.has_vector_store = False
    st.session_state.vector_ready = False
    st.session_state.selected_session_label = "— 새 세션 —"
    st.session_state.prev_selected_session_label = "— 새 세션 —"
    st.session_state.show_vectordb = False
    st.session_state.vectordb_files = []
    st.session_state.last_error = None


def render_chat_history() -> None:
    for message in st.session_state.chat_history:
        role = "user" if message["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(message["content"])


def render_auth_page(client: Client | None, key_msg: str | None) -> None:
    render_header()
    if key_msg:
        st.warning(key_msg)

    st.markdown('<div class="ena-auth-box">', unsafe_allow_html=True)
    st.subheader("🔐 로그인 / 회원가입")
    st.caption("Supabase Auth가 아닌 앱의 user 테이블로 계정을 관리합니다.")

    mode = st.radio(
        "모드 선택",
        ["로그인", "회원가입"],
        horizontal=True,
        key="auth_mode_radio",
    )
    st.session_state.auth_mode = mode

    login_id = st.text_input("아이디 (login_id)", key="auth_login_id")
    password = st.text_input("비밀번호", type="password", key="auth_password")

    if mode == "로그인":
        if st.button("로그인", use_container_width=True, type="primary"):
            if client is None:
                st.error(key_msg or "Supabase 연결에 실패했습니다.")
            else:
                ok, msg = login_user(client, login_id, password)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
    else:
        if st.button("회원가입", use_container_width=True, type="primary"):
            if client is None:
                st.error(key_msg or "Supabase 연결에 실패했습니다.")
            else:
                ok, msg = register_user(client, login_id, password)
                if ok:
                    st.success(msg)
                    st.session_state.auth_mode = "로그인"
                else:
                    st.error(msg)

    st.markdown("</div>", unsafe_allow_html=True)


def render_sidebar(
    client: Client | None,
    openai_api_key: str,
    key_msg: str | None,
) -> str:
    user_id = require_user_id()

    st.sidebar.header("⚙️ 설정")
    st.sidebar.markdown(f"**로그인:** `{st.session_state.login_id}`")
    if st.sidebar.button("로그아웃", use_container_width=True):
        logout_user()
        st.rerun()

    st.sidebar.markdown(f"**LLM 모델:** `{MODEL_NAME}`")

    rag_option = st.sidebar.radio(
        "RAG (PDF 검색) 선택",
        ["사용 안 함", "RAG 사용"],
        index=0,
    )

    uploaded_files = st.sidebar.file_uploader(
        "PDF 파일 업로드",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.sidebar.button("파일 처리하기"):
        if client is None:
            st.sidebar.error(key_msg or "Supabase 연결에 실패했습니다.")
        elif not uploaded_files:
            st.sidebar.warning("업로드할 PDF 파일을 선택해 주세요.")
        else:
            with st.sidebar.spinner("PDF를 임베딩하고 Supabase에 저장하는 중..."):
                try:
                    session_id = ensure_current_session(client, user_id)
                    names, error = process_pdf_files_to_supabase(
                        client, uploaded_files, session_id, openai_api_key
                    )
                    if error:
                        st.sidebar.error(error)
                    else:
                        merged = list(
                            dict.fromkeys(st.session_state.processed_files + names)
                        )
                        st.session_state.processed_files = merged
                        st.session_state.has_vector_store = True
                        st.session_state.vector_ready = True
                        update_session_meta(
                            client, user_id, session_id, processed_files=merged
                        )
                        autosave_session(client, user_id)
                        st.sidebar.success(
                            f"{len(names)}개 PDF 파일 처리가 완료되었습니다."
                        )
                except Exception as exc:
                    LOGGER.error("파일 처리 실패: %s", exc)
                    st.sidebar.error(f"파일 처리 중 오류: {exc}")

    if st.session_state.processed_files:
        st.sidebar.write("처리된 파일:")
        for file_name in st.session_state.processed_files:
            st.sidebar.write(f"- {file_name}")

    st.sidebar.divider()
    st.sidebar.subheader("📁 세션 관리")

    if client is not None:
        try:
            st.session_state.sessions = fetch_sessions(client, user_id)
        except Exception as exc:
            st.sidebar.error(f"세션 목록 로드 실패: {exc}")

    options = ["— 새 세션 —"] + [
        session_label(s) for s in st.session_state.sessions
    ]
    current_label = st.session_state.selected_session_label
    if current_label not in options:
        current_label = options[0]

    selected = st.sidebar.selectbox(
        "세션 선택",
        options,
        index=options.index(current_label),
        key="session_selectbox",
    )

    if selected != st.session_state.prev_selected_session_label:
        st.session_state.selected_session_label = selected
        st.session_state.prev_selected_session_label = selected
        if client is not None and not selected.startswith("—"):
            sid = resolve_session_id_from_label(selected)
            if sid:
                try:
                    apply_loaded_session(client, user_id, sid)
                    st.sidebar.success("세션을 불러왔습니다.")
                    st.rerun()
                except Exception as exc:
                    st.sidebar.error(f"세션 로드 실패: {exc}")
        elif selected.startswith("—"):
            reset_chat_screen()
            st.rerun()

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("세션저장", use_container_width=True):
            if client is None:
                st.sidebar.error(key_msg or "Supabase 연결 실패")
            else:
                with st.spinner("세션을 저장하는 중..."):
                    try:
                        new_id = insert_session_snapshot(client, user_id, openai_api_key)
                        if new_id:
                            st.sidebar.success("세션이 저장되었습니다.")
                            st.rerun()
                    except Exception as exc:
                        st.sidebar.error(f"세션 저장 실패: {exc}")

        if st.button("세션삭제", use_container_width=True):
            if client is None:
                st.sidebar.error(key_msg or "Supabase 연결 실패")
            else:
                sid = st.session_state.current_session_id
                if not sid:
                    st.sidebar.warning("삭제할 세션이 없습니다.")
                else:
                    try:
                        delete_session(client, user_id, sid)
                        reset_chat_screen()
                        st.session_state.sessions = fetch_sessions(client, user_id)
                        st.sidebar.success("세션을 삭제했습니다.")
                        st.rerun()
                    except Exception as exc:
                        st.sidebar.error(f"세션 삭제 실패: {exc}")

    with col2:
        if st.button("세션로드", use_container_width=True):
            if client is None:
                st.sidebar.error(key_msg or "Supabase 연결 실패")
            else:
                sid = resolve_session_id_from_label(st.session_state.selected_session_label)
                if not sid:
                    st.sidebar.warning("로드할 세션을 선택해 주세요.")
                else:
                    try:
                        apply_loaded_session(client, user_id, sid)
                        st.sidebar.success("세션을 불러왔습니다.")
                        st.rerun()
                    except Exception as exc:
                        st.sidebar.error(f"세션 로드 실패: {exc}")

        if st.button("화면초기화", use_container_width=True):
            reset_chat_screen()
            st.rerun()

    if st.sidebar.button("vectordb", use_container_width=True):
        if client is None:
            st.sidebar.error(key_msg or "Supabase 연결 실패")
        else:
            try:
                files = list_vector_file_names(
                    client, st.session_state.current_session_id
                )
                st.session_state.vectordb_files = files
                st.session_state.show_vectordb = True
            except Exception as exc:
                st.sidebar.error(f"vectordb 조회 실패: {exc}")

    if st.session_state.show_vectordb:
        st.sidebar.markdown("**Vector DB 파일**")
        if st.session_state.vectordb_files:
            for name in st.session_state.vectordb_files:
                st.sidebar.write(f"- {name}")
        else:
            st.sidebar.info("저장된 벡터 파일이 없습니다.")

    st.sidebar.divider()
    st.sidebar.subheader("현재 설정")
    st.sidebar.text(f"모델: {MODEL_NAME}")
    st.sidebar.text(f"RAG: {rag_option}")
    st.sidebar.text(f"사용자: {st.session_state.login_id}")
    st.sidebar.text(f"처리된 파일 수: {len(st.session_state.processed_files)}")
    st.sidebar.text(f"대화 기록 수: {len(st.session_state.chat_history)}")
    sid_short = (st.session_state.current_session_id or "-")[:8]
    st.sidebar.text(f"현재 세션: {sid_short}")

    return rag_option


def handle_user_input(
    user_query: str,
    rag_option: str,
    client: Client | None,
    openai_api_key: str,
    key_msg: str | None,
) -> None:
    user_id = require_user_id()
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    if key_msg:
        error_message = (
            f"⚠️ {key_msg}\n\n"
            "Secrets 또는 `.env` 파일에 필요한 키를 설정한 뒤 다시 시도해 주세요."
        )
        st.session_state.chat_history.append({"role": "assistant", "content": error_message})
        update_conversation_memory(
            user_query, error_message, st.session_state.conversation_memory
        )
        with st.chat_message("assistant"):
            st.error(error_message)
        return

    assert client is not None

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            llm = get_llm(openai_api_key)
            if rag_option == "RAG 사용":
                session_id = ensure_current_session(client, user_id)
                file_count = len(list_vector_file_names(client, session_id))
                if file_count == 0:
                    warning_message = (
                        "⚠️ RAG를 사용하려면 먼저 PDF 파일을 업로드하고 "
                        "'파일 처리하기' 버튼을 눌러 주세요."
                    )
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": warning_message}
                    )
                    update_conversation_memory(
                        user_query, warning_message, st.session_state.conversation_memory
                    )
                    placeholder.warning(warning_message)
                    return

                final_answer = generate_rag_answer(
                    client=client,
                    llm=llm,
                    user_query=user_query,
                    session_id=session_id,
                    conversation_memory=st.session_state.conversation_memory,
                    placeholder=placeholder,
                    openai_api_key=openai_api_key,
                )
            else:
                final_answer = generate_direct_llm_answer(
                    llm=llm,
                    user_query=user_query,
                    conversation_memory=st.session_state.conversation_memory,
                    placeholder=placeholder,
                )

            st.session_state.chat_history.append(
                {"role": "assistant", "content": final_answer}
            )
            update_conversation_memory(
                user_query, final_answer, st.session_state.conversation_memory
            )

            session_id = ensure_current_session(client, user_id)
            user_msgs = [
                m for m in st.session_state.chat_history if m["role"] == "user"
            ]
            if len(user_msgs) == 1:
                title = generate_session_title(openai_api_key, user_query, final_answer)
                update_session_meta(client, user_id, session_id, title=title)
            autosave_session(client, user_id)

        except Exception as exc:
            LOGGER.error("답변 생성 중 오류: %s", exc)
            friendly_message = (
                "답변 생성 중 오류가 발생했습니다. "
                "잠시 후 다시 시도해 주세요."
            )
            st.session_state.chat_history.append(
                {"role": "assistant", "content": friendly_message}
            )
            update_conversation_memory(
                user_query, friendly_message, st.session_state.conversation_memory
            )
            placeholder.error(friendly_message)


def main() -> None:
    st.set_page_config(
        page_title="ENA RAG 챗봇",
        page_icon="📚",
        layout="wide",
    )

    inject_custom_css()
    init_session_state()

    keys = load_api_keys()
    key_msg = missing_keys_message(keys)
    client = (
        get_supabase_client(keys["SUPABASE_URL"], keys["SUPABASE_ANON_KEY"])
        if not key_msg
        else None
    )

    if not st.session_state.logged_in:
        render_auth_page(client, key_msg)
        return

    render_header()
    if key_msg:
        st.warning(key_msg)

    rag_option = render_sidebar(client, keys["OPENAI_API_KEY"], key_msg)
    render_chat_history()

    user_query = st.chat_input("메시지를 입력하세요...")
    if user_query:
        handle_user_input(
            user_query,
            rag_option,
            client,
            keys["OPENAI_API_KEY"],
            key_msg,
        )


if __name__ == "__main__":
    main()
