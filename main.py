import asyncio
import os
import sqlite3
import logging
from typing import Optional, Tuple, List

import pandas as pd
from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# OpenAI
from openai import OpenAI

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("boot")

# ---------------- Env ----------------
load_dotenv()
BOT_TOKEN = (os.getenv("BOT_TOKEN") or "").strip()
SUPPORT_CHAT_ID = int((os.getenv("SUPPORT_CHAT_ID") or "0").strip() or "0")
SIMILARITY_THRESHOLD = float((os.getenv("SIMILARITY_THRESHOLD") or "0.4").strip() or "0.4")
USE_LLM_FALLBACK = (os.getenv("USE_LLM_FALLBACK","true").lower() in ["1","true","yes"])
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
OPENAI_MODEL = (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()

if not BOT_TOKEN:
    raise SystemExit("Please set BOT_TOKEN in .env")

# ---------------- KB loader ----------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "knowledge_base.csv")

kb_df: pd.DataFrame
vectorizer: TfidfVectorizer
kb_matrix = None

def load_kb():
    global kb_df, vectorizer, kb_matrix
    if not os.path.exists(DATA_PATH):
        raise SystemExit(f"Knowledge base not found at {DATA_PATH}")
    kb_df = pd.read_csv(DATA_PATH)
    if not set(["question", "answer", "url"]).issubset(kb_df.columns):
        raise SystemExit("CSV must contain columns: question, answer, url")
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5))
    kb_matrix = vectorizer.fit_transform(kb_df["question"].fillna("").astype(str))
    log.info(f"KB loaded: {len(kb_df)} rows")

def top_k_context(user_text: str, k: int = 5) -> List[dict]:
    q = vectorizer.transform([user_text])
    sims = cosine_similarity(q, kb_matrix)[0]
    idxs = sims.argsort()[::-1][:k]
    rows = []
    for i in idxs:
        row = kb_df.iloc[int(i)]
        rows.append({"question": str(row["question"]), "answer": str(row["answer"]), "url": str(row["url"])})
    return rows

def answer_from_kb(user_text: str) -> Tuple[Optional[str], float]:
    if not user_text or not user_text.strip():
        return None, 0.0
    q = vectorizer.transform([user_text])
    sims = cosine_similarity(q, kb_matrix)[0]
    best_idx = sims.argmax()
    best_score = float(sims[best_idx])
    if best_score >= SIMILARITY_THRESHOLD:
        row = kb_df.iloc[best_idx]
        url = str(row.get("url") or "").strip()
        ans = str(row.get("answer") or "").strip()
        if url:
            ans = f"{ans}\n\nПодробнее: {url}"
        return ans, best_score
    return None, best_score

load_kb()

# ---------------- OpenAI LLM fallback ----------------
client = None
if USE_LLM_FALLBACK and OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
    log.info(f"LLM fallback enabled with model={OPENAI_MODEL}")
else:
    if USE_LLM_FALLBACK:
        log.warning("LLM fallback requested but OPENAI_API_KEY not set. Fallback disabled.")
    USE_LLM_FALLBACK = False

def llm_answer(user_text: str) -> Optional[str]:
    if not client:
        log.warning("LLM client is not initialized")
        return None
    ctx_items = top_k_context(user_text, k=5)
    ctx_str = "\n\n".join([f"Q: {it['question']}\nA: {it['answer']}" for it in ctx_items])
    sys_prompt = (
        "Ты ассистент поддержки сайта https://meff.netlify.app/. Отвечай кратко и по делу. "
        "Отвечай строго на основе предоставленного контекста (FAQ/услуги компании/вопросы по сайту). "
        "Если точной информации нет — скажи, что не уверен, и предложи связаться с оператором."
    )
    user_prompt = (
        f"Вопрос пользователя: {user_text}\n\n"
        f"Контекст:\n{ctx_str}\n\n"
        "Если уместно, добавь в конце строку: 'Подробнее: https://meff.netlify.app/'"
    )
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        # Покажем в логах полную ошибку — сразу видно 401/проект/лимит
        import traceback
        log.error(f"LLM error: {e}\n{traceback.format_exc()}")
        return None

# ---------------- DB (tickets) ----------------
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "bot.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("""CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    support_msg_id INTEGER,
    status TEXT DEFAULT 'open'
)""")
conn.commit()

def create_ticket(user_id: int, support_msg_id: Optional[int]) -> int:
    cur.execute("INSERT INTO tickets (user_id, support_msg_id, status) VALUES (?, ?, 'open')",
                (user_id, support_msg_id))
    conn.commit()
    return cur.lastrowid

def get_user_id_by_support_msg(support_msg_id: int) -> Optional[int]:
    cur.execute("SELECT user_id FROM tickets WHERE support_msg_id = ?", (support_msg_id,))
    row = cur.fetchone()
    return int(row[0]) if row else None

# ---------------- FSM ----------------
class SupportForm(StatesGroup):
    waiting_for_description = State()

# ---------------- Bot & Router ----------------
bot = Bot(BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()
router = Router()
dp.include_router(router)

# ---------------- Keyboards ----------------
def main_menu_kb():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Задать вопрос", callback_data="ask")],
        [InlineKeyboardButton(text="Написать оператору", callback_data="support")]
    ])

def escalate_kb():
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="Отправить оператору", callback_data="esc_yes"),
            InlineKeyboardButton(text="Не надо", callback_data="esc_no")
        ]
    ])

# ---------------- Commands ----------------
@router.message(CommandStart())
async def on_start(m: Message):
    await m.answer(
        "Привет! Я бот поддержки компании. Отвечаю на частые вопросы 24/7 "
        "и при необходимости передам запрос живому оператору.",
        reply_markup=main_menu_kb()
    )

@router.message(Command("help"))
async def on_help(m: Message):
    await m.answer(
        "Отправьте свой вопрос текстом — я проверю базу знаний.\n"
        "Если уверенности мало, предложу эскалацию оператору.\n\n"
        "Команды:\n"
        "• /start — главное меню\n"
        "• /support — написать оператору\n"
        "• /id — показать chat_id (для группы)\n"
        "• /reload — перечитать базу знаний\n"
        "• /mode — показать режимы ответа"
    )

@router.message(Command("reload"))
async def cmd_reload(m: Message):
    try:
        load_kb()
        await m.answer(f"KB reloaded: {len(kb_df)} entries.")
    except Exception as e:
        await m.answer(f"Reload error: {e}")

@router.message(Command("llm"))
async def cmd_llm(m: Message):
    if not USE_LLM_FALLBACK:
        await m.answer("LLM fallback выключен (USE_LLM_FALLBACK=false).")
        return
    if not OPENAI_API_KEY:
        await m.answer("OPENAI_API_KEY пуст — добавь ключ в .env.")
        return
    q = (m.text or "").split(maxsplit=1)
    if len(q) < 2:
        await m.answer("Напиши: <code>/llm твой вопрос</code>")
        return
    test_q = q[1]
    ans = llm_answer(test_q)
    if ans:
        await m.answer("Ответ LLM:\n\n" + ans)
    else:
        await m.answer("LLM не ответил. Смотри логи консоли — там будет причина (401/429/проект и т.п.).")

@router.message(Command("mode"))
async def cmd_mode(m: Message):
    await m.answer(
        f"Similarity threshold: {SIMILARITY_THRESHOLD}\n"
        f"LLM fallback: {'ON' if USE_LLM_FALLBACK else 'OFF'} ({OPENAI_MODEL if USE_LLM_FALLBACK else ''})"
    )

@router.message(Command("support"))
async def support_cmd(m: Message, state: FSMContext):
    await state.set_state(SupportForm.waiting_for_description)
    await m.answer("Опишите, пожалуйста, вашу проблему одним сообщением — я передам оператору.")

@router.message(Command("id"))
async def cmd_id(m: Message):
    await m.answer(f"chat_id = <code>{m.chat.id}</code>\nchat_type = {m.chat.type}")
    log.info(f"/id -> chat_id={m.chat.id}, type={m.chat.type}, title={getattr(m.chat, 'title', '')}")

# ---------------- Callbacks ----------------
@router.callback_query(F.data == "ask")
async def ask_cb(cq: CallbackQuery):
    await cq.message.answer("Напишите ваш вопрос текстом. Я проверю базу знаний.")
    await cq.answer()

@router.callback_query(F.data == "support")
async def support_cb(cq: CallbackQuery, state: FSMContext):
    await state.set_state(SupportForm.waiting_for_description)
    await cq.message.answer("Опишите, пожалуйста, вашу проблему одним сообщением — я передам оператору.")
    await cq.answer()

@router.callback_query(F.data == "esc_yes")
async def esc_yes_cb(cq: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    original_q = data.get("last_question")
    await state.set_state(SupportForm.waiting_for_description)
    if original_q:
        await state.update_data(original_question=original_q)
    await cq.message.answer("Окей, напишите кратко детали (1 сообщением) — передам оператору вместе с вашим вопросом.")
    await cq.answer()

@router.callback_query(F.data == "esc_no")
async def esc_no_cb(cq: CallbackQuery):
    await cq.message.answer("Хорошо! Если что, всегда можно набрать /support.")
    await cq.answer()

# ---------------- User text ----------------
@router.message(SupportForm.waiting_for_description)
async def handle_support_description(m: Message, state: FSMContext):
    details = m.text or ""
    data = await state.get_data()
    original_q = data.get("original_question")
    user = m.from_user
    title = f"Новый тикет от @{user.username or 'без_юзернейма'} (ID {user.id})"
    block = f"<b>{title}</b>\n"
    if original_q:
        block += f"\nИсходный вопрос:\n<blockquote>{original_q}</blockquote>\n"
    block += f"Описание:\n<blockquote>{details}</blockquote>"
    if SUPPORT_CHAT_ID == 0:
        await m.answer("Ошибка: не задан SUPPORT_CHAT_ID в .env")
        await state.clear()
        return
    sent = await bot.send_message(chat_id=SUPPORT_CHAT_ID, text=block)
    t_id = create_ticket(user_id=user.id, support_msg_id=sent.message_id)
    await m.answer(f"Спасибо! Создан тикет №{t_id}. Оператор скоро ответит здесь.")
    await state.clear()

@router.message(F.chat.type == "private")
async def handle_user_question(m: Message, state: FSMContext):
    q = (m.text or "").strip()
    if not q:
        return

    # 1. Сохраняем последний вопрос пользователя
    await state.update_data(last_question=q)

    # 2. Пытаемся ответить из базы знаний
    ans, score = answer_from_kb(q)
    if ans and score >= SIMILARITY_THRESHOLD:
        await m.answer(ans)
        return

    # 3. Если база не уверена — спрашиваем LLM
    if USE_LLM_FALLBACK:
        gen = llm_answer(q)
        if gen:
            await m.answer(gen)
            return

    # 4. Если и LLM не ответил — только тогда оператор
    await m.answer(
        "Я не уверен в ответе 🤔. Хочешь, передам вопрос оператору?",
        reply_markup=escalate_kb()
    )

# ---------------- Support group relay + debug ----------------
@router.message()
async def support_group_relay(m: Message):
    if SUPPORT_CHAT_ID and (m.chat and m.chat.id == SUPPORT_CHAT_ID) and m.reply_to_message:
        ref_id = m.reply_to_message.message_id
        user_id = get_user_id_by_support_msg(ref_id)
        if user_id:
            text = m.text or "(без текста)"
            try:
                await bot.send_message(chat_id=user_id, text=f"Ответ оператора:\n\n{text}")
            except Exception:
                pass
    if m.chat and m.chat.type != "private":
        logging.getLogger("group").info(
            f"[DEBUG] chat_id={m.chat.id}, type={m.chat.type}, title={getattr(m.chat, 'title', '')}"
        )

# ---------------- Run ----------------
async def main():
    log.info("Bot is starting...")
    if SUPPORT_CHAT_ID == 0:
        log.warning("WARNING: SUPPORT_CHAT_ID is 0. Set it in .env after /id in the group.")
    me = await bot.get_me()
    log.info(f"getMe OK: @{me.username} (id={me.id})")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        log.exception(f"FATAL: {e}")
        raise
