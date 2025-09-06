# main_webhook.py — aiogram v3 webhook server for Deta Space
import os
import logging
from aiohttp import web
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from dotenv import load_dotenv

# Import your configured Dispatcher with handlers from main.py
# IMPORTANT: make sure main.py starts polling only under if __name__ == "__main__"
from main import dp  # dp: Dispatcher

load_dotenv()
BOT_TOKEN = (os.getenv("BOT_TOKEN") or "").strip()
if not BOT_TOKEN:
    raise SystemExit("BOT_TOKEN is required in env")

# Public base URL for webhook (add it in Deta Space Environment later)
BASE_URL = (os.getenv("WEBHOOK_BASE_URL") or "").rstrip("/")

WEBHOOK_PATH = f"/webhook/{BOT_TOKEN}"
WEBHOOK_URL = f"{BASE_URL}{WEBHOOK_PATH}" if BASE_URL else None

bot = Bot(BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

async def on_startup(app: web.Application):
    if WEBHOOK_URL:
        await bot.set_webhook(WEBHOOK_URL, drop_pending_updates=True)
        logging.getLogger("boot").info(f"Webhook set: {WEBHOOK_URL}")
    else:
        logging.getLogger("boot").warning("WEBHOOK_BASE_URL not set yet; set it and redeploy.")

async def on_shutdown(app: web.Application):
    try:
        await bot.delete_webhook(drop_pending_updates=False)
    except Exception:
        pass
    await bot.session.close()

def create_app() -> web.Application:
    app = web.Application()
    SimpleRequestHandler(dispatcher=dp, bot=bot).register(app, path=WEBHOOK_PATH or "/webhook")
    setup_application(app, dp, bot=bot)
    app.add_routes([web.get("/", lambda _: web.Response(text="ok"))])
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    return app

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    app = create_app()
    # Deta Space expects 8080
    port = int(os.getenv("PORT", "8080"))
    web.run_app(app, host="0.0.0.0", port=port)
