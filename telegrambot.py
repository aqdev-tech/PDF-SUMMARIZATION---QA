import os
import asyncio
import logging
import shelve
from typing import Dict, Any
from io import BytesIO
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    CallbackQueryHandler, ContextTypes, filters
)
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from core import OpenRouterLLM, PDFProcessor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Bot Configuration
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Session cache
SESSION_CACHE = "user_sessions.db"

# Initialize PDFProcessor
pdf_processor = PDFProcessor()

def get_session(user_id: int) -> Dict[str, Any]:
    with shelve.open(SESSION_CACHE) as db:
        return db.get(str(user_id), {"status": "new"})

def save_session(user_id: int, session_data: Dict[str, Any]):
    with shelve.open(SESSION_CACHE) as db:
        db[str(user_id)] = session_data

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    save_session(user_id, {"status": "waiting_for_pdf"})
    welcome_text = """
🤖 **Welcome to PDF Q&A Bot!**

I can help you analyze PDF documents by:
📄 **Reading PDFs** - Upload any PDF document
❓ **Answering Questions** - Ask me anything about the content
📋 **Creating Summaries** - Get formatted summaries

**How to use:**
1. Send me a PDF file (max 20MB)
2. Wait for processing confirmation
3. Ask questions or request summaries

Ready? Send me a PDF to get started! 📎
    """
    await update.message.reply_text(welcome_text, parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = """
🆘 **Help - How to Use PDF Q&A Bot**

**Commands:**
• `/start` - Start the bot
• `/help` - Show this help message
• `/status` - Check current session status
• `/clear` - Clear current PDF session

**Features:**
📤 **Upload PDF:** Send any PDF file (up to 20MB)
❓ **Ask Questions:** Type any question about your PDF
📋 **Get Summary:** Use summary buttons or type "summarize"

**Examples:**
• "What is this document about?"
• "Summarize the main points"
• "What are the key findings?"
• "Explain the methodology"

**Tips:**
✅ Ensure your PDF has readable text
✅ Wait for "✅ Ready!" before asking questions
✅ Ask specific questions for better answers
✅ Use /clear to upload a new PDF

Need more help? Just ask! 🤝
    """
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    session = get_session(user_id)
    status = session.get("status", "unknown")

    if status == "new" or status == "waiting_for_pdf":
        status_text = "⏳ Waiting for PDF upload. Please send a PDF file."
    elif status == "processing":
        status_text = "🔄 Processing your PDF. Please wait..."
    elif status == "ready":
        pdf_name = session.get("pdf_name", "Unknown")
        char_count = session.get("char_count", 0)
        chunk_count = session.get("chunk_count", 0)
        status_text = f"""
✅ **Session Ready!**

📄 **PDF:** {pdf_name}
📊 **Characters:** {char_count:,}
📦 **Chunks:** {chunk_count}

You can now ask questions or request summaries!
        """
    else:
        status_text = "❓ Unknown status. Use /start to restart."
    
    await update.message.reply_text(status_text, parse_mode='Markdown')

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    save_session(user_id, {"status": "waiting_for_pdf"})
    await update.message.reply_text("🗑️ Session cleared! Send me a new PDF to analyze.", parse_mode='Markdown')

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    document = update.message.document

    if document.file_size > 20 * 1024 * 1024:
        await update.message.reply_text("❌ File too large! Please send a PDF smaller than 20MB.")
        return

    save_session(user_id, {"status": "processing", "pdf_name": document.file_name})
    processing_msg = await update.message.reply_text("🔄 **Processing your PDF...**\n\n⏳ Downloading file...", parse_mode='Markdown')

    try:
        file = await context.bot.get_file(document.file_id)
        pdf_bytes = await file.download_as_bytearray()
        
        await processing_msg.edit_text("🔄 **Processing your PDF...**\n\n✅ Downloaded\n⏳ Extracting text...", parse_mode='Markdown')
        text = pdf_processor.extract_text_from_pdf(bytes(pdf_bytes))

        if not text.strip():
            await processing_msg.edit_text("❌ **Error:** Could not extract text from PDF.\nMake sure your PDF contains readable text (not just images).")
            save_session(user_id, {"status": "waiting_for_pdf"})
            return

        await processing_msg.edit_text("🔄 **Processing your PDF...**\n\n✅ Downloaded\n✅ Text extracted\n⏳ Creating chunks...", parse_mode='Markdown')
        chunks = pdf_processor.split_text(text)

        await processing_msg.edit_text("🔄 **Processing your PDF...**\n\n✅ Downloaded\n✅ Text extracted\n✅ Chunks created\n⏳ Building vector database...", parse_mode='Markdown')
        vectorstore = pdf_processor.create_vector_store(chunks)

        if not vectorstore:
            await processing_msg.edit_text("❌ **Error:** Could not create vector database. Please try again.")
            save_session(user_id, {"status": "waiting_for_pdf"})
            return

        llm = OpenRouterLLM(api_key=OPENROUTER_API_KEY)
        
        session_data = {
            "status": "ready",
            "pdf_name": document.file_name,
            "char_count": len(text),
            "chunk_count": len(chunks),
            "full_text": text,
            "vectorstore": vectorstore,
            "llm": llm
        }
        save_session(user_id, session_data)

        keyboard = [
            [InlineKeyboardButton("❓ Ask Question", callback_data="ask_question")],
            [InlineKeyboardButton("📋 Formal Summary", callback_data="summary_formal"), InlineKeyboardButton("😊 Casual Summary", callback_data="summary_casual")],
            [InlineKeyboardButton("📌 Bullet Points", callback_data="summary_bullet")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        success_text = f"""
✅ **PDF Ready for Analysis!**

📄 **File:** {document.file_name}
📊 **Characters:** {len(text):,}
📦 **Chunks:** {len(chunks)}

**What would you like to do?**
        """
        await processing_msg.edit_text(success_text, parse_mode='Markdown', reply_markup=reply_markup)

    except Exception as e:
        logger.error(f"Document processing error: {str(e)}")
        await processing_msg.edit_text(f"❌ **Error processing PDF:** {str(e)}\n\nPlease try again with a different file.")
        save_session(user_id, {"status": "waiting_for_pdf"})

async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    session = get_session(user_id)

    if session.get("status") != "ready":
        await update.message.reply_text("❌ Please upload a PDF first using /start")
        return

    question = update.message.text.strip()

    if any(word in question.lower() for word in ['summary', 'summarize', 'sum up']):
        await handle_summary_request(update, context, "formal")
        return

    thinking_msg = await update.message.reply_text(f"🤔 **Question:** {question}\n\n⏳ Searching for answer...")

    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=session['llm'],
            chain_type="stuff",
            retriever=session['vectorstore'].as_retriever(search_kwargs={"k": 3})
        )
        answer = qa_chain.run(question)
        response_text = f"""
❓ **Question:** {question}

💡 **Answer:**
{answer}

---
💬 Ask another question or use /clear for a new PDF
        """
        await thinking_msg.edit_text(response_text, parse_mode='Markdown')

    except Exception as e:
        logger.error(f"Question handling error: {str(e)}")
        await thinking_msg.edit_text(f"❌ Error getting answer: {str(e)}\n\nPlease try rephrasing your question.")

async def handle_summary_request(update: Update, context: ContextTypes.DEFAULT_TYPE, tone: str) -> None:
    user_id = update.effective_user.id
    session = get_session(user_id)

    if session.get("status") != "ready":
        if update.callback_query:
            await update.callback_query.answer("Please upload a PDF first!")
        return

    tone_prompts = {
        "formal": "Please provide a formal, professional summary of this document:",
        "casual": "Give me a friendly, easy-to-understand summary of this document:",
        "bullet": "Summarize this document using clear, concise bullet points:"
    }
    tone_emojis = {"formal": "🎩", "casual": "😊", "bullet": "📌"}

    if update.callback_query:
        await update.callback_query.answer()
        thinking_msg = await update.callback_query.edit_message_text(f"{tone_emojis[tone]} **Generating {tone} summary...**\n\n⏳ Please wait...")
    else:
        thinking_msg = await update.message.reply_text(f"{tone_emojis[tone]} **Generating {tone} summary...**\n\n⏳ Please wait...")

    try:
        prompt = f"{tone_prompts[tone]}\n\n{session['full_text'][:4000]}..."
        summary = session['llm'](prompt)
        response_text = f"""
{tone_emojis[tone]} **{tone.title()} Summary:**

{summary}

---
💬 Ask a question or request another summary type
        """
        await thinking_msg.edit_text(response_text, parse_mode='Markdown')

    except Exception as e:
        logger.error(f"Summary generation error: {str(e)}")
        await thinking_msg.edit_text(f"❌ Error generating summary: {str(e)}\n\nPlease try again.")

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    data = query.data

    if data == "ask_question":
        await query.answer()
        await query.edit_message_text("❓ **Ask me anything about your PDF!**\n\nJust type your question in the chat.", parse_mode='Markdown')
    elif data.startswith("summary_"):
        tone = data.replace("summary_", "")
        await handle_summary_request(update, context, tone)

async def handle_unknown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    session = get_session(user_id)

    if session.get("status") == "waiting_for_pdf":
        await update.message.reply_text("📎 Please send me a PDF file to analyze.\n\nUse /help if you need assistance!")
    elif session.get("status") == "ready":
        await handle_question(update, context)
    else:
        await update.message.reply_text("❓ I didn't understand that. Use /help for available commands!")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Exception while handling an update: {context.error}")

def main():
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN not provided!")
        return
    
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY not found in environment variables!")
        return

    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(MessageHandler(filters.Document.PDF, handle_document))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_unknown))
    application.add_error_handler(error_handler)
    
    logger.info("Starting PDF Q&A Telegram Bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
