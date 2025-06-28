# PDF Q&A and Summarization Tool

This project provides a versatile tool for interacting with PDF documents, offering both a Streamlit web interface and a Telegram bot. You can upload PDF files, ask questions about their content, and generate summaries in various tones.

## Features

- **PDF Text Extraction**: Extracts text content from uploaded PDF documents.
- **Intelligent Q&A**: Ask natural language questions about your PDF and get AI-powered answers.
- **Customizable Summarization**: Generate summaries in formal, casual, or bullet-point formats.
- **Streamlit Web Interface**: An intuitive and easy-to-use web application for direct interaction.
- **Telegram Bot Integration**: Interact with your PDFs on the go via a Telegram bot.
- **Modular Design**: Shared core logic for better maintainability and reduced code duplication.
- **Persistent Sessions (Telegram Bot)**: User sessions in the Telegram bot are now persistent across restarts using a file-based cache.

## Technologies Used

- **Streamlit**: For creating the interactive web application.
- **Python-Telegram-Bot**: For building the Telegram bot.
- **LangChain**: For orchestrating the LLM and vector store interactions.
- **PyMuPDF (fitz)**: For efficient PDF text extraction.
- **HuggingFace Embeddings**: For generating text embeddings.
- **FAISS**: For efficient similarity search and vector storage.
- **OpenRouter API**: For accessing various large language models.
- **python-dotenv**: For managing environment variables securely.
- **shelve**: For file-based session caching in the Telegram bot.

## Setup and Installation

Follow these steps to set up and run the project locally:

### 1. Clone the Repository

```bash
git clone https://github.com/abdul183/PDF-SUMMARIZATION---QA.git
cd PDF-SUMMARIZATION---QA
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory of the project based on the `.env.example` file:

```bash
cp .env.example .env
```

Open the newly created `.env` file and add your API keys:

- **`OPENROUTER_API_KEY`**: Get your API key from [OpenRouter](https://openrouter.ai/)
- **`TELEGRAM_TOKEN`**: Get your bot token from BotFather on Telegram.

Your `.env` file should look like this:

```
OPENROUTER_API_KEY="your_openrouter_api_key_here"
TELEGRAM_TOKEN="your_telegram_bot_token_here"
```

### 5. Run the Applications

You can run the Streamlit web app and the Telegram bot independently.

#### Run the Streamlit Web App

```bash
streamlit run app.py
```

This will open the Streamlit application in your web browser.

#### Run the Telegram Bot

```bash
python telegrambot.py
```

The Telegram bot will start polling for updates. You can then interact with your bot on Telegram.

## Usage

### Streamlit Web App

1.  **Upload PDF**: Use the file uploader to select a PDF document.
2.  **Wait for Processing**: The app will extract text and build a vector database.
3.  **Ask Questions**: Type your questions in the input box and click "Get Answer".
4.  **Generate Summary**: Choose a summary tone (Formal, Casual, Bullet Points) and click "Generate Summary".

### Telegram Bot

1.  **Start the Bot**: Send `/start` to your bot on Telegram.
2.  **Upload PDF**: Send a PDF file directly to the bot.
3.  **Wait for Processing**: The bot will confirm once the PDF is processed.
4.  **Ask Questions**: Type your questions directly in the chat.
5.  **Generate Summary**: Use the inline keyboard buttons (Formal Summary, Casual Summary, Bullet Points) or type commands like "summarize" to get a summary.
6.  **Manage Sessions**: Use `/status` to check the current PDF session and `/clear` to clear the current session and upload a new PDF.

## Project Structure

```
PDF-SUMMARIZATION---QA/
├── .env                 # Environment variables (ignored by Git)
├── .env.example         # Example environment variables file
├── .gitignore           # Specifies intentionally untracked files to ignore
├── app.py               # Streamlit web application
├── core.py              # Shared core logic (LLM, PDF processing, embeddings)
├── README.md            # Project README file
├── requirements.txt     # Python dependencies
└── telegrambot.py       # Telegram bot application
```

## Contributing

Feel free to fork the repository, open issues, or submit pull requests to improve the project.

## License

This project is open-source and available under the MIT License. (You might want to add a LICENSE file if you haven't already.)
