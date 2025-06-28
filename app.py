import streamlit as st
import os
import asyncio
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from core import OpenRouterLLM, PDFProcessor

# Load environment variables
load_dotenv()

# Initialize PDFProcessor
pdf_processor = PDFProcessor()

def get_tone_prompt(tone: str) -> str:
    """Get the appropriate prompt prefix based on selected tone"""
    tone_prompts = {
        "formal": "Please summarize this document in a formal, professional tone:",
        "casual": "Give me a relaxed, friendly summary of this document:",
        "bullet": "Summarize this document using clear, concise bullet points:"
    }
    return tone_prompts.get(tone, tone_prompts["formal"])

async def process_pdf(uploaded_file):
    """Asynchronously process the uploaded PDF file."""
    loop = asyncio.get_event_loop()
    
    with st.spinner("🔄 Extracting text from PDF..."):
        pdf_bytes = uploaded_file.read()
        text = await loop.run_in_executor(None, pdf_processor.extract_text_from_pdf, pdf_bytes)
    
    if not text:
        st.error("❌ Could not extract text from PDF")
        return

    st.info(f"📝 Extracted {len(text)} characters from PDF")

    with st.spinner("✂️ Processing text chunks..."):
        chunks = await loop.run_in_executor(None, pdf_processor.split_text, text)
    
    st.info(f"📊 Created {len(chunks)} text chunks")

    with st.spinner("🧠 Building vector database..."):
        vectorstore = await loop.run_in_executor(None, pdf_processor.create_vector_store, chunks)

    if vectorstore:
        st.success("✅ Vector database ready!")
        st.session_state['vectorstore'] = vectorstore
        st.session_state['full_text'] = text
    else:
        st.error("❌ Failed to create vector database")

async def main():
    st.set_page_config(
        page_title="PDF Q&A & Summarization Tool",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 PDF Q&A & Summarization Tool")
    st.markdown("Upload a PDF, ask questions, and get AI-powered summaries!")
    
    with st.sidebar:
        st.header("🔧 Configuration")
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
        
        if not openrouter_api_key:
            st.warning("⚠️ OpenRouter API key not found. Please set it in the .env file.")
            return
        
        st.success("✅ API key configured in backend")
        
        with st.expander("ℹ️ Optional Settings"):
            st.info("The app is configured with:\n- Model: llama-3.3-8b-instruct\n- Referer: GitHub project\n- Title: PDF Q&A Tool")

    try:
        llm = OpenRouterLLM(api_key=openrouter_api_key)
        st.session_state['llm'] = llm
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to analyze"
        )
        
        if uploaded_file and 'vectorstore' not in st.session_state:
            await process_pdf(uploaded_file)

    with col2:
        st.header("💬 Q&A & Summary")
        
        if 'vectorstore' in st.session_state:
            st.subheader("❓ Ask Questions")
            question = st.text_input(
                "Enter your question about the document:",
                placeholder="What is the main topic of this document?"
            )
            
            if st.button("🔍 Get Answer", type="primary"):
                if question:
                    with st.spinner("🤔 Thinking..."):
                        try:
                            qa_chain = RetrievalQA.from_chain_type(
                                llm=st.session_state['llm'],
                                chain_type="stuff",
                                retriever=st.session_state['vectorstore'].as_retriever(
                                    search_kwargs={"k": 3}
                                )
                            )
                            answer = qa_chain.run(question)
                            st.success("✅ Answer:")
                            st.markdown(f"**Q:** {question}")
                            st.markdown(f"**A:** {answer}")
                        except Exception as e:
                            st.error(f"Error getting answer: {str(e)}")
                else:
                    st.warning("Please enter a question")
            
            st.divider()
            
            st.subheader("📋 Document Summary")
            
            tone = st.selectbox(
                "Choose summary tone:",
                options=["formal", "casual", "bullet"],
                format_func=lambda x: {
                    "formal": "🎩 Formal & Professional",
                    "casual": "😊 Casual & Friendly", 
                    "bullet": "📌 Bullet Points"
                }[x]
            )
            
            if st.button("📝 Generate Summary", type="secondary"):
                with st.spinner("✍️ Generating summary..."):
                    try:
                        tone_prompt = get_tone_prompt(tone)
                        full_prompt = f"{tone_prompt}\n\n{st.session_state['full_text']}"
                        summary = st.session_state['llm'](full_prompt)
                        st.success("✅ Summary:")
                        st.markdown(summary)
                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")
        else:
            st.info("👆 Please upload a PDF file first to start asking questions and generating summaries")
    
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        Built with ❤️ using Streamlit, LangChain, FAISS, and OpenRouter
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    asyncio.run(main())