import streamlit as st
import os
import asyncio
from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
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

async def process_pdfs(uploaded_files):
    """Asynchronously process the uploaded PDF files."""
    loop = asyncio.get_event_loop()
    
    all_texts = {}
    for uploaded_file in uploaded_files:
        with st.spinner(f"🔄 Extracting text from {uploaded_file.name}..."):
            pdf_bytes = uploaded_file.read()
            text = await loop.run_in_executor(None, pdf_processor.extract_text_from_pdf, pdf_bytes)
        
        if not text:
            st.error(f"❌ Could not extract text from {uploaded_file.name}")
            continue

        st.info(f"📝 Extracted {len(text)} characters from {uploaded_file.name}")
        all_texts[uploaded_file.name] = text

    if not all_texts:
        st.error("❌ No text could be extracted from the uploaded PDFs.")
        return

    with st.spinner("✂️ Processing text chunks..."):
        # We need to pass metadata to the splitter
        documents = []
        for filename, text in all_texts.items():
            docs = pdf_processor.split_text_with_metadata(text, {"source": filename})
            documents.extend(docs)
    
    st.info(f"📊 Created {len(documents)} text chunks from {len(all_texts)} PDF(s)")

    with st.spinner("🧠 Building vector database..."):
        vectorstore = await loop.run_in_executor(None, pdf_processor.create_vector_store_with_metadata, documents)

    if vectorstore:
        st.success("✅ Vector database ready!")
        st.session_state['vectorstore'] = vectorstore
        st.session_state['all_texts'] = all_texts # Store all texts for summarization
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
        st.header("📤 Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            help="Upload one or more PDF documents to analyze",
            accept_multiple_files=True
        )
        
        if uploaded_files and 'vectorstore' not in st.session_state:
            await process_pdfs(uploaded_files)

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
                                ),
                                return_source_documents=True
                            )
                            result = qa_chain(question)
                            answer = result['result']
                            source_docs = result['source_documents']
                            
                            st.success("✅ Answer:")
                            st.markdown(f"**Q:** {question}")
                            st.markdown(f"**A:** {answer}")
                            
                            if source_docs:
                                st.markdown("**Sources:**")
                                for doc in source_docs:
                                    st.markdown(f"- *{doc.metadata['source']}*")

                        except Exception as e:
                            st.error(f"Error getting answer: {str(e)}")
                else:
                    st.warning("Please enter a question")
            
            st.divider()
            
            st.subheader("📋 Document Summary")

            # Add a selection for which document to summarize
            if len(st.session_state['all_texts']) > 1:
                doc_to_summarize = st.selectbox(
                    "Choose a document to summarize:",
                    options=list(st.session_state['all_texts'].keys())
                )
            else:
                doc_to_summarize = list(st.session_state['all_texts'].keys())[0]
            
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
                        text_to_summarize = st.session_state['all_texts'][doc_to_summarize]
                        tone_prompt = get_tone_prompt(tone)
                        full_prompt = f"{tone_prompt}\n\n{text_to_summarize}"
                        summary = st.session_state['llm'](full_prompt)
                        st.success(f"✅ Summary for {doc_to_summarize}:")
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