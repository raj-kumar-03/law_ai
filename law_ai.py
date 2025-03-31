import streamlit as st
from langchain_community.document_loaders import PDFMinerLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain, RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import fitz
import base64
import datetime
import tempfile
import google.generativeai as genai
import os
import warnings

# ✅ Ignore DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

st.set_page_config(page_title="Legal Document Summarizer", layout="wide")

# ✅ Retrieve API key from Streamlit Secrets
api_key = st.secrets.get("GOOGLE_API_KEY")


if not api_key:
    raise ValueError("Missing GOOGLE_API_KEY in Streamlit Secrets")
#If Running in local Machine
#load_dotenv() 
#api_key = os.getenv("GOOGLE_API_KEY")
#if not api_key:
  #  raise ValueError("Missing GOOGLE_API_KEY environment variable")

# ✅ Configure Google Generative AI
genai.configure(api_key=api_key)


class LegalSummarizer:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                            temperature=0.3,
                                            convert_system_message_to_human=True)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=200,
            separators=["\n\nSECTION", "\n\nArticle", "\n\nCLAUSE"]
        )

    def process_document(self, uploaded_file):
        """Process different document formats (PDF, DOCX, TXT)"""
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # For TXT files, we'll handle them directly in memory
        if file_extension == 'txt':
            try:
                documents = self._process_txt_in_memory(uploaded_file)
                return self.text_splitter.create_documents(documents)
            except Exception as e:
                st.error(f"Error processing TXT file: {str(e)}")
                raise
        
        # For PDF and DOCX, we'll use temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name
        
        try:
            if file_extension == 'pdf':
                documents = self._process_pdf(temp_path)
            elif file_extension == 'docx':
                documents = self._process_docx(temp_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
            return self.text_splitter.create_documents(documents)
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _process_pdf(self, file_path):
        """PDF processing with PyMuPDF"""
        documents = []
        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc):
                text = page.get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES)
                documents.append(f"Page {page_num+1}:\n{text}")
        return documents

    def _process_docx(self, file_path):
        """DOCX processing with docx2txt"""
        loader = Docx2txtLoader(file_path)
        docs = loader.load()
        # Format the document similar to PDF pages for consistency
        documents = []
        for i, doc in enumerate(docs):
            documents.append(f"Section {i+1}:\n{doc.page_content}")
        return documents

    def _process_txt_in_memory(self, uploaded_file):
        """Process TXT files directly in memory without using TextLoader"""
        # Read the content of the uploaded file
        text_content = uploaded_file.getvalue().decode('utf-8', errors='replace')
        
        # Split text into manageable sections (either by paragraphs or fixed chunks)
        documents = []
        
        # Try to split by paragraphs first (double newlines)
        sections = text_content.split("\n\n")
        
        # If there are too few sections, use a more aggressive splitting approach
        if len(sections) < 3:
            # Split by single newlines
            sections = text_content.split("\n")
            
            # If still too few sections, use fixed-size chunks
            if len(sections) < 3:
                chunk_size = 2000
                sections = [text_content[i:i+chunk_size] for i in range(0, len(text_content), chunk_size)]
        
        # Create documents from the sections
        for i, section in enumerate(sections):
            if section.strip():  # Skip empty sections
                documents.append(f"Section {i+1}:\n{section}")
                
        return documents

    def create_summary_chain(self, template):
        """Generic summary chain with customizable template."""
        return LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(template),
            verbose=False,  # Changed to False for cleaner Streamlit output
        )

    def create_qa_system(self, docs):
        """Document-aware Q&A with citations"""
        vectorstore = FAISS.from_documents(docs, self.embeddings)
        qa_prompt = """Answer using ONLY the provided legal document context. Cite page numbers or sections. Do not repeat information from the FAQ.
        Context: {context}
        Question: {question}
        Answer:"""
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": PromptTemplate.from_template(qa_prompt)},
            return_source_documents=True
        )

    def generate_faq(self, docs):
        """Generate FAQ from the document."""
        qa_prompt = """Generate 5 frequently asked questions (FAQs) based on the legal document.
        Context: {context}
        FAQs:"""
        qa_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(qa_prompt),
            verbose=False,  # Changed to False for cleaner Streamlit output
        )
        context = "\n".join([doc.page_content for doc in docs])
        faq_text = qa_chain.run(context)
        return faq_text

    def generate_title(self, docs):
        """Generates title for the document"""
        title_template = """Generate a short title for the following legal document.
        Document: {text}
        Title:"""
        title_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(title_template),
            verbose=False,  # Changed to False for cleaner Streamlit output
        )
        context = "\n".join([doc.page_content for doc in docs])
        title = title_chain.run(context)
        return title


def get_download_link(text, filename, link_text):
    """Generate a download link for text content"""
    b64 = base64.b64encode(text.encode()).decode()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    download_filename = f"{filename}_{timestamp}.txt"
    href = f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{link_text}</a>'
    return href


def setup_ui():
    """Set up the Streamlit user interface"""
    st.title("Legal Document Summarizer")

    # Initialize session state variables
    if 'docs' not in st.session_state:
        st.session_state.docs = None
    if 'title' not in st.session_state:
        st.session_state.title = None
    if 'summarizer' not in st.session_state:
        st.session_state.summarizer = LegalSummarizer()
    if 'faq_text' not in st.session_state:
        st.session_state.faq_text = None
    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = None
    if 'current_summary' not in st.session_state:
        st.session_state.current_summary = None
    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = None
    if 'current_answer' not in st.session_state:
        st.session_state.current_answer = None
    if 'current_faq_answer' not in st.session_state:
        st.session_state.current_faq_answer = None
    if 'file_format' not in st.session_state:
        st.session_state.file_format = None


def document_upload_section():
    """File upload and processing section"""
    uploaded_file = st.file_uploader("Upload a legal document", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        st.session_state.file_format = file_extension.upper()
        
        if st.button("Process Document"):
            with st.spinner(f"Processing {file_extension.upper()} document..."):
                try:
                    st.session_state.docs = st.session_state.summarizer.process_document(uploaded_file)
                    st.session_state.title = st.session_state.summarizer.generate_title(st.session_state.docs)
                    st.session_state.faq_text = None  # Reset FAQ
                    st.session_state.qa_system = None  # Reset QA system
                    
                    # Reset all current content
                    st.session_state.current_summary = None
                    st.session_state.current_analysis = None
                    st.session_state.current_answer = None
                    st.session_state.current_faq_answer = None
                    
                    st.success(f"{file_extension.upper()} document processed successfully!")
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
                    st.info("Please try a different file or file format.")
                    return False
            
    return uploaded_file is not None and st.session_state.docs is not None


def summary_tab():
    """Summary tab content"""
    st.subheader("Summary Options")
    summary_type = st.radio(
        "Select summary type:",
        ["Brief paragraph summary", "Bullet point summary"]
    )
    
    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            if summary_type == "Brief paragraph summary":
                paragraph_template = f"""Title: {st.session_state.title}\nSummarize the following legal document into a brief and concise paragraph. Document: {{text}}\nSummary:"""
                summary_chain = st.session_state.summarizer.create_summary_chain(paragraph_template)
                summary = summary_chain.run(st.session_state.docs)
            else:  # Bullet point summary
                bullet_template = f"""Title: {st.session_state.title}\nSummarize the following legal document into 5-6 concise bullet points. Document: {{text}}\nSummary:"""
                summary_chain = st.session_state.summarizer.create_summary_chain(bullet_template)
                summary = summary_chain.run(st.session_state.docs)
            
            st.session_state.current_summary = summary
            
            st.markdown("### Summary")
            st.write(summary)
            
            # Download button for summary
            st.markdown(
                get_download_link(
                    summary,
                    f"{st.session_state.title.replace(' ', '_')}_summary",
                    "Download Summary"
                ),
                unsafe_allow_html=True
            )


def detailed_analysis_tab():
    """Detailed analysis tab content"""
    st.subheader("Detailed Analysis")
    detail_option = st.selectbox(
        "Select analysis type:",
        ["Key Obligations", "Rights & Responsibilities", "Legal Risks & Penalties", "Important Dates & Deadlines"]
    )
    
    if st.button("Generate Detailed Analysis"):
        with st.spinner("Generating detailed analysis..."):
            if detail_option == "Key Obligations":
                detail_template = f"""Title: {st.session_state.title}\nKey Obligations: List the major commitments and duties mentioned in the document with relevant clauses. Document: {{text}}"""
            elif detail_option == "Rights & Responsibilities":
                detail_template = f"""Title: {st.session_state.title}\nRights & Responsibilities: Outline the rights and responsibilities of each party involved with citations (Page X). Document: {{text}}"""
            elif detail_option == "Legal Risks & Penalties":
                detail_template = f"""Title: {st.session_state.title}\nLegal Risks & Penalties: Highlight possible penalties or risks for non-compliance with severity (1-5). Document: {{text}}"""
            else:  # Important Dates & Deadlines
                detail_template = f"""Title: {st.session_state.title}\nImportant Dates & Deadlines: Mention key deadlines, renewal dates, or legal timelines with dates and consequences. Document: {{text}}"""
            
            summary_chain = st.session_state.summarizer.create_summary_chain(detail_template)
            analysis = summary_chain.run(st.session_state.docs)
            
            st.session_state.current_analysis = analysis
            
            st.markdown(f"### {detail_option}")
            st.write(analysis)
            
            # Download button for detailed analysis
            st.markdown(
                get_download_link(
                    analysis,
                    f"{st.session_state.title.replace(' ', '_')}_{detail_option.replace(' ', '_').lower()}",
                    f"Download {detail_option} Analysis"
                ),
                unsafe_allow_html=True
            )


def qa_tab():
    """Q&A tab content"""
    st.subheader("Ask a Question")
    
    # Initialize QA system if not already done
    if st.session_state.qa_system is None:
        with st.spinner("Setting up Q&A system..."):
            st.session_state.qa_system = st.session_state.summarizer.create_qa_system(st.session_state.docs)
    
    # User question input
    user_question = st.text_input("Enter your question about the document:")
    
    if st.button("Get Answer") and user_question:
        with st.spinner("Finding answer..."):
            result = st.session_state.qa_system({"query": user_question})
            answer = result['result']
            
            # Format source information
            sources = "\n\nSources:\n"
            for i, doc in enumerate(result['source_documents'], 1):
                sources += f"Source {i}: {doc.page_content[:200]}...\n\n"
            
            full_answer = f"Question: {user_question}\n\nAnswer: {answer}{sources}"
            st.session_state.current_answer = full_answer
            
            st.markdown("### Answer")
            st.write(answer)
            
            st.markdown("### Sources")
            for doc in result['source_documents']:
                st.text(doc.page_content[:200] + "...")  # Show a preview
            
            # Download button for Q&A result
            st.markdown(
                get_download_link(
                    full_answer,
                    f"{st.session_state.title.replace(' ', '_')}_qa_response",
                    "Download Answer"
                ),
                unsafe_allow_html=True
            )


def faq_tab():
    """FAQ tab content"""
    st.subheader("Frequently Asked Questions")
    
    # Generate FAQ if not already done
    if st.session_state.faq_text is None:
        if st.button("Generate FAQ"):
            with st.spinner("Generating FAQ..."):
                st.session_state.faq_text = st.session_state.summarizer.generate_faq(st.session_state.docs)
    
    # Display FAQ
    if st.session_state.faq_text is not None:
        faq_lines = st.session_state.faq_text.split('\n')
        faq_lines = [line for line in faq_lines if line.strip()]  # Remove empty lines
        
        st.markdown("### FAQ")
        for line in faq_lines:
            st.write(line)
        
        # Download button for FAQ list
        st.markdown(
            get_download_link(
                st.session_state.faq_text,
                f"{st.session_state.title.replace(' ', '_')}_faq_list",
                "Download FAQ List"
            ),
            unsafe_allow_html=True
        )
        
        # Allow selecting a question
        faq_questions = [q.split('. ', 1)[1] if '. ' in q else q for q in faq_lines]
        selected_question = st.selectbox("Select a question to answer:", faq_questions)
        
        if st.button("Answer Selected Question"):
            with st.spinner("Finding answer..."):
                # Initialize QA system if not already done
                if st.session_state.qa_system is None:
                    st.session_state.qa_system = st.session_state.summarizer.create_qa_system(st.session_state.docs)
                
                result = st.session_state.qa_system({"query": selected_question})
                answer = result['result']
                
                # Format source information
                sources = "\n\nSources:\n"
                for i, doc in enumerate(result['source_documents'], 1):
                    sources += f"Source {i}: {doc.page_content[:200]}...\n\n"
                
                full_answer = f"Question: {selected_question}\n\nAnswer: {answer}{sources}"
                st.session_state.current_faq_answer = full_answer
                
                st.markdown("### Answer")
                st.write(answer)
                
                st.markdown("### Sources")
                for doc in result['source_documents']:
                    st.text(doc.page_content[:200] + "...")  # Show a preview
                
                # Download button for FAQ answer
                st.markdown(
                    get_download_link(
                        full_answer,
                        f"{st.session_state.title.replace(' ', '_')}_faq_answer",
                        "Download FAQ Answer"
                    ),
                    unsafe_allow_html=True
                )


def export_all_tab():
    """Export all generated content tab"""
    st.subheader("Export All Generated Content")
    
    # Check if we have any content to export
    has_content = (
        st.session_state.current_summary is not None or 
        st.session_state.current_analysis is not None or 
        st.session_state.current_answer is not None or 
        st.session_state.faq_text is not None or
        st.session_state.current_faq_answer is not None
    )
    
    if has_content:
        all_content = f"# Legal Document Summary Report\n\n"
        all_content += f"## Document: {st.session_state.title}\n"
        all_content += f"Format: {st.session_state.file_format}\n"
        all_content += f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        if st.session_state.current_summary is not None:
            all_content += f"## Document Summary\n\n{st.session_state.current_summary}\n\n"
        
        if st.session_state.current_analysis is not None:
            all_content += f"## Detailed Analysis\n\n{st.session_state.current_analysis}\n\n"
        
        if st.session_state.faq_text is not None:
            all_content += f"## Frequently Asked Questions\n\n{st.session_state.faq_text}\n\n"
        
        if st.session_state.current_answer is not None:
            all_content += f"## User Question & Answer\n\n{st.session_state.current_answer}\n\n"
        
        if st.session_state.current_faq_answer is not None:
            all_content += f"## FAQ Question & Answer\n\n{st.session_state.current_faq_answer}\n\n"
        
        st.markdown(
            get_download_link(
                all_content,
                f"{st.session_state.title.replace(' ', '_')}_complete_report",
                "Download Complete Report"
            ),
            unsafe_allow_html=True
        )
        
        st.info("Click the link above to download a complete report with all generated content.")
    else:
        st.info("Generate some content first using the other tabs, then come back here to export everything.")


def sidebar_content():
    """Add sidebar content with instructions"""
    st.sidebar.title("Instructions")
    st.sidebar.markdown("""
    1. Upload a legal document (PDF, DOCX, or TXT format).
    2. Click "Process Document" to analyze the content.
    3. Use the tabs to navigate between different features:
       - **Summary**: Get brief or bullet-point summaries  
       - **Detailed Analysis**: Examine specific aspects of the document  
       - **Q&A**: Ask questions about the document  
       - **FAQ**: View and get answers to frequently asked questions  
       - **Export All**: Download all generated content  
    4. Each tab has a download option to save the results.
    """)

    # Add file format support info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Supported File Formats")
    st.sidebar.markdown("""
    - 📄 **PDF** (.pdf)  
    - 📝 **Word Document** (.docx)  
    - 📃 **Text File** (.txt)  
    """)

    # Add footer with hyperlinks
    st.sidebar.markdown("---")
    st.sidebar.markdown("🚀 **Legal Document Summarizer** powered by LangChain and Gemini")
    st.sidebar.markdown("💡 Developed with passion and always open to new opportunities!")
    st.sidebar.markdown("📩 **[Contact Me](mailto:raj251003@gmail.com)**") 
    st.sidebar.markdown("🌐 **[Lets Speak](https://www.linkedin.com/in/raj-kumar-372b75320/)**")  
    st.sidebar.markdown("🌐 **[Check out my projects](https://github.com/raj-kumar-03)**")  


def main():
    """Main function to run the Streamlit app"""
    # Setup UI and initialize session state
    setup_ui()
    
    # Add sidebar content
    sidebar_content()
    
    # Document upload section
    file_uploaded = document_upload_section()
    
    # If document has been processed, show tabs with different functionalities
    if st.session_state.docs is not None and file_uploaded:
        st.header(f"Document Title: {st.session_state.title}")
        st.subheader(f"Format: {st.session_state.file_format}")
        
        # Create tabs for different functionalities
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary", "Detailed Analysis", "Q&A", "FAQ", "Export All"])
        
        with tab1:
            summary_tab()
        
        with tab2:
            detailed_analysis_tab()
        
        with tab3:
            qa_tab()
        
        with tab4:
            faq_tab()
            
        with tab5:
            export_all_tab()
    else:
        st.info("Please upload a document (PDF, DOCX, or TXT) to begin.")


if __name__ == "__main__":
    main()
