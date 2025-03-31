# Legal Document Summarizer

A Streamlit application that leverages LangChain and Google's Gemini AI to analyze, summarize, and extract insights from legal documents.

## Features

- **Multiple File Format Support**: Process PDF, DOCX, and TXT documents
- **Document Summarization**: Generate concise paragraph or bullet-point summaries
- **Detailed Analysis**: Extract specific information such as:
  - Key Obligations
  - Rights & Responsibilities
  - Legal Risks & Penalties
  - Important Dates & Deadlines
- **Interactive Q&A**: Ask questions about the document and receive answers with source citations
- **Automatic FAQ Generation**: AI-generated frequently asked questions based on document content
- **Export Options**: Download individual analyses or a complete report

## Requirements

- Python 3.7+
- Streamlit
- LangChain components:
  - `langchain_community` (for document loaders)
  - `langchain_text_splitters`
  - `langchain` (for chains)
  - `langchain_core` (for prompts)
  - `langchain_google_genai` (for Google AI integration)
  - `langchain_community.vectorstores` (for FAISS)
- PyMuPDF (fitz)
- Google Generative AI Python SDK
- FAISS vector database
- docx2txt (for Word document processing)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/raj-kumar-03/law_ai.git
   cd law_ai
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   venv\Scripts\activate
   source venv/bin/activate #linux use this
   
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up your Google API key:
   - Create a `.streamlit/secrets.toml` file with:
     ```
     GOOGLE_API_KEY = "your_google_api_key_here"
     ```
   - Sign up for a [Google AI Studio](https://ai.google.dev/) account to get an API key

## Usage

1. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

2. Access the application in your web browser (typically at http://localhost:8501)

3. Upload a legal document (PDF, DOCX, or TXT format)

4. Click "Process Document" to analyze the content

5. Navigate through the tabs to:
   - Generate summaries
   - Create detailed analyses
   - Ask questions about the document
   - View AI-generated FAQs
   - Export all generated content

## How It Works

1. **Document Processing**: The application uses PyMuPDF for PDFs, docx2txt for Word documents, and in-memory processing for TXT files.

2. **Text Chunking**: Documents are split into manageable chunks using RecursiveCharacterTextSplitter, with special attention to legal document structures.

3. **Embedding and Vector Storage**: Document chunks are embedded using Google's embedding model and stored in a FAISS vector database for efficient retrieval.

4. **AI Analysis**: Google's Gemini 1.5 Flash model processes the document to generate:
   - Summaries
   - Detailed analyses
   - Answers to user questions
   - Frequently asked questions

5. **Results Export**: All generated content can be downloaded individually or as a complete report.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)

## Contact

For questions or collaboration opportunities, please contact:
- Email: raj251003@gmail.com
- Linkedin: [Linkedin](https://www.linkedin.com/in/raj-kumar-372b75320/)
