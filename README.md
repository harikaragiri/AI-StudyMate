# AI StudyMate – AI Powered Study Assistant

AI StudyMate is an AI-powered study assistant that allows students to upload notes, search for relevant answers, and generate concise summaries using semantic search and Natural Language Processing (NLP). It uses vector-based retrieval to deliver accurate and context-aware information from documents. Built with Python and Streamlit, the project showcases practical skills in AI, NLP, document processing, and intelligent information retrieval.

## FEATURES
 - Semantic Question Answering from uploaded study notes
 - Upload and process PDF and TXT documents
 - AI-powered Sentence Embedding Search using Sentence Transformers
 - Automatic Text Summarization using embedding relevance scoring
 - Search History Tracking for user queries
 - Lightweight Local Vector Database using JSON
 - Clean, modern, and responsive Streamlit UI
 - Efficient chunking and ingestion pipeline for large documents
 - Fully offline and API-free AI processing

## TECH STACK
1. Python 3.12 — Core backend logic
2. Streamlit — Frontend UI & app deployment
3. Sentence Transformers — AI Embeddings & NLP
4. PyPDF2 — PDF text extraction
5. NumPy — Vector similarity computations
6. JSON — Lightweight local database
7. Git & GitHub — Version control & collaboration

## PROJECT STRUCTURE
AI-StudyMate-Endee/

├── app.py                 
├── ingest.py             
├── query.py              
├── endee_db.json         
├── search_history.json   
├── requirements.txt      
├── README.md             
└── screenshots/          


## INSTALLATION & SETUP
1. Clone the Repository
git clone https://github.com/harikaragiri/AI-StudyMate.git
cd AI-StudyMate

2. Install Dependencies
pip install -r requirements.txt

3. Run the Project
streamlit run app.py

4. Open Browser
http://localhost:8501

## HOW TO TEST THE WORKING OF THE USE CASE (IMPORTANT)
1. Upload Study Notes

    Upload a PDF or TXT file containing study material
    The system automatically extracts and chunks the content

2. Document Processing
    Text is converted into vector embeddings
    Embeddings are stored in a local JSON-based vector database

3. Ask a Question
    Enter a question related to the uploaded notes
    Example: “What is machine learning?”

4. Semantic Search Execution
    Query is converted into an embedding
    System finds the most relevant text chunks using cosine similarity

5. View AI Answer
    App displays a context-aware answer from the document
    A short summary is generated when applicable

6. Summarize Paragraph or Content 
    Paste any paragraph or section in the input box
    The AI provides a concise summarized version of the text
    Helps quickly understand large content

7. Check Search History
    Previous user queries are stored in search_history.json
    Confirms query tracking feature

8. Test Multiple Queries
    Ask different questions to validate retrieval accuracy
    Confirm responses change based on document content

## Real-World Use Cases

- Smart study companion for students
- AI-based note search engine
- Academic research assistant
- Knowledge retrieval tool for institutions
- Prototype for enterprise AI knowledge systems
- Personal AI learning assistant

## Submission Note
This project is submitted as part of the Endee.io assignment and demonstrates semantic search, document ingestion, and AI-powered study assistance using local embeddings.
