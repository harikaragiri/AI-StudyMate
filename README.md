## AI StudyMate – AI Powered Study Assistant

AI StudyMate is an AI-powered study assistant designed to help students and learners search answers from notes, upload study materials, and generate concise summaries using semantic search and natural language processing.
This project demonstrates real-world AI application, efficient document ingestion, vector-based retrieval, and a clean Streamlit user interface.

Built as part of the Endee.io Technical Assignment, this project showcases skills in AI, NLP, Python, and full-stack application development.

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

## How It Works (Technical Overview):
1. Uploaded documents are split into chunks for efficient retrieval
2. Each chunk is converted into a vector embedding using Sentence Transformers
3. Embeddings are stored in a local vector database
4. User questions are encoded into vectors
5. The system performs cosine similarity matching to retrieve the most relevant answers
6. A summarization module extracts the most informative sentences
7. Query history is saved to enhance user experience
This demonstrates applied AI, NLP pipelines, vector search systems, and scalable retrieval design.

## Real-World Use Cases

- Smart study companion for students
- AI-based note search engine
- Academic research assistant
- Knowledge retrieval tool for institutions
- Prototype for enterprise AI knowledge systems
- Personal AI learning assistant

## Submission Note
This project is submitted as part of the Endee.io assignment and demonstrates semantic search, document ingestion, and AI-powered study assistance using local embeddings.
