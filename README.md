## Interactive Agentic RAG Investigation

This project is an interactive AI investigation system built using *The Sign of Four* by Arthur Conan Doyle. The PDF is used as the raw corpus and is preprocessed using PyMuPDF by extracting the text, removing unnecessary pages, identifying chapters, preserving page information and splitting the book into 181 chunks.

A hybrid RAG approach is used for retrieval, combining BM25 for keyword-based search and Sentence Transformers (`all-MiniLM-L6-v2`) for semantic search. Both results are combined using Reciprocal Rank Fusion (RRF) to retrieve relevant evidence.

The Gemini API (`gemini-3.6-flash`) is used as the reasoning layer. An Investigator Agent takes the user question and retrieved evidence, forms a hypothesis and can refine the search when more information is needed. A separate Fact-Checker Agent performs additional retrieval to find supporting or contradicting evidence. Finally, a grounded verdict is generated with evidence IDs and evidence marked as verified, misleading or unverified.

An evidence graph is also created using NetworkX to show relationships between questions, claims, evidence and people. The backend is built using FastAPI and provides APIs for investigation, evidence search, candidate interrogation, graph data and verdict submission. The frontend is built using Streamlit with sections for the Case Room, Evidence Search, Candidate Interrogation and Evidence Graph.

Tech Stack: Python, FastAPI, Streamlit, Gemini API, BM25, Sentence Transformers, PyMuPDF, Pydantic, NumPy, NetworkX and Plotly.

### Links

Frontend: https://agenticaiinvestigator-fpzuyxwxwo93bhdwpugxnj.streamlit.app/

Backend: https://agentic-ai-investigator-backend.onrender.com/docs

Demo video drive link: https://drive.google.com/drive/folders/1D1Vbkgi5PCM1Vee_PYxg6rxsIkNEGqMV?usp=sharing

### Deployment Note

The backend is deployed on Render and basic APIs like health, corpus and candidates are working. However, the complete investigation API currently gives a 502 error on the Render free tier because of its limited CPU and 512 MB memory. The semantic retrieval model and complete agent pipeline require more runtime resources. The complete pipeline has been tested locally, but paid deployment was not used, so the public deployment is partially limited by the free-tier resources.