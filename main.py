import streamlit as st
import json
from navigator.decision_pipeline import runnable, build_report
from utils.chromadb_handler import chroma_db
import os
from langchain_core.agents import AgentAction

# Set page configuration
st.set_page_config(page_title="AI Research Assistant Navigator", layout="wide")
st.title("AI Research Assistant Navigator")

st.markdown("""
This app retrieves relevant research papers from a CSV dataset, downloads the selected paper, 
stores it into ChromaDB, and then uses an AI Navigator to compile a final research report.
""")

# ---- Step 1: Retrieve Relevant Research Papers from CSV ----
st.header("Step 1: Retrieve Relevant Research Papers")
paper_query = st.text_input("Enter a research query to find relevant papers:", value="Recent advances in AI")

if st.button("Retrieve Papers"):
    st.info("Searching for relevant papers...")
    relevant_papers = chroma_db.retrieve_relevant_papers(query_text=paper_query, top_k=5)

    if relevant_papers:
        st.success(f"Found {len(relevant_papers)} relevant papers:")
        st.table(relevant_papers)
    else:
        st.warning("No relevant papers found. Try a different query.")

# ---- Step 2: Download & Process a Selected Paper ----
st.header("Step 2: Select and Process a Research Paper")
selected_pdf_url = st.text_input("Enter the PDF URL of the paper to download:")
selected_arxiv_id = st.text_input("Enter the ArXiv ID of the selected paper:")

if st.button("Download and Store Paper"):
    if selected_pdf_url and selected_arxiv_id:
        st.info("Downloading paper PDF...")
        pdf_path = chroma_db.download_pdf(selected_pdf_url)

        if not pdf_path:
            st.error("Failed to download the PDF. Please check the URL.")
        else:
            st.success(f"Downloaded PDF: {pdf_path}")
            st.info("Storing paper chunks into the knowledge base...")
            chroma_db.store_chunks(selected_arxiv_id, pdf_path)
            st.success("Paper chunks stored successfully!")
    else:
        st.error("Please provide both the PDF URL and the ArXiv ID.")

# ---- Step 3: Run Navigator with Final Query & Selected ArXiv ID ----
st.header("Step 3: Generate Research Report via Navigator")
final_query = st.text_input("Enter your final research query for the Navigator:", value="Summarize key insights from the paper")
final_arxiv_id = st.text_input("Enter the ArXiv ID to be used by the Navigator:")

if st.button("Run Navigator"):
    st.info("Running Navigator Pipeline...")
    try:
        initial_state = {
            'input': final_query,
            'chat_history': [],
            'intermediate_steps': [
                (AgentAction(tool="fetch_arxiv", tool_input={"input": final_arxiv_id}, log="TBD"), "")
            ]
        }

        st.write("### DEBUG: Initial State:", initial_state)

        result_state = runnable.invoke(initial_state)

        if not result_state or 'intermediate_steps' not in result_state:
            st.error("Navigator did not return expected results.")
            st.stop()

        st.write("### DEBUG: Result State:", result_state)

        output = {
            "introduction": "This report summarizes the research findings based on your query.",
            "research_steps": [
                f"{step[0].tool}: {json.dumps(step[0].tool_input, indent=2)} -> {step[1]}"
                for step in result_state['intermediate_steps']
            ],
            "main_body": "Detailed insights are extracted from the selected paper and complementary web search results.",
            "conclusion": "The research demonstrates the potential and breadth of current AI developments.",
            "sources": ["ArXiv", "SerpAPI", "ChromaDB"]
        }

        report = build_report(output)
        st.success("Research Report Generated!")
        st.text_area("Final Research Report", report, height=500)

    except Exception as e:
        st.error(f"Error running Navigator: {e}")
