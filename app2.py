import streamlit as st
import requests
import json
import time
from typing import Dict, List, Any
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd

# Configure Streamlit page
st.set_page_config(
    page_title="RAG Chatbot with Chain of Thought",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_BASE_URL = "http://localhost:8000"
REASONING_TYPE_COLORS = {
    "simple": "#2E8B57",
    "complex": "#FF6B6B",
    "analytical": "#4ECDC4",
    "comparative": "#45B7D1",
    "creative": "#FFA07A"
}

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "cot_history" not in st.session_state:
    st.session_state.cot_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "confirm_reset" not in st.session_state:
    st.session_state.confirm_reset = False

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def upload_document(file):
    """Upload document to the API"""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = requests.post(f"{API_BASE_URL}/ingest", files=files, timeout=60)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def query_with_cot(query: str):
    """Query with Chain of Thought reasoning"""
    try:
        data = {"query": query}
        response = requests.post(f"{API_BASE_URL}/query", data=data, timeout=60)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def query_simple(query: str):
    """Simple query without Chain of Thought"""
    try:
        data = {"query": query}
        response = requests.post(f"{API_BASE_URL}/query-simple", data=data, timeout=60)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def get_reasoning_stats():
    """Get reasoning statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/reasoning-stats", timeout=10)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def reset_database():
    """Reset the vector database"""
    try:
        response = requests.delete(f"{API_BASE_URL}/reset", timeout=30)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def display_thought_process(thought_process: List[Dict]):
    """Display the Chain of Thought process"""
    st.subheader("🧠 Chain of Thought Process")
    
    for step in thought_process:
        with st.expander(f"Step {step['step_number']}: {step['reasoning'][:50]}..."):
            st.write(f"**Reasoning:** {step['reasoning']}")
            if step.get('evidence'):
                st.write(f"**Evidence:** {step['evidence']}")
            if step.get('sub_questions'):
                st.write("**Sub-questions:**")
                for sub_q in step['sub_questions']:
                    st.write(f"- {sub_q}")
            
            # Confidence meter
            confidence = step.get('confidence', 0.0)
            st.metric("Confidence", f"{confidence:.2f}")
            st.progress(confidence)

def display_reasoning_stats():
    """Display reasoning statistics"""
    success, stats = get_reasoning_stats()
    
    if success and "total_cot_queries" in stats:
        st.subheader("📊 Reasoning Analytics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total CoT Queries", stats["total_cot_queries"])
        
        with col2:
            st.metric("Average Confidence", f"{stats['average_confidence']:.3f}")
        
        with col3:
            if stats.get("confidence_range"):
                conf_range = stats["confidence_range"]
                st.metric("Confidence Range", f"{conf_range['min']:.2f} - {conf_range['max']:.2f}")
        
        # Reasoning type distribution
        if stats.get("reasoning_type_distribution"):
            st.subheader("Reasoning Type Distribution")
            reasoning_data = stats["reasoning_type_distribution"]
            
            # Create pie chart
            fig = px.pie(
                values=list(reasoning_data.values()),
                names=list(reasoning_data.keys()),
                title="Query Reasoning Types",
                color_discrete_map=REASONING_TYPE_COLORS
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No reasoning statistics available yet. Ask some questions to see analytics!")

def main():
    st.title("🧠 RAG Chatbot with Chain of Thought")
    st.markdown("*Intelligent document querying with step-by-step reasoning*")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Controls")
        
        # API Health Check
        health_success, health_data = check_api_health()
        if health_success:
            st.success("✅ API Connected")
            if health_data.get("components"):
                components = health_data["components"]
                st.write(f"**ChromaDB:** {components.get('chromadb', 'unknown')}")
                st.write(f"**Ollama:** {components.get('ollama', 'unknown')}")
                st.write(f"**Embedding Model:** {components.get('embedding_model', 'unknown')}")
        else:
            st.error("❌ API Disconnected")
            st.write("Make sure your FastAPI server is running on http://localhost:8000")
            return
        
        st.divider()
        
        # Document Upload
        st.header("📄 Document Upload")
        uploaded_file = st.file_uploader(
            "Choose a document",
            type=['pdf', 'docx', 'pptx', 'txt', 'md', 'csv'],
            help="Upload PDF, DOCX, PPTX, TXT, MD, or CSV files"
        )
        
        if uploaded_file is not None:
            if st.button("📤 Upload Document"):
                with st.spinner("Uploading and processing document..."):
                    success, result = upload_document(uploaded_file)
                    if success:
                        st.success(f"✅ Document uploaded successfully!")
                        st.write(f"**File:** {result.get('filename')}")
                        st.write(f"**Chunks:** {result.get('chunks_created')}")
                        st.session_state.uploaded_files.append({
                            "filename": result.get('filename'),
                            "chunks": result.get('chunks_created'),
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                    else:
                        st.error(f"❌ Upload failed: {result.get('error')}")
        
        # Show uploaded files
        if st.session_state.uploaded_files:
            st.subheader("📚 Uploaded Documents")
            for file_info in st.session_state.uploaded_files[-5:]:  # Show last 5
                st.write(f"📄 {file_info['filename']}")
                st.caption(f"Chunks: {file_info['chunks']} | {file_info['timestamp']}")
        
        st.divider()
        
        # Settings
        st.header("⚙️ Settings")
        query_mode = st.radio(
            "Query Mode",
            ["Chain of Thought", "Simple"],
            help="Choose between CoT reasoning or simple querying"
        )
        
        show_reasoning = st.checkbox("Show Reasoning Process", value=True)
        show_sources = st.checkbox("Show Sources", value=True)
        
        st.divider()
        
        # Reset Database
        st.header("🔄 Reset")
        
        # Two-step confirmation process
        if not st.session_state.confirm_reset:
            if st.button("🗑️ Reset Database", type="secondary"):
                st.session_state.confirm_reset = True
                st.rerun()
        else:
            st.warning("⚠️ Are you sure you want to reset the database? This will delete all uploaded documents.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, Reset", type="primary"):
                    with st.spinner("Resetting database..."):
                        success, result = reset_database()
                        if success:
                            st.success("✅ Database reset successfully!")
                            st.session_state.uploaded_files = []
                            st.session_state.messages = []
                            st.session_state.cot_history = []
                            st.session_state.confirm_reset = False
                            st.rerun()
                        else:
                            st.error(f"❌ Reset failed: {result.get('error')}")
                            st.session_state.confirm_reset = False
            with col2:
                if st.button("❌ Cancel"):
                    st.session_state.confirm_reset = False
                    st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show reasoning process if available
                if message.get("reasoning_data") and show_reasoning:
                    display_thought_process(message["reasoning_data"]["thought_process"])
                
                # Show metadata
                if message.get("metadata"):
                    metadata = message["metadata"]
                    col_meta1, col_meta2, col_meta3 = st.columns(3)
                    with col_meta1:
                        if metadata.get("confidence_score"):
                            st.metric("Confidence", f"{metadata['confidence_score']:.2f}")
                    with col_meta2:
                        if metadata.get("reasoning_type"):
                            st.metric("Reasoning Type", metadata["reasoning_type"].title())
                    with col_meta3:
                        if metadata.get("context_count"):
                            st.metric("Context Sources", metadata["context_count"])
                
                # Show sources
                if message.get("sources") and show_sources:
                    with st.expander("📚 Sources"):
                        for source in message["sources"]:
                            st.write(f"• {source}")
    
    with col2:
        st.header("📊 Analytics")
        display_reasoning_stats()
        
        # CoT History
        if st.session_state.cot_history:
            st.subheader("🕐 Recent CoT Queries")
            for i, item in enumerate(reversed(st.session_state.cot_history[-5:])):
                with st.expander(f"Query {len(st.session_state.cot_history) - i}: {item['query'][:30]}..."):
                    st.write(f"**Query:** {item['query']}")
                    st.write(f"**Reasoning Type:** {item['response']['reasoning_type'].title()}")
                    st.write(f"**Confidence:** {item['response']['confidence_score']:.2f}")
                    st.write(f"**Time:** {item['timestamp'].strftime('%H:%M:%S')}")
        
        # Quick Actions
        st.subheader("🚀 Quick Actions")
        
        sample_queries = [
            "What are the main topics in the documents?",
            "Compare the different approaches mentioned",
            "Analyze the key findings",
            "What are the implications of the results?",
            "Create a summary of the main points"
        ]
        
        for query in sample_queries:
            if st.button(query, key=f"sample_{query}", use_container_width=True):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": query})
                
                # Process the query immediately
                if query_mode == "Chain of Thought":
                    success, result = query_with_cot(query)
                    
                    if success:
                        # Store reasoning data
                        reasoning_data = {
                            "thought_process": result["thought_process"],
                            "reasoning_type": result["reasoning_type"],
                            "confidence_score": result["confidence_score"]
                        }
                        
                        # Add to session state
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["final_answer"],
                            "reasoning_data": reasoning_data,
                            "metadata": {
                                "confidence_score": result["confidence_score"],
                                "reasoning_type": result["reasoning_type"],
                                "context_count": result["context_count"]
                            },
                            "sources": result.get("evidence_sources", [])
                        })
                        
                        # Add to CoT history
                        st.session_state.cot_history.append({
                            "query": query,
                            "response": result,
                            "timestamp": datetime.now()
                        })
                    else:
                        error_msg = f"❌ Error: {result.get('error', 'Unknown error')}"
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })
                
                else:  # Simple mode
                    success, result = query_simple(query)
                    
                    if success:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["response"],
                            "metadata": {
                                "context_count": result["context_count"]
                            },
                            "sources": result.get("sources", [])
                        })
                    else:
                        error_msg = f"❌ Error: {result.get('error', 'Unknown error')}"
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })
                
                # Rerun to show the new messages
                st.rerun()
    
    # Chat input (must be outside columns/sidebar)
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if query_mode == "Chain of Thought":
                    success, result = query_with_cot(prompt)
                    
                    if success:
                        st.markdown(result["final_answer"])
                        
                        # Store reasoning data
                        reasoning_data = {
                            "thought_process": result["thought_process"],
                            "reasoning_type": result["reasoning_type"],
                            "confidence_score": result["confidence_score"]
                        }
                        
                        # Add to session state
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["final_answer"],
                            "reasoning_data": reasoning_data,
                            "metadata": {
                                "confidence_score": result["confidence_score"],
                                "reasoning_type": result["reasoning_type"],
                                "context_count": result["context_count"]
                            },
                            "sources": result.get("evidence_sources", [])
                        })
                        
                        # Add to CoT history
                        st.session_state.cot_history.append({
                            "query": prompt,
                            "response": result,
                            "timestamp": datetime.now()
                        })
                        
                        if show_reasoning:
                            display_thought_process(result["thought_process"])
                        
                        # Show metadata
                        col_meta1, col_meta2, col_meta3 = st.columns(3)
                        with col_meta1:
                            st.metric("Confidence", f"{result['confidence_score']:.2f}")
                        with col_meta2:
                            st.metric("Reasoning Type", result["reasoning_type"].title())
                        with col_meta3:
                            st.metric("Context Sources", result["context_count"])
                        
                        if show_sources and result.get("evidence_sources"):
                            with st.expander("📚 Sources"):
                                for source in result["evidence_sources"]:
                                    st.write(f"• {source}")
                    else:
                        error_msg = f"❌ Error: {result.get('error', 'Unknown error')}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })
                
                else:  # Simple mode
                    success, result = query_simple(prompt)
                    
                    if success:
                        st.markdown(result["response"])
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["response"],
                            "metadata": {
                                "context_count": result["context_count"]
                            },
                            "sources": result.get("sources", [])
                        })
                        
                        # Show metadata
                        st.metric("Context Sources", result["context_count"])
                        
                        if show_sources and result.get("sources"):
                            with st.expander("📚 Sources"):
                                for source in result["sources"]:
                                    st.write(f"• {source}")
                    else:
                        error_msg = f"❌ Error: {result.get('error', 'Unknown error')}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })

if __name__ == "__main__":
    main()