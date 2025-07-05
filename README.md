# rag_mcp
A sophisticated Retrieval-Augmented Generation (RAG) chatbot that combines multi-agent architecture with Chain of Thought (CoT) reasoning for transparent, step-by-step problem-solving.

## 🌟 Features

### Chain of Thought Reasoning
- **Intelligent Query Classification**: Automatically categorizes queries into reasoning types
- **Step-by-Step Thinking**: Transparent reasoning process with confidence scores
- **Evidence Tracking**: Links responses to source documents
- **Confidence Assessment**: Provides reliability scores for answers

### Multi-Agent Architecture
- **IngestionAgent**: Handles document processing and storage
- **RetrievalAgent**: Manages context retrieval and search
- **LLMResponseAgent**: Generates responses with CoT reasoning
- **CoTReasoningAgent**: Orchestrates the thinking process

### Document Support
- **PDF** (.pdf) - Text extraction from multi-page documents
- **Word Documents** (.docx) - Full text content processing
- **PowerPoint** (.pptx) - Slide content extraction
- **CSV Files** (.csv) - Data analysis and summarization
- **Text Files** (.txt, .md) - Plain text and Markdown support

### Reasoning Types
- **SIMPLE**: Direct factual questions
- **COMPLEX**: Multi-part queries requiring synthesis
- **ANALYTICAL**: Cause-effect analysis and explanations
- **COMPARATIVE**: Side-by-side comparisons
- **CREATIVE**: Idea generation and brainstorming

## Quick Start

### Prerequisites

1. **Python 3.8+** installed  
2. **Ollama** running locally  
3. **Required Python packages** 

### Steps

1. To manage environments I recommend using miniconda https://www.anaconda.com/docs/getting-started/miniconda?install#windows-installation 
2. conda create -n main python=3.11
3. conda activate main 
4. In this path use pip install -r requirements.txt 
 
