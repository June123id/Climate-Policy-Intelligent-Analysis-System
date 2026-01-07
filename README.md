# Climate-Policy-Intelligent-Analysis-System
 A Multi-Agent Climate Policy Analysis and Retrieval System Based on LangGraph Architecture

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0.32-green.svg)](https://github.com/langchain-ai/langgraph)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Project Overview

This system is an intelligent analysis tool for climate mitigation policy research, integrating **142,412** global climate policy records from the GCCMPD database. The system employs a multi-agent collaborative architecture, combining local vector retrieval with large language model inference to provide policy querying, entity extraction, and similarity analysis capabilities.

### Core Features

- 🔍 **Intelligent Intent Recognition**: Automatically determines whether user input is a query request or policy text analysis
- 📊 **Automatic Entity Extraction**: Extracts 7 types of key entities from policy text (policy instruments, target sectors, geographic scope, etc.)
- 🔗 **Semantic Similarity Search**: Vector retrieval based on BGE-M3 model to recommend similar policies
- 💬 **Natural Language Interaction**: Supports conversational queries in both Chinese and English
- ⚡ **Streaming Response**: Real-time display of processing progress and intermediate results
- 🎯 **Multi-dimensional Query**: Supports precise queries by country, year, sector, policy instrument, and more

### Technical Architecture

```
User Input
    ↓
[Router Agent] ──→ Intent Classification (query/analysis)
    ↓
    ├─→ [Query Agent] ──→ Database Query ──→ [Format Response]
    │
    └─→ [Analysis Agent] ──→ Entity Extraction
            ↓
        [Similarity Agent] ──→ Vector Retrieval ──→ [Format Response]
```

**Technology Stack**:
- **Agent Orchestration**: LangGraph (state graph workflow)
- **Vector Retrieval**: FAISS + BGE-M3 (local embedding model)
- **Data Storage**: SQLite (142K+ policy records)
- **LLM Inference**: LLM API (Such as Moonshot AI)
- **Backend Framework**: FastAPI + Python 3.10+

---

## 🚀 Quick Start

### System Requirements

- **Operating System**: Windows 10/11, Linux, macOS
- **Python**: 3.10 or 3.11 (3.10 recommended)
- **Memory**: 8GB+ RAM (for loading BGE-M3 model)
- **Disk Space**: ~5GB (model + data + index)
- **Network**: Access to LLM API required

### Environment Setup

#### 1. Clone the Project

```bash
git clone <repository-url>
cd code
```

#### 2. Create Virtual Environment

**Windows (PowerShell)**:
```powershell
cd backend
python -m venv venv_workflow
venv_workflow\Scripts\Activate.ps1
```

**Windows (CMD)**:
```cmd
cd backend
python -m venv venv_workflow
venv_workflow\Scripts\activate.bat
```

**Linux/macOS**:
```bash
cd backend
python -m venv venv_workflow
source venv_workflow/bin/activate
```

#### 3. Install Dependencies

```bash
# Ensure virtual environment is activated
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Core Dependencies**:
```
langgraph==0.0.32
langchain-core==0.1.53
pydantic==2.12.0
openai==2.3.0
sentence-transformers==5.1.1
faiss-cpu==1.12.0
numpy==2.3.3
aiosqlite==0.21.0
```

#### 4. Configure Environment Variables

Edit the `backend/.env` file and configure the following required parameters:

```env
#  LLM API Configuration (Required)
API_BASE=
API_KEY=your_api_key_here  # Replace with your API Key
MODEL=such as GLM 4.6

# Local Model Path (Required)
BGE_M3_PATH=D:/path/to/your/bge_m3  # Replace with your BGE-M3 model path

# Other Configurations (Optional, use defaults)
MAX_RETRIES=3
REQUEST_TIMEOUT=30
TOP_K_SIMILAR=5
BATCH_SIZE=32
```

**Obtaining Kimi API Key**:
1. Visit [Moonshot AI Platform](https://platform.moonshot.cn/)
2. Register and log in
3. Create an API Key in the console

**Downloading BGE-M3 Model**:
- Download model files from [Hugging Face](https://huggingface.co/BAAI/bge-m3)
- Or use the `transformers` library for automatic download:
  ```python
  from sentence_transformers import SentenceTransformer
  model = SentenceTransformer('BAAI/bge-m3')
  model.save('path/to/bge_m3')
  ```

#### 5. Verify Data Files

Ensure the following data files exist in the `data/` folder at the project root:

```
data/
├── policies.db          # SQLite database (~242 MB)
├── faiss_index.bin      # FAISS vector index (~583 MB)
└── metadata.pkl         # Metadata file (~9 MB)
```

If data files don't exist, you need to run the data preprocessing script (see "Data Preparation" section below).

#### 6. Test Environment Configuration

Run the path diagnostic script to verify all configurations are correct:

```bash
# In the backend directory with virtual environment activated
python check_paths.py
```

**Expected Output**:
```
================================================================================
Path Diagnostic Tool
================================================================================

Current working directory: E:\...\code\backend
...

Configured Paths:
================================================================================

DB_PATH: E:\...\code\data\policies.db
  Exists: True
  ...

Database Connection Test:
================================================================================

✓ Database connection successful!
Tables: ['policies', 'sqlite_sequence']
Policy count: 142,412
```

---

## 🌐 Web Interface

### Starting the Web Interface

The system now includes a React-based web interface for easier interaction.

#### Step 1: Start Backend API Server

**Windows**:
```cmd
cd backend
run_api.bat
```

**Linux/macOS**:
```bash
cd backend
chmod +x run_api.sh
./run_api.sh
```

The API server will start on `http://localhost:8000`

#### Step 2: Start Frontend Development Server

**Windows**:
```cmd
start_frontend.bat
```

**Linux/macOS**:
```bash
chmod +x start_frontend.sh
./start_frontend.sh
```

The frontend will be available at `http://localhost:3000`

#### Step 3: Open in Browser

Navigate to `http://localhost:3000` in your web browser to access the web interface.

### Web Interface Features

- **Chat-based Interaction**: Natural conversation interface
- **Real-time Analysis**: Submit policy text or queries and see results
- **Entity Extraction Visualization**: View extracted entities in organized cards
- **Similar Policy Recommendations**: See top-K similar policies with similarity scores
- **Query Results**: Browse matching policies with expandable details

### Using the Web Interface

1. **Policy Analysis**: Paste or type a policy text in the input box and click "发送"
   - The system will extract entities and find similar policies
   - Results are displayed in organized cards

2. **Policy Query**: Ask questions like "查找中国的交通政策" or "Find energy policies from 2020"
   - The system will search the database and return matching policies
   - Results can be expanded to see full details

3. **Bilingual Support**: The interface supports both Chinese and English inputs

For detailed frontend documentation, see [frontend/README.md](frontend/README.md)

---

## 🎮 Running Interactive Demo

### Launch Interactive Test Program

**Method 1: Using Batch Script (Windows)**

```cmd
cd backend
run_interactive.bat
```

**Method 2: Manual Launch (Cross-platform)**

```bash
cd backend
# Ensure virtual environment is activated
venv_workflow\Scripts\python.exe interactive_test.py  # Windows
# or
venv_workflow/bin/python interactive_test.py  # Linux/macOS
```

### Usage Instructions

After launching, you will see the welcome screen:

```
================================================================================
Climate Mitigation Policy Intelligent Analysis System - Interactive Test
================================================================================

Welcome to the interactive testing tool!

You can:
  1. Input policy text for analysis
  2. Input query questions to search policies
  3. Type 'quit' or 'exit' to quit
  4. Type 'examples' to view examples

================================================================================
Please enter your question or policy text (or command):
```

### Example 1: Policy Text Analysis

**Input**:
```
The Chinese government announced a new electric vehicle promotion policy. 
The policy includes subsidies for EV purchases, tax incentives for manufacturers, 
and a target to install 12,000 charging stations in major cities by 2025. 
The policy is effective from 2020 to 2030 and targets the transport sector.
```

**System Processing Flow**:
```
📍 [Intent Recognition] Intent classified as: analysis
   Result: analysis

🔍 [Entity Extraction] Extracting entities from policy text...
   instrument_type: subsidies, tax incentives
   target_sector: transport
   geographic_scope: China
   ...

🔗 [Similar Policies] Finding similar policies...
   Found 5 similar policies

   Top 3 similar policies:
   1. Electric Vehicle Promotion Policy (CN) - Similarity: 92.3%
   2. New Energy Vehicle Subsidy Program (CN) - Similarity: 88.7%
   3. Transport Electrification Strategy (CN) - Similarity: 85.1%

✅ [Complete] Processing completed

📝 Final Response:

Based on the policy text you provided, I have extracted the following key information:

**Policy Instruments**: subsidies, tax incentives
**Target Sector**: transport
**Geographic Scope**: China
**Time Range**: 2020-2030

I have also found 5 similar policies for you, with the most similar being...
```

### Example 2: Policy Query

**Input**:
```
Find transport policies from China in 2020
```

**System Processing Flow**:
```
📍 [Intent Recognition] Intent classified as: query
   Result: query

🔍 [Query Results] Executing database query...
   Found 156 policies

   Top 5 results:
   1. Electric Vehicle Promotion Policy (CN, 2020)
   2. Public Transport Subsidy Program (CN, 2020)
   3. Green Freight Initiative (CN, 2020)
   4. Urban Mobility Plan (CN, 2020)
   5. Low-Carbon Transport Strategy (CN, 2020)

✅ [Complete] Processing completed

📝 Final Response:

Based on your query, I found 156 policies that match the criteria. Here are some results:

1. **Electric Vehicle Promotion Policy**
   - Country: China
   - Year: 2020
   - Sector: Transport
   - Policy Instruments: subsidies, tax incentives
   ...
```

### Example 3: Chinese Query

**Input**:
```
查找欧洲的可再生能源政策
```

**System Response**:
```
📍 [Intent Recognition] Intent classified as: query
   Result: query

🔍 [Query Results] Executing database query...
   Found 2,847 policies
   ...
```

### Command Reference

- **`examples`**: Display more example inputs
- **`quit` / `exit` / `q`**: Exit the program
- **Empty input**: System will prompt for re-entry

---

## 📊 Project Structure

```
code/
├── backend/                      # Backend service
│   ├── agents/                   # Agent modules
│   │   ├── router.py            # Intent routing agent
│   │   ├── query.py             # Database query agent
│   │   ├── analysis.py          # Entity analysis agent
│   │   └── similarity.py        # Similarity retrieval agent
│   ├── workflows/                # LangGraph workflows
│   │   ├── langgraph_flow.py    # Main workflow definition
│   │   └── README.md            # Workflow documentation
│   ├── models/                   # Data models
│   │   └── schemas.py           # Pydantic model definitions
│   ├── database/                 # Database operations
│   │   └── db.py                # SQLite connection management
│   ├── utils/                    # Utility functions
│   ├── config.py                 # Configuration management
│   ├── .env                      # Environment variables
│   ├── requirements.txt          # Python dependencies
│   ├── interactive_test.py       # Interactive test script
│   ├── check_paths.py            # Path diagnostic tool
│   └── venv_workflow/            # Virtual environment directory
├── data/                         # Data files
│   ├── policies.db              # SQLite database
│   ├── faiss_index.bin          # FAISS vector index
│   └── metadata.pkl             # Metadata
├── scripts/                      # Data preprocessing scripts
│   └── preprocess.py            # Data import and index building
├── frontend/                     # Frontend application (optional)
├── .kiro/                        # Kiro IDE configuration
│   └── specs/                   # Project specification documents
└── README.md                     # This document
```

---

## 🔧 Advanced Configuration

### Data Preparation (Optional)

If you need to rebuild the database and index, ensure the following files exist in the project root:
- `GCCMPD1_with_entities1.xlsx`
- `GCCMPD1_with_entities2.xlsx`

Run the preprocessing script:

```bash
python scripts/preprocess.py
```

**Note**: The preprocessing process may take 30-60 minutes depending on hardware performance.

### Performance Optimization

**Adjust Batch Size**:
```env
BATCH_SIZE=16  # Reduce batch size to lower memory usage
```

**Adjust Number of Similar Policies**:
```env
TOP_K_SIMILAR=10  # Increase recommendation count
```

**Adjust API Timeout**:
```env
REQUEST_TIMEOUT=60  # Increase timeout (seconds)
```

### Test Suite

Run the complete test suite:

```bash
# Structure test
python test_workflow_structure.py

# Streaming workflow test
python test_stream_workflow.py

# Individual agent tests
python test_router_agent.py
python test_query_agent.py
python test_analysis_agent.py
python test_similarity_agent.py
```

---

## 📖 Technical Documentation

### Agent Descriptions

1. **Router Agent** (`agents/router.py`)
   - Function: Intent classification (query/analysis)
   - Input: Raw user input
   - Output: Intent label

2. **Query Agent** (`agents/query.py`)
   - Function: Structured database queries
   - Input: Query text
   - Output: List of matching policies

3. **Analysis Agent** (`agents/analysis.py`)
   - Function: Entity extraction
   - Input: Policy text
   - Output: 7 types of entities (policy instruments, target sectors, etc.)

4. **Similarity Agent** (`agents/similarity.py`)
   - Function: Vector similarity retrieval
   - Input: Policy text
   - Output: Top-K similar policies

### Workflow Documentation

For detailed workflow documentation, see: [`backend/workflows/README.md`](backend/workflows/README.md)

### Data Model

Database schema:

```sql
CREATE TABLE policies (
    id INTEGER PRIMARY KEY,
    policy_name TEXT,
    country_code TEXT,
    year INTEGER,
    target_sector TEXT,
    instrument_type TEXT,
    geographic_scope TEXT,
    policy_objective TEXT,
    implementation_status TEXT,
    policy_description TEXT,
    ...
);
```

---

## 🐛 Troubleshooting

### Issue 1: Database Connection Failed

**Error Message**: `unable to open database file`

**Solution**:
1. Run `python check_paths.py` to check path configuration
2. Confirm `data/policies.db` file exists
3. Check if path configurations in `.env` file are commented out

### Issue 2: Model Loading Failed

**Error Message**: `OSError: Unable to load weights from pytorch checkpoint file`

**Solution**:
1. Check if `BGE_M3_PATH` configuration is correct
2. Confirm model files are complete (~2GB)
3. Re-download BGE-M3 model

### Issue 3: API Call Failed

**Error Message**: `API request failed` or `Unauthorized`

**Solution**:
1. Check if `KIMI_API_KEY` is correct
2. Confirm API Key has sufficient quota
3. Check network connection

### Issue 4: Out of Memory

**Error Message**: `MemoryError` or system freezing

**Solution**:
1. Reduce `BATCH_SIZE` parameter (e.g., change to 16 or 8)
2. Close other memory-intensive programs
3. Consider using a smaller embedding model

### Issue 5: Virtual Environment Activation Failed (Windows PowerShell)

**Error Message**: `cannot be loaded because running scripts is disabled`

**Solution**:
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Or use CMD instead of PowerShell.

---

## 📚 References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [BGE-M3 Model](https://huggingface.co/BAAI/bge-m3)
- [Kimi API Documentation](https://platform.moonshot.cn/docs)

---

## 📄 License

MIT License

---

**Last Updated**: 2025-10-12
