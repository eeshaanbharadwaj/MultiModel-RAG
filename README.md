# Multimodal RAG System (PDF with Images)

A powerful Retrieval-Augmented Generation (RAG) system that processes PDFs containing both text and images using CLIP embeddings and GPT-4V for unified semantic understanding.

## 🎯 Features

- **Multimodal Processing**: Extracts and processes both text and images from PDF documents
- **Unified Embeddings**: Uses OpenAI's CLIP model to create compatible embeddings for text and images
- **Semantic Search**: FAISS vector store enables efficient similarity-based retrieval
- **GPT-4V Integration**: Leverages GPT-4 Vision to understand and answer questions about visual content
- **Context-Aware Responses**: Combines text and image context for comprehensive answers

## 🛠️ Technologies Used

- **PyMuPDF (fitz)**: PDF text and image extraction
- **OpenAI CLIP**: Unified embeddings for text and images
- **FAISS**: Vector similarity search and storage
- **LangChain**: LLM orchestration and prompt management
- **GPT-4V**: Vision-capable language model for understanding images
- **Transformers**: Loading pre-trained CLIP models
- **scikit-learn**: Vector similarity calculations
- **PIL**: Image processing

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key with GPT-4V access
- Required Python packages (see Installation)

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/eeshaanbharadwaj/MultiModel-RAG.git
cd MultiModel-RAG
```

2. **Create a virtual environment** (optional but recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## 💻 Usage

1. **Prepare your PDF**
   - Place your PDF file in the project directory
   - Update the `pdf_path` variable in the notebook to point to your PDF

2. **Run the notebook**
```bash
jupyter notebook 1-multimodalopenai.ipynb
```

3. **Query the system**
```python
# Example query
answer = multimodal_pdf_rag_pipeline("What does the chart on page 1 show about revenue trends?")
print(answer)
```

## 🔄 How It Works

```
PDF Document
    ↓
Extract Text & Images (PyMuPDF)
    ↓
Text Chunks (RecursiveCharacterTextSplitter) | Image Extraction
    ↓
CLIP Embeddings (Unified Representation)
    ↓
FAISS Vector Store
    ↓
Query → Semantic Search
    ↓
Retrieve Relevant Documents (Text + Images)
    ↓
Create Multimodal Message (Text + Base64 Images)
    ↓
GPT-4V Response
    ↓
Answer with Visual Understanding
```

## 📊 Key Components

### 1. **PDF Processing**
- Extracts text from each page
- Identifies and extracts embedded images
- Converts images to PIL format and Base64 encoding

### 2. **Embedding Generation**
- Uses CLIP model for unified text-image embeddings
- Normalizes embeddings to unit vectors for consistent similarity computation

### 3. **Vector Storage**
- FAISS index stores all embeddings with metadata
- Enables fast K-nearest neighbor search

### 4. **Retrieval & Response**
- Encodes query using CLIP
- Retrieves similar documents (text + images)
- Constructs multimodal prompt for GPT-4V
- Returns comprehensive answer with visual understanding

## 📝 Example Queries

- "What does the chart on page 1 show about revenue trends?"
- "Summarize the main findings from the document"
- "What visual elements are present in the document?"
- "Explain the relationship between this graph and the text"

## 🔧 Configuration

Adjust these parameters in the notebook as needed:

```python
# Text chunking
chunk_size = 500
chunk_overlap = 100

# Retrieval
k = 5  # Number of documents to retrieve

# CLIP model
model_name = "openai/clip-vit-base-patch32"
```

## ⚠️ Limitations

- Requires valid OpenAI API key with GPT-4V access
- Processing large PDFs may consume significant API tokens
- Image quality affects CLIP embedding accuracy
- CLIP has a maximum text token length of 77

## 🤝 Contributing

Feel free to fork, modify, and enhance this project! Some ideas:

- Support for other document formats (DOCX, PPTX)
- Batch processing for multiple PDFs
- Caching mechanisms for reduced API costs
- Fine-tuning CLIP for domain-specific tasks
- Web interface for easier interaction

## 📜 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- OpenAI for CLIP and GPT-4V models
- LangChain community for the excellent framework
- PyMuPDF for robust PDF processing

## 📧 Contact

For questions or suggestions, feel free to reach out!

---

**Star ⭐ this repo if you found it helpful!**
