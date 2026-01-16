# 🚀 Multimodal RAG System (PDF with Images)

Ever wished you could ask questions about a PDF and get answers that actually understand both the text *and* the charts, diagrams, and images? Well, now you can! 

This is a **Retrieval-Augmented Generation (RAG) system** that reads PDFs like a human would—processing text and images together to give you comprehensive, context-aware answers.

## ✨ What Makes It Special

- **Reads Both Text & Images**: Extracts everything from your PDF and actually understands both
- **Unified Understanding**: Uses AI to treat text and images as the same "language" (CLIP embeddings)
- **Lightning-Fast Search**: Finds relevant information in seconds using FAISS
- **Smart Answers**: GPT-4V doesn't just read text—it looks at your charts, graphs, and diagrams
- **Gets the Full Picture**: Combines everything it finds to give you complete answers

## 🛠️ Built With

- **PyMuPDF** — Extracts text and images from PDFs cleanly
- **OpenAI CLIP** — The secret sauce: creates embeddings that understand both text and pictures
- **FAISS** — Super-fast database that finds similar content instantly
- **LangChain** — Makes working with AI models smooth and organized
- **GPT-4V** — The smart brain that sees images and understands them
- **Transformers** — Loads all the AI models we need
- **scikit-learn** — Math library for comparing embeddings
- **PIL** — Handles image processing

## 📋 Before You Start

- Python 3.8 or newer (grab it from python.org if you don't have it)
- An OpenAI account with GPT-4V access (you'll need an API key)
- A few Python packages (we'll install those for you)

## 🚀 Get Started in 5 Minutes

**Step 1: Clone the repo**
```bash
git clone https://github.com/eeshaanbharadwaj/MultiModel-RAG.git
cd MultiModel-RAG
```

**Step 2: Set up a virtual environment** (keep things clean!)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

**Step 3: Install everything**
```bash
pip install -r requirements.txt
```

**Step 4: Add your API key**
Create a `.env` file in the project folder and add:
```
OPENAI_API_KEY=your_openai_api_key_here
```

**Step 5: Run it!**
```bash
jupyter notebook 1-multimodalopenai.ipynb
```

That's it! You're ready to go.

## 💻 How to Use It

1. **Get your PDF ready**
   - Drop your PDF in the project folder
   - Update the `pdf_path` variable to point to it

2. **Start the notebook**
```bash
jupyter notebook 1-multimodalopenai.ipynb
```

3. **Ask questions!**
```python
# Just run this:
answer = multimodal_pdf_rag_pipeline("What does the chart show?")
print(answer)
```

That's all there is to it. Ask anything, get smart answers. 🎉

## 🔄 How It Actually Works (In Plain English)

```
📄 You drop a PDF here
          ↓
🔍 We pull out all the text and images
          ↓
📝 Break text into chunks | 🖼️ Extract images
          ↓
🧠 Convert everything to CLIP embeddings
   (think of it as "language" that AI understands)
          ↓
💾 Store in FAISS (a super-fast search database)
          ↓
❓ You ask a question
          ↓
⚡ System finds relevant stuff in milliseconds
          ↓
🎨 Builds a message with text + images + question
          ↓
👀 GPT-4V looks at everything and thinks
          ↓
💬 Gives you a smart, complete answer
```

## 🧩 The Moving Parts

### 1. PDF Processing 📄
- Grabs all the text from each page
- Finds every image embedded in the PDF
- Converts images to a format AI can work with
- Turns images into Base64 (fancy encoding for web)

### 2. Creating Embeddings 🧠
- Uses CLIP to turn everything into numbers (embeddings)
- Text and images get the same treatment—unified!
- Normalizes them so they're easy to compare

### 3. Building the Database 🗄️
- FAISS stores all your embeddings with their info
- Super fast at finding the most similar stuff

### 4. Getting Answers 🎯
- Takes your question and turns it into an embedding too
- Searches for related documents (text + images)
- Builds a smart message with everything relevant
- GPT-4V reads it all and gives you an answer

## 📝 Example Questions to Ask

These are the kinds of things you can ask:

- "What does the chart on page 1 show about revenue?"
- "Explain the relationship between this graph and the text"
- "What's the main takeaway from this document?"
- "Describe all the visual elements in this PDF"
- "What do these diagrams tell us?"

## ⚙️ Tweaking Things

Want to customize how it works? These are the main knobs you can turn:

```python
# How big should each text chunk be?
chunk_size = 500
chunk_overlap = 100

# How many results should we find?
k = 5  # More = more context but slower

# Which CLIP model to use?
model_name = "openai/clip-vit-base-patch32"
```

## 🤔 Heads Up

- You'll need OpenAI API credits (costs vary with usage)
- Big PDFs will use more tokens—watch your API usage
- Image quality matters—clear charts work better
- CLIP has a limit of 77 tokens per text (usually not a problem)

## 🎨 Ideas to Make This Even Better

Have ideas? Here are some cool things you could add:

- Support other file types (Word docs, PowerPoints, etc.)
- Process multiple PDFs at once
- Cache results to save on API costs
- Train CLIP on your specific industry
- Build a web interface so anyone can use it
- Add conversation memory (remember earlier questions)
- Export answers as formatted reports

Feel free to fork and improve! 🚀

## � License & Credits

MIT License — use it however you like!

**Big thanks to:**
- OpenAI for CLIP and GPT-4V (the brains of this project)
- LangChain team for making AI integration painless
- PyMuPDF devs for reliable PDF handling

---

**Found this helpful?** Please drop a ⭐ on GitHub and share it with your network!
