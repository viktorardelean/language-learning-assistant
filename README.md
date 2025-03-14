# Language Learning Assistant

A generative AI-powered tool designed for the **Generative AI Bootcamp**.

## 📌 Overview

**Difficulty:** Level 200 *(Due to RAG implementation and integration with multiple AWS services)*  

This project demonstrates how **Retrieval-Augmented Generation (RAG)** and **agents** enhance language learning by grounding AI-generated responses in real Spanish lesson content. It showcases the evolution from basic LLM responses to a fully contextual learning assistant, helping students grasp both **technical implementation** and **practical benefits** of RAG.

---

## 🎯 Business Goals

- Provide a **progressive learning** experience using generative AI.
- Enhance **language learning** through structured and contextualized responses.
- Demonstrate the **technical evolution** from a base LLM to a RAG-based interactive assistant.

---

## ⚙️ Technical Challenges & Uncertainties

1. **Bilingual Processing**: How effectively can we structure and retrieve **Spanish/English** content for RAG?
2. **Content Chunking & Embedding**: What’s the **optimal strategy** for breaking down and embedding Spanish language materials?
3. **Stepwise Evolution**: How can we **clearly illustrate** the transition from a **basic LLM** to **RAG-based contextual learning**?
4. **Context Retention**: Can we **accurately maintain context** when retrieving Spanish learning materials?
5. **Balance in Response Types**: How do we balance **direct answers** with **educational guidance**?
6. **Multiple-Choice Questions**: What’s the **best approach** to structure MCQs from retrieved content?

---

## 🚧 Technical Constraints & Requirements

✅ **AWS Services**
- Must use **Amazon Bedrock**:
  - API: **Converse, Guardrails, Embeddings, Agents** ([Boto3 Docs](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html))
  - **Amazon Nova Micro** for text generation ([Nova Micro](https://aws.amazon.com/ai/generative-ai/nova))
  - **Titan** for embeddings  

✅ **Implementation Stack**
- **Frontend:** [Streamlit](https://streamlit.io/)
- **Data Handling:** [pandas](https://pandas.pydata.org/) for visualization
- **Storage:** [SQLite](https://www.sqlite.org/) for vector storage
- **Transcripts:** YouTube transcripts as a knowledge source ([YouTubeTranscriptApi](https://pypi.org/project/youtube-transcript-api/))

✅ **Development Guidelines**
- Must demonstrate a **clear stepwise progression**:
  1. Base LLM
  2. Raw transcript processing
  3. Structured data extraction
  4. RAG-powered retrieval
  5. Interactive learning features
- Ensure **modular architecture** to aid teaching and reuse.
- **Error Handling:** Proper handling of **Spanish text processing issues**.
- Provide **clear visualizations** of the **RAG process**.
- Optimize costs by **staying within AWS free-tier limits** when possible.

---

## 📚 Knowledge Base

For vector database support, this project utilizes **ChromaDB**:  
🔗 [Chroma GitHub Repository](https://github.com/chroma-core/chroma)

---

## 🛠 Using the Transcript Downloader

The application includes a **command-line tool** for **downloading YouTube transcripts**.

### 📌 Options

| Option | Description |
|--------|-------------|
| **Required** | YouTube video URL (in quotes) |
| `--print` or `-p` | Print the transcript to the console after downloading |
| `--help` or `-h` | Display help information |

### 🚀 Usage Examples

**Download transcript only:**
```bash
python get_transcript.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

**Download and print transcript:**
```bash
python get_transcript.py "https://www.youtube.com/watch?v=VIDEO_ID" --print
```
**Get help:**
```bash
python get_transcript.py --help
```
📌 Output: The transcript will be saved in:
📁 backend/transcripts/VIDEO_ID.txt

## 📝 Notes
- The script defaults to retrieving Spanish (es) and English (en) transcripts.
- If transcripts are unavailable, it will display available language options.
- Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

---

## 📑 Structuring Transcripts

The tool includes a **script for structuring transcripts** into meaningful **lesson sections**.

### 📌 Options

| Option         | Description                                        |
|---------------|----------------------------------------------------|
| `--file` or `-f` | Process a specific transcript file              |
| `--all` or `-a`  | Process all transcripts in the directory       |
| `--help` or `-h` | Display help information                        |

### 🚀 Usage Examples

**Process a specific transcript:**
```bash
python structured_data.py --file VIDEO_ID.txt


**Process all transcripts:**
```bash
python structured_data.py --all
```

**Show help:**
```bash
python structured_data.py --help
```

### 📌 Output Format

The **structured data** is saved as JSON files in the same directory, with `_structured.json` appended to the filename:

📁 Example output file:  
```
backend/transcripts/COAtv5wS_tU_structured.json
```

### 📌 Data Structure

Each structured transcript consists of the following sections:

| Section        | Description                                         |
|---------------|-----------------------------------------------------|
| **Introduction** | Brief overview of the lesson or topic            |
| **Conversation** | The main dialogue or language content            |
| **Questions**   | Any exercises or multiple-choice questions        |

Each section **includes both** the **original Spanish text** and an **English analysis**.
