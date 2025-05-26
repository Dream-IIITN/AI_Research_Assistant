# 🚀 AI-Powered Research Navigator  

An AI-driven research assistant that retrieves, filters, and analyzes academic papers, web content, and knowledge sources to generate structured research reports.

 

---

## 📌 **Features**
✅ Natural language research queries  
✅ Retrieval-Augmented Generation (RAG) for relevant information  
✅ Web search and Arxiv paper fetching  
✅ AI-powered summarization & report generation  
✅ Dynamic tool invocation & execution  

---

## 🛠 **Tech Stack**
| Category         | Technologies & Tools |
|-----------------|---------------------|
| **Programming Language** | Python |
| **Frameworks & Libraries** | LangChain, LangGraph |
| **LLM & AI Models** | Ollama Custom LLM |
| **Database / Vector Store** | ChromaDB (if used) |
| **APIs & Tools** | Arxiv API, Web Search API |
| **Backend Architecture** | FastAPI / Flask (if applicable) |
| **Dependency Management** | pip / conda |
| **Version Control & Deployment** | GitHub, Docker (if applicable) |

---
## Output demo 
complee video input on given topic
![Final Video Ouput][https://drive.google.com/file/d/19zFeyE8Dt-AL1In6ENDQfUGUB0t1vjU3/view?usp=sharing)

## 📸 **Screenshots**
_Add screenshots of the UI/CLI results here._

1. **Research Query Input:**  
Input text prompt on the topic you want to work on!
   ![Step 1](/images/step_1.png)  
Output will contain 5 semantically related Research Papers
   ![Sol 1](/images/1_sol.png) 

2. **Tool Execution & RAG Processing:**  
Now select the Research Paper from above 5 and input its ID to process it.
   ![Step 2](/images/step2.png) 
The Research Paper will be downloaded and stored in chunks and processed. 
   ![Sol 2](/images/2_sol.png)  

3. **Generated Research Report:**  
Input a final Text prompt for your Report Work based on the Research Paper Selected
   ![Step 3](/images/step3.png)  
Here is the final Output with Introduction, Abstract & Methodology
   ![Sol 3](/images/3_final.png)

---

## 🚀 **Installation & Setup**
### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### Create a Virtual Environment (Optional)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
4️⃣ Run the Application
```bash
python main.py  # Or the entry script of your project
```
