Chat with PDF Flask Application
Overview
This Flask application allows users to upload PDF documents and ask questions related to the content of those PDFs. It uses Azure OpenAI and Pinecone for embeddings and vector search to retrieve relevant information from the uploaded documents.

Features
Upload PDF files and query their contents.
Real-time question answering using Azure OpenAI.
Document chunking and embedding with Pinecone for efficient search and retrieval.
Prerequisites
Python 3.8+
Azure OpenAI API key and endpoint
Pinecone API key
Flask
Setup
Step 1: Clone the Repository
bash
Copy code
git clone https://github.com/your-username/chat-with-pdf.git
cd chat-with-pdf
Step 2: Create and Activate Virtual Environment
bash
Copy code
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Step 3: Install Dependencies
bash
Copy code
pip install -r requirements.txt
Step 4: Set Up Configuration
Create a config.ini file in the root directory of the project with the following content:

ini
Copy code
[API]
OPENAI_ENDPOINT = your_openai_endpoint
OPENAI_API_VERSION = your_openai_api_version
OPENAI_API_KEY = your_openai_api_key
DEPLOYMENT_NAME = your_deployment_name
Set the Pinecone API key as an environment variable:

bash
Copy code
export PINECONE_API_KEY=your_pinecone_api_key
Step 5: Run the Application
bash
Copy code
python app.py
The application will be accessible at http://127.0.0.1:5000/.

Usage
Landing Page
Navigate to the landing page by accessing http://127.0.0.1:5000/. From here, you can choose to interact with the PDF chat or ask general questions.

Chat with PDF
Go to http://127.0.0.1:5000/chat_with_pdf.
Upload a PDF file.
Enter your query related to the uploaded PDF.
Receive the answer based on the content of the PDF.
Ask Me Anything
Go to http://127.0.0.1:5000/ask_me_anything.
Enter any query you have.
Receive an answer from the Azure OpenAI model.
Project Structure
app.py: Main Flask application file.
config.ini: Configuration file for API keys and endpoints.
templates/: Folder containing HTML templates.
index.html: Landing page template.
ChatWithPDF.html: Template for the PDF chat interface.
AskMeAnything.html: Template for the general question interface.
document/: Folder to store uploaded PDF files.
