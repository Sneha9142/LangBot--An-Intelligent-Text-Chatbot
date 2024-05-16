from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import AzureOpenAI
import os
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

openai_endpoint = config.get('API', 'OPENAI_ENDPOINT')
openai_api_version = config.get('API', 'OPENAI_API_VERSION')
openai_api_key = config.get('API', 'OPENAI_API_KEY')
openai_deployment = config.get('API', 'DEPLOYMENT_NAME')

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = 'document'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_uploaded_file(file):
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print(filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return filename


def read_doc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents


def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc = text_splitter.split_documents(docs)
    return docs


# Initialize embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=openai_endpoint,
    openai_api_version=openai_api_version,
    openai_api_key=openai_api_key,
)

# Initialize AzureOpenAI LLM
llm = AzureOpenAI(
    azure_endpoint=openai_endpoint,
    openai_api_version=openai_api_version,
    openai_api_key=openai_api_key,
    azure_deployment=openai_deployment,
    temperature=0.2
)

os.environ["PINECONE_API_KEY"] = "e50a90fe-1e33-4cc4-9033-63c0753e86de"
index_name = "bookpdf"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
docs = text_splitter.split_documents(read_doc('document/'))
docsearch = PineconeVectorStore.from_documents(index_name=index_name,
                                               documents=docs,
                                               embedding=embeddings,
                                               )


def retrieve_answers(query, doc_data):
    if doc_data:
        doc_search = retrieve_query(query, doc_data)
    else:
        doc_search = retrieve_query(query)
    chain = load_qa_chain(llm, chain_type="stuff")
    input_data = {
        "input_documents": doc_search,
        "question": query,
    }
    response = chain.invoke(input=input_data)
    return response['output_text']


def retrieve_query(query, doc_data=None, k=2):
    if doc_data:
        matching_results = docsearch.similarity_search(query, k=k, documents=doc_data)
    else:
        matching_results = docsearch.similarity_search(query, k=k)
    return matching_results


@app.route('/', methods=['GET'])
def landing_page():
    return render_template('index.html')


@app.route('/chat_with_pdf', methods=['GET', 'POST'])
def chat_with_pdf():
    if request.method == 'POST':

        if 'file' in request.files:
            uploaded_file = request.files['file']
            print(uploaded_file)
            if uploaded_file.filename != '':
                filename = save_uploaded_file(uploaded_file)

                uploaded_doc_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                uploaded_doc = read_doc(uploaded_doc_path)

                chunked_docs = chunk_data(uploaded_doc)

                user_query = request.form['query']
                answer = retrieve_answers(user_query, chunked_docs)
                return render_template('ChatWithPDF.html', query=user_query, answer=answer)

        user_query = request.form['query']
        answer = retrieve_answers(user_query, docs)
        return render_template('ChatWithPDF.html', query=user_query, answer=answer)

    return render_template('ChatWithPDF.html')


@app.route('/ask_me_anything', methods=['GET', 'POST'])
def ask_me_anything():
    if request.method == 'POST':
        query = request.form['query']
        answer = llm.generate([query]).generations[0][0].text
        if query != "":
            return render_template('AskMeAnything.html', answer=answer)
        else:
            return render_template('AskMeAnything.html', answer="Please ask question....")

    return render_template('AskMeAnything.html')


if __name__ == "__main__":
    app.run(debug=True)
