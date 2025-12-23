from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import os
import json
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='static')
CORS(app)

# Global variables for RAG components
vectorstore = None
chat_histories = {}  # Store chat history per session


def load_and_process_documents(file_path: str):
    """Load documents and split into chunks"""
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from the document")
    return chunks


def create_vectorstore(chunks):
    """Create FAISS vectorstore with OpenAI embeddings"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    print("Vectorstore created successfully")
    return vectorstore


def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)


def get_chat_history(session_id: str):
    """Get or create chat history for a session"""
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    return chat_histories[session_id]


def add_to_history(session_id: str, human_msg: str, ai_msg: str):
    """Add messages to chat history (buffer memory)"""
    history = get_chat_history(session_id)
    history.append(HumanMessage(content=human_msg))
    history.append(AIMessage(content=ai_msg))
    # Keep only last 10 exchanges (20 messages) as buffer
    if len(history) > 20:
        chat_histories[session_id] = history[-20:]


def get_rag_response(question: str, session_id: str):
    """Get response from RAG pipeline with memory"""

    # Get retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Get relevant documents
    docs = retriever.invoke(question)
    context = format_docs(docs)

    # Get chat history
    history = get_chat_history(session_id)

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are BizWiz, an AI assistant for TechNova Solutions Pvt Ltd.
You help users with information about the company, its services, products, contact details, and general inquiries.
Be helpful, professional, and friendly. If you don't know the answer based on the context, say so politely.

Context from company knowledge base:
{context}

Use the above context to answer the user's question. If the question is not related to the context,
try to help generally but mention that you specialize in TechNova Solutions information."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    # Create LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # Create chain
    chain = prompt | llm | StrOutputParser()

    # Get response
    response = chain.invoke({
        "context": context,
        "chat_history": history,
        "question": question
    })

    # Add to history (buffer memory)
    add_to_history(session_id, question, response)

    return response, docs


def stream_rag_response(question: str, session_id: str):
    """Stream response from RAG pipeline with memory"""

    # Get retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Get relevant documents
    docs = retriever.invoke(question)
    context = format_docs(docs)

    # Get chat history
    history = get_chat_history(session_id)

    # Create prompt template (same as non-streaming)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are BizWiz, an AI assistant for TechNova Solutions Pvt Ltd.
You help users with information about the company, its services, products, contact details, and general inquiries.
Be helpful, professional, and friendly. If you don't know the answer based on the context, say so politely.

Context from company knowledge base:
{context}

Use the above context to answer the user's question. If the question is not related to the context,
try to help generally but mention that you specialize in TechNova Solutions information."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    # Create LLM with streaming enabled
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, streaming=True)

    # Create chain
    chain = prompt | llm | StrOutputParser()

    # Collect full response for history
    full_response = ""

    # Stream the response
    for chunk in chain.stream({
        "context": context,
        "chat_history": history,
        "question": question
    }):
        full_response += chunk
        yield chunk

    # Add to history after streaming completes
    add_to_history(session_id, question, full_response)


def initialize_rag():
    """Initialize the RAG pipeline"""
    global vectorstore

    data_file = os.path.join(os.path.dirname(__file__), 'Custom_data.txt')

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    chunks = load_and_process_documents(data_file)
    vectorstore = create_vectorstore(chunks)
    print("RAG pipeline initialized successfully!")


# Routes
@app.route('/')
def index():
    """Serve the main page"""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')

        if not user_message:
            return jsonify({'success': False, 'error': 'Message is required'}), 400

        # Get RAG response
        response, source_docs = get_rag_response(user_message, session_id)

        # Extract sources
        sources = []
        for doc in source_docs:
            content = doc.page_content
            sources.append({
                'content': content[:200] + '...' if len(content) > 200 else content
            })

        return jsonify({
            'success': True,
            'response': response,
            'sources': sources
        })

    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """Handle streaming chat requests"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')

        if not user_message:
            return jsonify({'success': False, 'error': 'Message is required'}), 400

        def generate():
            try:
                for chunk in stream_rag_response(user_message, session_id):
                    # Send each chunk as SSE data
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                # Send done signal
                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'
            }
        )

    except Exception as e:
        print(f"Error in stream endpoint: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    """Clear conversation history for a session"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')

        if session_id in chat_histories:
            del chat_histories[session_id]

        return jsonify({'success': True, 'message': 'Conversation history cleared'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'vectorstore_initialized': vectorstore is not None
    })


if __name__ == '__main__':
    print("Initializing RAG pipeline...")
    initialize_rag()
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
