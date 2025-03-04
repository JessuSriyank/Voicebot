import os
import uuid
import logging
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOpenAI
import redis
import json
from langdetect import detect
from deep_translator import GoogleTranslator
from langchain_mistralai import ChatMistralAI
import chromadb
import shutil
import tiktoken
import requests
 
import re
def translate_text(text, source_language, target_language):
    """
    Translate text from source_language to target_language using deep-translator.
    """
    return GoogleTranslator(source=source_language, target=target_language).translate(text)
    
# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize Redis client
redis_client = redis.StrictRedis(host="localhost", port=6379,db=1, decode_responses=True)

# Initialize Hugging Face Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize ChatGroq LLM with your Groq API key
groq_api_key = "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"  # Replace with your actual API key
groq_api_key_1="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"
llm2 = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant",max_completion_tokens=1024)
#llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key="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")
#llm2 = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key="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")

llm1 = ChatOpenAI(model="gpt-4", openai_api_key="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")
#llm = ChatMistralAI(
#    model="ministral-8b-latest",
#    temperature=0,
#    max_retries=2,
#    api_key="jmRg4VTOsZgOx3djz4KMjeRpj3NiYc3J"
#)
#llm2 = ChatMistralAI(
#    model="ministral-8b-latest",
#    temperature=0,
#    max_retries=2,
#    api_key="BvkaObyTPN5hawbD0fhhXlCYLS2AJeK2"
#)
# Initialize FastAPI app
app = FastAPI()
vector_store = None 

vector_store_path = "/home/dtel/Voicebot_new/chroma_storage"
upload_base_path = "user_uploads"    # Base path for raw PDF storage
public_upload_path = "/var/www/html/uploads/"  # Publicly accessible path
server_url = "https://convoxai.deepijatel.ai/uploads"  # Base URL for files

# Set permissions (Only needs to be run once, can be removed later)
os.chmod(upload_base_path, 0o777)
os.chmod(public_upload_path, 0o777)
os.chmod(vector_store_path, 0o777)

client = chromadb.PersistentClient()
# Paths for persistence
logging.basicConfig(level=logging.INFO)
# Function to detect language
def detect_language(text):
    return detect(text)

def get_session_language(session_id):
    """
    Get the session language from Redis.
    """
    return redis_client.hget(f"{session_id}:metadata", "current_language")

def set_session_language(session_id, language):
    redis_client.hset(f"{session_id}:metadata", "current_language", language)  # Set initial session language



# Preload vector store (modified)
def preload_vector_store():
    global vector_store
    if os.path.exists(vector_store_path):
        try:
            vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)
            logging.info("Chroma vector store loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading Chroma vector store: {e}")
    else:
        logging.warning("Chroma vector store not found. Upload a document to initialize.")

# Load persisted data on startup
preload_vector_store()

# Redis Utility Functions
def save_conversation_to_redis(conversation_id, history):
    """Save conversation history to Redis."""
    redis_client.set(conversation_id, history)

def load_conversation_from_redis(conversation_id):
    """Load conversation history from Redis."""
    history = redis_client.get(conversation_id)
    return history if history else []

@app.get("/")
async def root():
    """
    Root endpoint to verify the API is running.
    """
    return {"message": "Hello World"}

def get_user_vector_store(user_id: str):
    """Retrieve the user's vector store from Chroma."""
    user_collection_path = os.path.join(vector_store_path, user_id)

    # Ensure the user directory exists
    if not os.path.exists(user_collection_path):
        logging.error(f"User collection directory not found: {user_collection_path}")
        return None

    # Check for Chroma database file (supports both .sqlite and .sqlite3)
    db_file_sqlite3 = os.path.join(user_collection_path, "chroma.sqlite3")
    db_file_sqlite = os.path.join(user_collection_path, "chroma.sqlite")

    if not os.path.exists(db_file_sqlite3) and not os.path.exists(db_file_sqlite):
        logging.error(f"Chroma database file missing for user: {user_id}")
        return None

    # Load the Chroma vector store
    return Chroma(persist_directory=user_collection_path, embedding_function=embeddings)



@app.post("/AI/v6/upload_file/")
async def upload_file1(file: UploadFile = File(...), user_id: str = Form(...)):
    """
    Upload and preprocess a file (e.g., PDF) and save it into a user-specific Chroma vector store collection.
    Also, move the file to a public directory (/var/www/html/uploads/) for UI display.
    """
    try:
        # Define user-specific directories
        user_upload_dir = os.path.join(upload_base_path, user_id)
        public_upload_dir = os.path.join(public_upload_path, user_id)
        user_collection_path = os.path.join(vector_store_path, user_id)

        # Ensure user directories exist
        os.makedirs(user_upload_dir, exist_ok=True)
        os.makedirs(user_collection_path, exist_ok=True)
        os.makedirs(public_upload_dir, exist_ok=True)

        # Set proper permissions
        os.chmod(user_upload_dir, 0o777)
        os.chmod(user_collection_path, 0o777)
        os.chmod(public_upload_dir, 0o777)

        # Save new file to user's directory
        user_file_path = os.path.join(user_upload_dir, file.filename)

        # Prevent overwriting existing files
        if os.path.exists(user_file_path):
            logging.warning(f"File {file.filename} already exists, overwriting...")

        with open(user_file_path, "wb") as f:
            f.write(await file.read())

        # Move file to the public directory
        public_file_path = os.path.join(public_upload_dir, file.filename)
        shutil.copy(user_file_path, public_file_path)

        # Process the file using PyPDFLoader
        try:
            loader = PyPDFLoader(user_file_path)
            docs = loader.load()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading PDF: {e}")

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_documents = text_splitter.split_documents(docs)

        # Check if Chroma database is writable
        chroma_db_path = os.path.join(user_collection_path, "chroma.sqlite3")
        if os.path.exists(chroma_db_path) and not os.access(chroma_db_path, os.W_OK):
            raise HTTPException(status_code=500, detail="Chroma database is read-only. Check permissions.")

        # Create or update the Chroma vector store
        try:
            user_vector_store = Chroma.from_documents(split_documents, embeddings, persist_directory=user_collection_path)
            user_vector_store.persist()
            logging.info(f"Chroma vector store created successfully for user: {user_id}")
        except Exception as e:
            logging.error("Error writing to Chroma vector store. Possible read-only database.")
            raise HTTPException(status_code=500, detail=f"Chroma database error: {e}")

        # Construct file URL
        file_url = f"{server_url}{user_id}/{file.filename}"

        return {"message": f"File '{file.filename}' uploaded by {user_id} and processed successfully.",
                "file_url": file_url}

    except Exception as e:
        logging.error("Error uploading file: %s", e)
        raise HTTPException(status_code=400, detail=f"Error uploading or processing file: {e}")




@app.delete("/AI/v6/delete_user_data/")
async def delete_user_data(user_id: str = Form(...)):
    """
    Delete all uploaded files and the Chroma vector store for a user.
    """
    try:
        # Define user-specific paths
        user_vector_store = get_user_vector_store(user_id)
        all_ids = user_vector_store.get(include=[])["ids"]  # Fetch all IDs
        print("All stored UUIDs:", all_ids) 
        user_vector_store.delete_collection()  # Deletes all entries in the collection
        user_upload_dir = os.path.join(upload_base_path, user_id)
        public_upload_dir = os.path.join(public_upload_path, user_id)
        user_collection_path = os.path.join(vector_store_path, user_id)
        
        print("Deleted all records from vector store.")
        # Delete uploaded files
        if os.path.exists(user_upload_dir):
            shutil.rmtree(user_upload_dir)
            logging.info(f"Deleted uploaded files for user: {user_id}")
        else:
            logging.warning(f"Upload directory not found for user: {user_id}")

        if os.path.exists(public_upload_dir):
            shutil.rmtree(public_upload_dir)
            logging.info(f"Deleted public files for user: {user_id}")
        else:
            logging.warning(f"Public directory not found for user: {user_id}")

        # Delete Chroma vector store
        if os.path.exists(user_collection_path):
            for item in os.listdir(user_collection_path):
                item_path = os.path.join(user_collection_path, item)
                if os.path.basename(item_path) != "chroma.sqlite3":
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
            logging.info(f"Deleted Chroma vector store for user: {user_id}")
        else:
           logging.warning(f"Chroma vector store not found for user: {user_id}")

        return {"message": f"All data for user '{user_id}' deleted successfully."}

    except Exception as e:
        logging.error(f"Error deleting user data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting user data: {e}")

@app.get("/AI/v6/get_user_files/")
async def get_user_files(user_id: str = Form(...)):
    """
    Fetch all uploaded files for a given user ID.
    """
    try:
        user_upload_dir = os.path.join(upload_base_path, user_id)

        # Check if user directory exists
        if not os.path.exists(user_upload_dir):
            raise HTTPException(status_code=404, detail="User ID not found or no files uploaded.")

        # List all files in the user's upload directory
        files = os.listdir(user_upload_dir)

        if not files:
            return {"message": f"No files found for user {user_id}"}
        
        file_list = [
            {
                "file_name": file,
                "file_url": f"{server_url}/{user_id}/{file}"
            } for file in files
        ]

        return {"user_id": user_id, "uploaded_files": file_list}

    except Exception as e:
        logging.error(f"Error fetching files for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching files: {str(e)}")

@app.post("/AI/v5/start_conversation/")
async def start_conversation():
    """
    Endpoint to generate a unique conversation ID.
    """
    unique_id = str(uuid.uuid4())  # Generate a unique conversation ID
    redis_client.set(unique_id, "[]")
    language="en"
    redis_client.hset(f"{unique_id}:metadata", "current_language", language)  # Set initial session language
  # Initialize an empty conversation history in Redis
    return {"conversation_id": unique_id, "status": "Success"}

def create_start_conversation(unique_id):
    """
    Endpoint to generate a unique conversation ID.
    """
    # Check if the key exists
    if redis_client.exists(unique_id):
        # If the key exists, update the metadata
        redis_client.hset(f"{unique_id}:metadata", "current_language", "en")
    else:
        # If the key does not exist, create a new entry
        redis_client.set(unique_id, "[]")
        redis_client.hset(f"{unique_id}:metadata", "current_language", "en")
    
    return unique_id

@app.post("/AI/v6/query/")
async def voicebot6(query: str = Form(...), conversation_id: str = Form(...), user_id: str = Form(...)):
    """
    Query endpoint to predict intent and process actions based on the intent.
    """
    #global vector_store

    try:
        # Retrieve user-specific vector store
        user_vector_store = get_user_vector_store(user_id)

        if user_vector_store is None:
            return {
                "query": query,
                "answer": "Sorry, no data found. Please upload documents first.",
                "status": "No Data"
            }

        # Retrieve relevant documents
        retriever = user_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        documents_with_context = retriever.get_relevant_documents(query)

        if not documents_with_context:
            logging.warning("No relevant documents found for query: %s", query)
            return {
                "query": query,
                "predicted_intent": "general",
                "intent_explanation": "No documents found; defaulting to general intent.",
                "action_response": "Sorry, I couldn't find any information. Please ask queries related to Convox.",
                "callend": "0",
                "calltransfer": "0",
                "status": "Success",
            }

        context_text = "\n".join([doc.page_content for doc in documents_with_context[:5]])

        # Load conversation history from Redis
        conversation_id=create_start_conversation(conversation_id)
        conversation_history = redis_client.get(conversation_id)
        if conversation_history:
            conversation_history = json.loads(conversation_history)
        else:
            conversation_history = []

        # Intent Prediction Chain
        intent_prompt_template = PromptTemplate(
            input_variables=["context", "input"],
           template=(
                "You are an intelligent assistant analyzing user queries to classify them into intents."
                "\n**Objective**: Accurately classify the user's query as 'calltransfer', 'general', or 'endconversation'."
                "\n\n**Examples:**"

                "\n<context>\nCustomer interacting with a support bot.\n</context>"
                "\nQuery: 'Can you transfer me to an agent?'"
                "\nIntent: calltransfer - User requests a call transfer."

                "\n<context>\nCustomer asking about services.\n</context>"
                "\nQuery: 'What services does Convox offer?'"
                "\nIntent: general - User asks a general question."

                "\n<context>\nCustomer ending the chat.\n</context>"
                "\nQuery: 'Thanks for your help. Goodbye!'"
                "\nIntent: endconversation - User indicates they want to end the conversation."

                "\n<context>\nCustomer having trouble with billing.\n</context>"
                "\nQuery: 'I have an issue with my billing, can I speak to someone?'"
                "\nIntent: calltransfer - User requests assistance involving a transfer."

                "\n<context>\nCustomer curious about support hours.\n</context>"
                "\nQuery: 'What are your customer support hours?'"
                "\nIntent: general - User asks about operational information."

                "\n<context>\nCustomer expresses gratitude and closes chat.\n</context>"
                "\nQuery: 'That answers my question. Thank you and goodbye!'"
                "\nIntent: endconversation - User concludes the interaction."

                "\n<context>\nCustomer asks about payment options.\n</context>"
                "\nQuery: 'What payment methods do you accept?'"
                "\nIntent: general - User asks for general information about payments."

                "\n<context>\nCustomer requests direct assistance.\n</context>"
                "\nQuery: 'I need help right now. Can you connect me to someone?'"
                "\nIntent: calltransfer - User requests immediate human assistance."

                "\n<context>\nCustomer concludes the interaction politely.\n</context>"
                "\nQuery: 'That's all I needed. Have a great day!'"
                "\nIntent: endconversation - User politely ends the conversation."

                "\n<context>\nCustomer interacting with a support bot.\n</context>"
                "\nQuery: 'Can you transfer me to an agent?'"
                "\nIntent: calltransfer - User requests a call transfer."

                "\n<context>\nCustomer requesting direct transfer.\n</context>"
                "\nQuery: 'Transfer to Neeraj please.'"
                "\nIntent: calltransfer - User requests direct transfer to Neeraj."

                "\n<context>\nCustomer specifying transfer to another individual.\n</context>"
                "\nQuery: 'Can you transfer the call to Rahul?'"
                "\nIntent: calltransfer - User requests transfer to Rahul."

                "\n<context>\nCustomer asking for support agent.\n</context>"
                "\nQuery: 'Connect me to an agent now.'"
                "\nIntent: calltransfer - User requests immediate human assistance."


                "\nNow classify:"
                "\n<context>\n{context}\n</context>\nQuery: {input}\nIntent:"
            ),
        )

        intent_chain = LLMChain(llm=llm2, prompt=intent_prompt_template)
        intent_response = intent_chain.run(input=query, context=context_text)

        # Extract intent and explanation
        intent_lines = intent_response.split("\n")
        predicted_intent = re.sub(r"[\\_/@]", "", intent_lines[0].split("-")[0].strip().lower())
        explanation = "-".join(intent_lines[0].split("-")[1:]).strip() if "-" in intent_lines[0] else "No explanation provided."

        # Validate intent
        if predicted_intent.startswith("intent:"):
            result = re.search(r"intent:\s*(\w+)",predicted_intent)
            predicted_intent = result.group(1) if result else None
        print("Padma",predicted_intent)
        valid_intents = {"calltransfer", "general", "endconversation"}
        if predicted_intent not in valid_intents:
            logging.warning("Unexpected intent predicted: %s", predicted_intent)
            predicted_intent = "general"
            explanation = "Fallback to general intent due to unexpected response."

        # Default flags
        callend = "0"
        calltransfer = "0"
        action_response = ""

        # Default flag for call status
        callstatus = "active"  # Possible values: "active", "transfer", "end_conversation"
        contact_number=""
        #sample CHANGE
        # Handle intent-based actions
        if predicted_intent == "calltransfer":
            # Check if the query directly specifies the transfer target
            match = re.search(r"(?:transfer to|connect with|connect me to|please connect me with|with)\s+(\w+)", query.lower()) or re.search(r"(?:transfer to|connect with|connect me to|please connect me with|with)\s+(\w+)", explanation.lower())

            if match:
                contact_name = match.group(1).capitalize()

                # Search the text file for the contact's number
                try:
                    with open("contacts.txt", "r") as file:
                        contacts = {line.split(":")[0].strip(): line.split(":")[1].strip() for line in file}
                        if contact_name in contacts:
                            contact_number = contacts[contact_name]

                            # Mock transfer logic
                            logging.info(f"Transferring call to {contact_name} at {contact_number}")
                            #action_response = f"Transferring your call to {contact_name} now."
                            action_response = (
                                f"Please wait while we transfer your call."
                            )
                            calltransfer = "1"
                            callstatus = "transfer"
                        else:
                            calltransfer = "1"
                            callstatus = "transfer"
                            action_response = (
                                f"Please wait while we transfer your call."
                            )
                except Exception as file_error:
                    logging.error("Error reading contacts file: %s", file_error)
                    action_response = "An error occurred while looking up the contact information."
            else:
                callstatus = "active"
                redis_client.set(conversation_id + "_state", "awaiting_transfer_confirmation")
                redis_client.set(conversation_id + "_transfer_query", query)
                calltransfer = "1"
                callstatus = "transfer"
                action_response = (
                                f"Please wait while we transfer your call."
                            )
                #action_response = (
                #    "To whom should I transfer the call? You can say 'transfer to Neeraj' or specify a name."
                #)

        # Handle transfer confirmation
        elif redis_client.get(conversation_id + "_state") == "awaiting_transfer_confirmation":
            redis_client.delete(conversation_id + "_state")  # Clear intermediate state
            transfer_query = redis_client.get(conversation_id + "_transfer_query")

            # Extract the name from the user's response
            match = re.search(r"(?:transfer to|connect with|connect me to|please connect me with|with)\s+(\w+)", query.lower()) or re.search(r"(?:transfer to|connect with|connect me to|please connect me with|with)\s+(\w+)", explanation.lower())

            if match:
                contact_name = match.group(1).capitalize()

                # Search the text file for the contact's number
                try:
                    with open("contacts.txt", "r") as file:
                        contacts = {line.split(":")[0].strip(): line.split(":")[1].strip() for line in file}
                        if contact_name in contacts:
                            contact_number = contacts[contact_name]

                            # Mock transfer logic
                            logging.info(f"Transferring call to {contact_name} at {contact_number}")
                            action_response = (
                                f"Please wait while we transfer your call."
                            )
                            #action_response = f"Transferring your call to {contact_name} now."
                            calltransfer = "1"
                            callstatus = "transfer"
                        else:
                            calltransfer = "1"
                            callstatus = "transfer"
                            action_response = (
                                f"Please wait while we transfer your call."
                            )
                except Exception as file_error:
                    logging.error("Error reading contacts file: %s", file_error)
                    action_response = "An error occurred while looking up the contact information."
            else:
                action_response = (f"Please wait while we transfer your call.")
                calltransfer = "1"
                callstatus = "transfer"
                #action_response = (
                #    "I couldn't understand your request. Please specify the name clearly, like 'transfer to Neeraj'."
                #)

        elif predicted_intent == "endconversation":
            callstatus = "endconversation"
            action_response = "Thank you for reaching out. Have a great day!"
        elif predicted_intent == "general":
            conversation_history_text = "\n".join(conversation_history)

            # Chain 2: Process Action for General Intent
            action_prompt_template = PromptTemplate(
                input_variables=["context", "input", "conversation_history"],
                template=(
                    "You are a voice bot assistant providing concise, accurate answers based on vector store data."
                    " **Guidelines**:"
                    " 1. Retrieve relevant information and respond clearly in 1 sentence."
                    " 2. Use a professional, conversational tone."
                    " 3. If no data is found, politely say: *Sorry, I couldn't find any information. Please ask queries related to Convox.*"
                    " 4. Stick to vector store content; avoid unsupported details."
                    " 5. Replace ConVox with convox in response."
                    "**Objective**: Deliver precise, voice-optimized responses for user queries."
                    "\n<context>\n{context}\n</context>\n\n"
                    "\n<conversation_history>\n{conversation_history}\n</conversation_history>\n\n"
                    "Question: {input}\nAnswer:"
                ),
            )

            action_chain = LLMChain(llm=llm, prompt=action_prompt_template)
            action_response = action_chain.run(input=query, context=context_text, conversation_history=conversation_history_text)

            # Fallback response if action chain fails
            if not action_response.strip():
                action_response = "Sorry, I couldn't find any information. Please ask queries related to Convox."

        # Update conversation history in Redis
        conversation_history.append(f"Q: {query}\nA: {action_response}")
        redis_client.set(conversation_id, json.dumps(conversation_history))

        # Prepare final API response
        api_response = {
            "query": query,
            "predicted_intent": predicted_intent,
            "intent_explanation": explanation,
            "action_response": action_response,
            "callstatus": callstatus,
            "contact_number":contact_number,
            "status": "Success",
        }

        logging.info(
            "Processed Query: %s | Intent: %s | Explanation: %s | Action Response: %s | Call End: %s | Call Transfer: %s",
            query,
            predicted_intent,
            explanation,
            action_response,
            callend,
            calltransfer,
        )
        return api_response

    except Exception as e:
        logging.error("Error processing query: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

@app.post("/AI/v6/add-contact/")
async def add_contact(name: str = Form(...), phone_number: str = Form(...)):
    """
    Add a new contact to the contacts.txt file.
    """
    try:
        if not name or not phone_number:
            raise HTTPException(status_code=400, detail="Name and phone number are required.")

        # Append the new contact to the file
        with open("contacts.txt", "a") as file:
            file.write(f"{name}:{phone_number}\n")

        return {"status": "Success", "message": f"Contact {name} added successfully."}

    except Exception as e:
        logging.error("Error adding contact: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.post("/AI/v6/hin-query/")
async def hin_voicebot(query: str = Form(...), conversation_id: str = Form(...), user_id: str = Form(...)):
    """
    Query endpoint to predict intent and process actions based on the intent.
    """
    try:
       # Retrieve user-specific vector store
        user_vector_store = get_user_vector_store(user_id)

        if user_vector_store is None:
            return {
                "query": query,
                "answer": "Sorry, no data found. Please upload documents first.",
                "status": "No Data"
            }

        # Retrieve relevant documents
        retriever = user_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        documents_with_context = retriever.get_relevant_documents(query)

        if not documents_with_context:
            logging.warning("No relevant documents found for query: %s", query)
            return {
                "query": query,
                "predicted_intent": "general",
                "intent_explanation": "No documents found; defaulting to general intent.",
                "action_response": "Sorry, I couldn't find any information. Please ask queries related to Convox.",
                "callend": "0",
                "calltransfer": "0",
                "status": "Success",
            }

        context_text = "\n".join([doc.page_content for doc in documents_with_context[:5]])

        # Load conversation history from Redis
        conversation_id=create_start_conversation(conversation_id)
        conversation_history = redis_client.get(conversation_id)
        if conversation_history:
            conversation_history = json.loads(conversation_history)
        else:
            conversation_history = []

        # Intent Prediction Chain
        intent_prompt_template = PromptTemplate(
            input_variables=["context", "input"],
            template = (
                "You are an intelligent assistant analyzing user queries to classify them into intents."
                "\n**Objective**: Accurately classify the user's query as 'calltransfer', 'general', or 'endconversation'."
                "\n\n**Instructions:**"
                "\n1. यदि उपयोगकर्ता कॉल ट्रांसफर करने का अनुरोध करता है (जैसे, 'कॉल ट्रांसफर कीजिए', 'मुझे एक एजेंट से जोड़ें', 'क्या आप कॉल ट्रांसफर कर सकते हैं?'), तो इसे 'calltransfer' के रूप में वर्गीकृत करें।"
                "\n2. यदि प्रश्न सामान्य जानकारी या कॉल ट्रांसफर से असंबंधित है (जैसे, 'आपके समर्थन घंटे क्या हैं?', 'आप कौन से भुगतान विकल्प स्वीकार करते हैं?'), तो इसे 'general' के रूप में वर्गीकृत करें।"
                "\n3. यदि उपयोगकर्ता आभार व्यक्त करता है, विदाई कहता है, या बातचीत समाप्त करने का संकेत देता है (जैसे, 'धन्यवाद', 'अलविदा'), तो इसे 'endconversation' के रूप में वर्गीकृत करें।"
                "\n4. यदि कोई अप्रत्याशित प्रतिक्रिया प्राप्त होती है (जैसे, अस्पष्ट आशय, एपीआई से अप्रासंगिक प्रतिक्रिया), तो इसे 'general' के रूप में वर्गीकृत करें और एक फॉलबैक प्रतिक्रिया लौटाएं।"
                "\n\n**Examples:**"

                "\n<context>\nग्राहक एक समर्थन बॉट के साथ बातचीत कर रहा है।\n</context>"
                "\nQuery: 'क्या आप मुझे एक एजेंट से जोड़ सकते हैं?'"
                "\nIntent: calltransfer - उपयोगकर्ता कॉल ट्रांसफर करने का अनुरोध कर रहा है।"

                "\n<context>\nग्राहक सेवाओं के बारे में पूछ रहा है।\n</context>"
                "\nQuery: 'Convox किन सेवाओं की पेशकश करता है?'"
                "\nIntent: general - उपयोगकर्ता एक सामान्य प्रश्न पूछ रहा है।"

                "\n<context>\nग्राहक बातचीत समाप्त कर रहा है।\n</context>"
                "\nQuery: 'आपकी सहायता के लिए धन्यवाद। अलविदा!'"
                "\nIntent: endconversation - उपयोगकर्ता बातचीत समाप्त करना चाहता है।"

                "\n<context>\nग्राहक बिलिंग में समस्या का सामना कर रहा है।\n</context>"
                "\nQuery: 'मुझे अपने बिलिंग में समस्या है, क्या मैं किसी से बात कर सकता हूँ?'"
                "\nIntent: calltransfer - उपयोगकर्ता सहायता मांग रहा है जिसमें ट्रांसफर शामिल है।"

                "\n<context>\nग्राहक सहायता घंटे के बारे में पूछ रहा है।\n</context>"
                "\nQuery: 'आपके ग्राहक सहायता घंटे क्या हैं?'"
                "\nIntent: general - उपयोगकर्ता संचालन संबंधी जानकारी मांग रहा है।"

                "\n<context>\nग्राहक आभार व्यक्त करता है और बातचीत समाप्त करता है।\n</context>"
                "\nQuery: 'इससे मेरा सवाल हल हो गया। धन्यवाद और अलविदा!'"
                "\nIntent: endconversation - उपयोगकर्ता बातचीत समाप्त कर रहा है।"

                "\n<context>\nग्राहक भुगतान विकल्पों के बारे में पूछ रहा है।\n</context>"
                "\nQuery: 'आप कौन से भुगतान तरीके स्वीकार करते हैं?'"
                "\nIntent: general - उपयोगकर्ता भुगतान विकल्पों की जानकारी मांग रहा है।"

                "\n<context>\nग्राहक तत्काल सहायता मांग रहा है।\n</context>"
                "\nQuery: 'मुझे अभी मदद की जरूरत है। क्या आप मुझे किसी से जोड़ सकते हैं?'"
                "\nIntent: calltransfer - उपयोगकर्ता तत्काल मानव सहायता मांग रहा है।"

                "\n<context>\nग्राहक विनम्रता से बातचीत समाप्त करता है।\n</context>"
                "\nQuery: 'यही सब चाहिए था। आपका दिन शुभ हो!'"
                "\nIntent: endconversation - उपयोगकर्ता विनम्रता से बातचीत समाप्त कर रहा है।"

                "\n<context>\nग्राहक एक समर्थन बॉट के साथ बातचीत कर रहा है।\n</context>"
                "\nQuery: 'क्या आप मुझे एक एजेंट से जोड़ सकते हैं?'"
                "\nIntent: calltransfer - उपयोगकर्ता कॉल ट्रांसफर करने का अनुरोध कर रहा है।"

                "\n<context>\nग्राहक ने एक विशेष व्यक्ति को ट्रांसफर करने का अनुरोध किया।\n</context>"
                "\nQuery: 'कृपया मुझे नीरज से जोड़ें।'"
                "\nIntent: calltransfer - उपयोगकर्ता नीरज को सीधे ट्रांसफर करने का अनुरोध कर रहा है।"

                "\n<context>\nग्राहक दूसरे व्यक्ति को ट्रांसफर करने का अनुरोध कर रहा है।\n</context>"
                "\nQuery: 'क्या आप कॉल राहुल को ट्रांसफर कर सकते हैं?'"
                "\nIntent: calltransfer - उपयोगकर्ता राहुल को ट्रांसफर करने का अनुरोध कर रहा है।"

                "\n<context>\nग्राहक सहायता एजेंट से जोड़ने के लिए कह रहा है।\n</context>"
                "\nQuery: 'मुझे अभी एक एजेंट से जोड़ें।'"
                "\nIntent: calltransfer - उपयोगकर्ता तत्काल सहायता मांग रहा है।"

                "\n<context>\nअप्रत्याशित या अस्पष्ट प्रतिक्रिया एपीआई से प्राप्त हुई।\n</context>"
                "\nQuery: 'HTTP/1.1 200 OK'"
                "\nIntent: general - अप्रत्याशित प्रतिक्रिया के कारण फॉलबैक।"
                "\nAction Response: 'क्षमा करें, मुझे कोई जानकारी नहीं मिली। कृपया convox से संबंधित जानकारी पुनः प्रयास करें।'"

                "\nअब वर्गीकृत करें:'"
                "\n<context>\n{context}\n</context>\nQuery: {input}\nIntent:"
            ),
            )

        intent_chain = LLMChain(llm=llm2, prompt=intent_prompt_template)
        intent_response = intent_chain.run(input=query, context=context_text)

        # Extract intent and explanation
        intent_lines = intent_response.split("\n")
        predicted_intent = re.sub(r"[\\_/@]", "", intent_lines[0].split("-")[0].strip().lower())
        explanation = "-".join(intent_lines[0].split("-")[1:]).strip() if "-" in intent_lines[0] else "No explanation provided."
        # Validate intent
        if predicted_intent.startswith("intent:"):
            result = re.search(r"intent:\s*(\w+)",predicted_intent)
            predicted_intent = result.group(1) if result else None
        print("Padma",predicted_intent)
        # Validate intent
        valid_intents = {"calltransfer", "general", "endconversation"}
        if predicted_intent not in valid_intents:
            logging.warning("Unexpected intent predicted: %s", predicted_intent)
            predicted_intent = "general"
            explanation = "Fallback to general intent due to unexpected response."

        # Default flags
        callend = "0"
        calltransfer = "0"
        action_response = ""

        # Default flag for call status
        callstatus = "active"  # Possible values: "active", "transfer", "end_conversation"
        contact_number=""
        # Handle intent-based actions
        if predicted_intent == "calltransfer":
            # Check if the query directly specifies the transfer target
            match = re.search(r"(?:transfer to|connect with|connect me to|please connect me with|with)\s+(\w+)", query.lower()) or re.search(r"(?:transfer to|connect with|connect me to|please connect me with|with)\s+(\w+)", explanation.lower())

            if match:
                contact_name = match.group(1).capitalize()

                # Search the text file for the contact's number
                try:
                    with open("contacts.txt", "r") as file:
                        contacts = {line.split(":")[0].strip(): line.split(":")[1].strip() for line in file}
                        if contact_name in contacts:
                            contact_number = contacts[contact_name]

                            # Mock transfer logic
                            logging.info(f"Transferring call to {contact_name} at {contact_number}")
                            #action_response = f"Transferring your call to {contact_name} now."
                            action_response = (
                                f"कृपया प्रतीक्षा करें जब तक हम आपकी कॉल को स्थानांतरित कर रहे हैं।"
                            )
                            calltransfer = "1"
                            callstatus = "transfer"
                        else:
                            calltransfer = "1"
                            callstatus = "transfer"
                            action_response = (
                                f"कृपया प्रतीक्षा करें जब तक हम आपकी कॉल को स्थानांतरित कर रहे हैं।"
                            )
                except Exception as file_error:
                    logging.error("Error reading contacts file: %s", file_error)
                    action_response = "An error occurred while looking up the contact information."
            else:
                callstatus = "active"
                redis_client.set(conversation_id + "_state", "awaiting_transfer_confirmation")
                redis_client.set(conversation_id + "_transfer_query", query)
                calltransfer = "1"
                callstatus = "transfer"
                action_response = (
                                f"कृपया प्रतीक्षा करें जब तक हम आपकी कॉल को स्थानांतरित कर रहे हैं।"
                            )
                #action_response = (
                #    "To whom should I transfer the call? You can say 'transfer to Neeraj' or specify a name."
                #)

        # Handle transfer confirmation
        elif redis_client.get(conversation_id + "_state") == "awaiting_transfer_confirmation":
            redis_client.delete(conversation_id + "_state")  # Clear intermediate state
            transfer_query = redis_client.get(conversation_id + "_transfer_query")

            # Extract the name from the user's response
            match = re.search(r"(?:transfer to|connect with|connect me to|please connect me with|with)\s+(\w+)", query.lower()) or re.search(r"(?:transfer to|connect with|connect me to|please connect me with|with)\s+(\w+)", explanation.lower())

            if match:
                contact_name = match.group(1).capitalize()

                # Search the text file for the contact's number
                try:
                    with open("contacts.txt", "r") as file:
                        contacts = {line.split(":")[0].strip(): line.split(":")[1].strip() for line in file}
                        if contact_name in contacts:
                            contact_number = contacts[contact_name]

                            # Mock transfer logic
                            logging.info(f"Transferring call to {contact_name} at {contact_number}")
                            action_response = (
                                f"कृपया प्रतीक्षा करें जब तक हम आपकी कॉल को स्थानांतरित कर रहे हैं।"
                            )
                            #action_response = f"Transferring your call to {contact_name} now."
                            calltransfer = "1"
                            callstatus = "transfer"
                        else:
                            calltransfer = "1"
                            callstatus = "transfer"
                            action_response = (
                                f"कृपया प्रतीक्षा करें जब तक हम आपकी कॉल को स्थानांतरित कर रहे हैं।"
                            )
                except Exception as file_error:
                    logging.error("Error reading contacts file: %s", file_error)
                    action_response = "An error occurred while looking up the contact information."
            else:
                action_response = (f"कृपया प्रतीक्षा करें जब तक हम आपकी कॉल को स्थानांतरित कर रहे हैं।")
                calltransfer = "1"
                callstatus = "transfer"
                #action_response = (
                #    "I couldn't understand your request. Please specify the name clearly, like 'transfer to Neeraj'."
                #)

        elif predicted_intent == "endconversation":
            callstatus = "endconversation"
            action_response = "संपर्क करने के लिए धन्यवाद। आपका दिन शुभ हो!"
        elif predicted_intent == "general":
            conversation_history_text = "\n".join(conversation_history)

            # Chain 2: Process Action for General Intent
            action_prompt_template = PromptTemplate(
                input_variables=["context", "input", "conversation_history"],
                template = (
                "You are a voice bot assistant providing concise, accurate answers based on vector store data. "
                "Always respond in Hindi, regardless of the input language."
                "**Guidelines**:"
                " 1. Retrieve relevant information and respond clearly in 1 sentence."
                " 2. Use a professional, conversational tone in Hindi."
                " 3. If no data is found, politely say in Hindi: *क्षमा करें, मुझे कोई जानकारी नहीं मिली। कृपया convox से संबंधित प्रश्न पूछें।*"
                " 4. Stick to vector store content; avoid unsupported details."
                " 5. Replace ConVox with convox in the response."
                "**Objective**: उपयोगकर्ता की क्वेरी के लिए स्पष्ट और सटीक हिंदी उत्तर प्रदान करें।"
                "\n<context>\n{context}\n</context>\n\n"
                "\n<conversation_history>\n{conversation_history}\n</conversation_history>\n\n"
                "प्रश्न: {input}\nउत्तर:"
            ),

            )

            action_chain = LLMChain(llm=llm, prompt=action_prompt_template)
            action_response = action_chain.run(input=query, context=context_text, conversation_history=conversation_history_text)

            # Fallback response if action chain fails
            if not action_response.strip():
                action_response = "माफ़ कीजिए, मुझे कोई जानकारी नहीं मिल सकी। कृपया Convox से संबंधित प्रश्न पूछें।"

        # Update conversation history in Redis
        conversation_history.append(f"Q: {query}\nA: {action_response}")
        redis_client.set(conversation_id, json.dumps(conversation_history))

        # Prepare final API response
        api_response = {
            "query": query,
            "predicted_intent": predicted_intent,
            "intent_explanation": explanation,
            "action_response": action_response,
            "callstatus": callstatus,
            "contact_number":contact_number,
            "status": "Success",
        }

        logging.info(
            "Processed Query: %s | Intent: %s | Explanation: %s | Action Response: %s | Call End: %s | Call Transfer: %s",
            query,
            predicted_intent,
            explanation,
            action_response,
            callend,
            calltransfer,
        )
        return api_response

    except Exception as e:
        logging.error("Error processing query: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

@app.post("/AI/v6/arabic-query/")
async def arabic_voicebot(query: str = Form(...), conversation_id: str = Form(...), user_id: str = Form(...)):
    """
    Query endpoint to predict intent and process actions based on the intent.
    """
    try:
       # Retrieve user-specific vector store
        user_vector_store = get_user_vector_store(user_id)

        if user_vector_store is None:
            return {
                "query": query,
                "answer": "Sorry, no data found. Please upload documents first.",
                "status": "No Data"
            }

        # Retrieve relevant documents
        retriever = user_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        documents_with_context = retriever.get_relevant_documents(query)

        if not documents_with_context:
            logging.warning("No relevant documents found for query: %s", query)
            return {
                "query": query,
                "predicted_intent": "general",
                "intent_explanation": "No documents found; defaulting to general intent.",
                "action_response": "Sorry, I couldn't find any information. Please ask queries related to Convox.",
                "callend": "0",
                "calltransfer": "0",
                "status": "Success",
            }

        context_text = "\n".join([doc.page_content for doc in documents_with_context[:5]])

        # Load conversation history from Redis
        conversation_id=create_start_conversation(conversation_id)
        conversation_history = redis_client.get(conversation_id)
        if conversation_history:
            conversation_history = json.loads(conversation_history)
        else:
            conversation_history = []

        # Intent Prediction Chain
        intent_prompt_template = PromptTemplate(
            input_variables=["context", "input"],
           template=(
                "You are an intelligent assistant analyzing user queries to classify them into intents."
                "\n**Objective**: Accurately classify the user's query as 'call_transfer', 'general', or 'end_conversation'."
                "\n\n**Examples:**"

                "\n<context>\nCustomer interacting with a support bot.\n</context>"
                "\nQuery: 'Can you transfer me to an agent?'"
                "\nIntent: call_transfer - User requests a call transfer."

                "\n<context>\nCustomer asking about services.\n</context>"
                "\nQuery: 'What services does Convox offer?'"
                "\nIntent: general - User asks a general question."

                "\n<context>\nCustomer ending the chat.\n</context>"
                "\nQuery: 'Thanks for your help. Goodbye!'"
                "\nIntent: end_conversation - User indicates they want to end the conversation."

                "\n<context>\nCustomer having trouble with billing.\n</context>"
                "\nQuery: 'I have an issue with my billing, can I speak to someone?'"
                "\nIntent: call_transfer - User requests assistance involving a transfer."

                "\n<context>\nCustomer curious about support hours.\n</context>"
                "\nQuery: 'What are your customer support hours?'"
                "\nIntent: general - User asks about operational information."

                "\n<context>\nCustomer expresses gratitude and closes chat.\n</context>"
                "\nQuery: 'That answers my question. Thank you and goodbye!'"
                "\nIntent: end_conversation - User concludes the interaction."

                "\n<context>\nCustomer asks about payment options.\n</context>"
                "\nQuery: 'What payment methods do you accept?'"
                "\nIntent: general - User asks for general information about payments."

                "\n<context>\nCustomer requests direct assistance.\n</context>"
                "\nQuery: 'I need help right now. Can you connect me to someone?'"
                "\nIntent: call_transfer - User requests immediate human assistance."

                "\n<context>\nCustomer concludes the interaction politely.\n</context>"
                "\nQuery: 'That's all I needed. Have a great day!'"
                "\nIntent: end_conversation - User politely ends the conversation."

                "\n<context>\nCustomer interacting with a support bot.\n</context>"
                "\nQuery: 'Can you transfer me to an agent?'"
                "\nIntent: call_transfer - User requests a call transfer."

                "\n<context>\nCustomer requesting direct transfer.\n</context>"
                "\nQuery: 'Transfer to Neeraj please.'"
                "\nIntent: call_transfer - User requests direct transfer to Neeraj."

                "\n<context>\nCustomer specifying transfer to another individual.\n</context>"
                "\nQuery: 'Can you transfer the call to Rahul?'"
                "\nIntent: call_transfer - User requests transfer to Rahul."

                "\n<context>\nCustomer asking for support agent.\n</context>"
                "\nQuery: 'Connect me to an agent now.'"
                "\nIntent: call_transfer - User requests immediate human assistance."


                "\nNow classify:"
                "\n<context>\n{context}\n</context>\nQuery: {input}\nIntent:"
            ),
        )

        intent_chain = LLMChain(llm=llm2, prompt=intent_prompt_template)
        intent_response = intent_chain.run(input=query, context=context_text)

        # Extract intent and explanation
        intent_lines = intent_response.split("\n")
        predicted_intent = re.sub(r"[\\_/@]", "", intent_lines[0].split("-")[0].strip().lower())
        explanation = "-".join(intent_lines[0].split("-")[1:]).strip() if "-" in intent_lines[0] else "No explanation provided."

        # Validate intent
        valid_intents = {"calltransfer", "general", "endconversation"}
        if predicted_intent not in valid_intents:
            logging.warning("Unexpected intent predicted: %s", predicted_intent)
            predicted_intent = "general"
            explanation = "Fallback to general intent due to unexpected response."

        # Default flags
        callend = "0"
        calltransfer = "0"
        action_response = ""

        # Default flag for call status
        callstatus = "active"  # Possible values: "active", "transfer", "end_conversation"
        contact_number=""
        # Handle intent-based actions
        if predicted_intent == "calltransfer":
            # Check if the query directly specifies the transfer target
            match = re.search(r"(?:transfer to|connect with|connect me to|please connect me with|with)\s+(\w+)", query.lower()) or re.search(r"(?:transfer to|connect with|connect me to|please connect me with|with)\s+(\w+)", explanation.lower())

            if match:
                contact_name = match.group(1).capitalize()

                # Search the text file for the contact's number
                try:
                    with open("contacts.txt", "r") as file:
                        contacts = {line.split(":")[0].strip(): line.split(":")[1].strip() for line in file}
                        if contact_name in contacts:
                            contact_number = contacts[contact_name]

                            # Mock transfer logic
                            logging.info(f"Transferring call to {contact_name} at {contact_number}")
                            #action_response = f"Transferring your call to {contact_name} now."
                            action_response = (
                                f"يرجى الانتظار بينما نقوم بتحويل مكالمتكم."
                            )
                            calltransfer = "1"
                            callstatus = "transfer"
                        else:
                            calltransfer = "1"
                            callstatus = "transfer"
                            action_response = (
                                f"يرجى الانتظار بينما نقوم بتحويل مكالمتكم."
                            )
                except Exception as file_error:
                    logging.error("Error reading contacts file: %s", file_error)
                    action_response = "An error occurred while looking up the contact information."
            else:
                callstatus = "active"
                redis_client.set(conversation_id + "_state", "awaiting_transfer_confirmation")
                redis_client.set(conversation_id + "_transfer_query", query)
                calltransfer = "1"
                callstatus = "transfer"
                action_response = (
                                f"يرجى الانتظار بينما نقوم بتحويل مكالمتكم."
                            )
                #action_response = (
                #    "To whom should I transfer the call? You can say 'transfer to Neeraj' or specify a name."
                #)

        # Handle transfer confirmation
        elif redis_client.get(conversation_id + "_state") == "awaiting_transfer_confirmation":
            redis_client.delete(conversation_id + "_state")  # Clear intermediate state
            transfer_query = redis_client.get(conversation_id + "_transfer_query")

            # Extract the name from the user's response
            match = re.search(r"(?:transfer to|connect with|connect me to|please connect me with|with)\s+(\w+)", query.lower()) or re.search(r"(?:transfer to|connect with|connect me to|please connect me with|with)\s+(\w+)", explanation.lower())

            if match:
                contact_name = match.group(1).capitalize()

                # Search the text file for the contact's number
                try:
                    with open("contacts.txt", "r") as file:
                        contacts = {line.split(":")[0].strip(): line.split(":")[1].strip() for line in file}
                        if contact_name in contacts:
                            contact_number = contacts[contact_name]

                            # Mock transfer logic
                            logging.info(f"Transferring call to {contact_name} at {contact_number}")
                            action_response = (
                                f"يرجى الانتظار بينما نقوم بتحويل مكالمتكم."
                            )
                            #action_response = f"Transferring your call to {contact_name} now."
                            calltransfer = "1"
                            callstatus = "transfer"
                        else:
                            calltransfer = "1"
                            callstatus = "transfer"
                            action_response = (
                                f"يرجى الانتظار بينما نقوم بتحويل مكالمتكم."
                            )
                except Exception as file_error:
                    logging.error("Error reading contacts file: %s", file_error)
                    action_response = "An error occurred while looking up the contact information."
            else:
                action_response = (f"يرجى الانتظار بينما نقوم بتحويل مكالمتكم.")
                calltransfer = "1"
                callstatus = "transfer"
                #action_response = (
                #    "I couldn't understand your request. Please specify the name clearly, like 'transfer to Neeraj'."
                #)

        elif predicted_intent == "endconversation":
            callstatus = "endconversation"
            action_response = "شكرًا لتواصلكم معنا. نتمنى لكم يومًا سعيدًا!"
        elif predicted_intent == "general":
            conversation_history_text = "\n".join(conversation_history)

            # Chain 2: Process Action for General Intent
            action_prompt_template = PromptTemplate(
                input_variables=["context", "input", "conversation_history"],
                template = (
    """أنت مساعد صوتي يقدم إجابات دقيقة ومختصرة باللغة العربية بناءً على بيانات مخزنة في Vector Store. 
يرجى الالتزام بالإرشادات التالية:
- استرجع المعلومات ذات الصلة وأجب بجملة واحدة واضحة باللغة العربية.
- استخدم نبرة مهنية ومحترمة في إجاباتك.
- إذا لم يتم العثور على بيانات، قل بأدب: "عذرًا، لم أجد أي معلومات. يرجى طرح سؤال متعلق بـ convox."
- استبدل كلمة "ConVox" بـ "convox" في الردود.
- التزم فقط بمحتوى بيانات Vector Store، وتجنب التفاصيل غير المدعومة.

الهدف: تقديم إجابات واضحة ودقيقة باللغة العربية لاستفسارات المستخدم.

<conversation_history>
{conversation_history}
</conversation_history>
<context>
{context}
</context>

السؤال: {input}
الإجابة:"""
),

            )

            action_chain = LLMChain(llm=llm, prompt=action_prompt_template)
            action_response = action_chain.run(input=query, context=context_text, conversation_history=conversation_history_text)

            # Fallback response if action chain fails
            if not action_response.strip():
                action_response = "عذرًا، لم أجد أي معلومات. يرجى طرح أسئلة متعلقة بـ Convox."

        # Update conversation history in Redis
        conversation_history.append(f"Q: {query}\nA: {action_response}")
        redis_client.set(conversation_id, json.dumps(conversation_history))

        # Prepare final API response
        api_response = {
            "query": query,
            "predicted_intent": predicted_intent,
            "intent_explanation": explanation,
            "action_response": action_response,
            "callstatus": callstatus,
            "contact_number":contact_number,
            "status": "Success",
        }

        logging.info(
            "Processed Query: %s | Intent: %s | Explanation: %s | Action Response: %s | Call End: %s | Call Transfer: %s",
            query,
            predicted_intent,
            explanation,
            action_response,
            callend,
            calltransfer,
        )
        return api_response

    except Exception as e:
        logging.error("Error processing query: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")




@app.post("/AI/v6/rasa-query/")
async def voicebot7(query: str = Form(...), conversation_id: str = Form(...)):
    """
    Query endpoint for voicebot v7:
      - All processing is offloaded to an external API.
      - Conversation history is maintained via Redis.
      - After calling the external API, the 'call_end' field is inspected:
           * If call_end == 0, callstatus is set to "active".
           * If call_end == 1, callstatus is set to "endconversation".
    """
    try:
        # Initialize or retrieve conversation history from Redis.
        conversation_id = create_start_conversation(conversation_id)
        history_data = redis_client.get(conversation_id)
        if history_data:
            conversation_history = json.loads(history_data)
        else:
            conversation_history = []

        # -------------------- Call External API -------------------- #
        external_api_url = "https://chat.deepijatel.in:7757/api/v1/rasabot"
        payload = json.dumps({
            "sender": conversation_id,
            "customer_name": "Neeraj",  # Adjust or extract this value as needed.
            "date": "24-02-2025",
            "amount": "0",
            "message": query,
            "bounce_charge": "0",
            "penal_charges": "0"
        })
        headers = {
            'session': 'ntFWpr',
            'X-Api-Key': 'fd77e5f9-3e41-40c6-bc08-a7b80312fa70',
            'Content-Type': 'application/json'
        }

        try:
            api_resp = requests.post(external_api_url, headers=headers, data=payload)
            if api_resp.status_code == 200:
                external_result = api_resp.json()
                print(external_result)
            else:
                logging.error("External API error: %s", api_resp.text)
                external_result = {
                    "query": query,
                    "predicted_intent": "",
                    "intent_explanation": "",
                    "action_response": "Sorry, I couldn't retrieve information at this time.",
                    "callstatus": "active",
                    "contact_number": "",
                    "status": "Error"
                }
        except Exception as api_error:
            logging.error("Error calling external API: %s", api_error)
            external_result = {
                "query": query,
                "predicted_intent": "",
                "intent_explanation": "",
                "action_response": "Sorry, an error occurred while processing your request.",
                "callstatus": "active",
                "contact_number": "",
                "status": "Error"
            }

        # -------------------- Set Call Status Based on 'call_end' -------------------- #
        # If external_result includes a "call_end" field, adjust the call status accordingly.
        if "callend" in external_result:
            try:
                # Convert to integer in case it's a string.
                call_end_value = int(external_result["callend"])
                if call_end_value == 0:
                    callstatus = "active"
                elif call_end_value == 1:
                    callstatus = "endconversation"
            except (ValueError, TypeError) as e:
                logging.error("Error converting call_end value: %s", e)
                # Default to active if conversion fails.
                callstatus = "active"
        action_response=external_result.get('data')
        # -------------------- Update Conversation History -------------------- #
        conversation_history.append(f"Q: {query}\nA: {external_result.get('data', '')}")
        redis_client.set(conversation_id, json.dumps(conversation_history))
        external_result = {
            "query": query,
            "predicted_intent": "predicted_intent",
            "intent_explanation": "explanation",
            "action_response": action_response,
            "callstatus": callstatus,
            "contact_number":"contact_number",
            "status": "Success",
        }
        return external_result

    except Exception as e:
        logging.error("Error processing query: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
