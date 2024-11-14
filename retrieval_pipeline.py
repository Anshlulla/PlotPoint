import faiss
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from templates import INITIAL_PROMPT_TEMPLATE, EDIT_PROMPT_TEMPLATE
import os
from dotenv import load_dotenv
import warnings
import torch
import base64
from io import BytesIO
from diffusers import StableDiffusionPipeline
from PIL import Image


warnings.filterwarnings("ignore")

load_dotenv()
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
google_api_key = os.getenv('GOOGLE_API_KEY')

# Initialize embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#embedding_model.eval()

faiss_path = r"C:\Users\Ansh Lulla\PycharmProjects\PlotPoint\venv\data\movie_embeddings_ivf.index"
metadata_path = r"C:\Users\Ansh Lulla\PycharmProjects\PlotPoint\venv\data\metadata.json"

# Load FAISS index
faiss_index = faiss.read_index(faiss_path)

# Load metadata from JSON
with open(metadata_path, "r") as f:
    metadata = json.load(f)

# Initialize the Google Generative AI model
llm = ChatGoogleGenerativeAI(
    model='gemini-pro',
    temperature=0.9,
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

# Initialize memory with a window size of 50 for recent exchanges
memory = ConversationBufferWindowMemory(k=50, return_messages=True)


class DocumentRetriever:
    """Handles document retrieval and similarity ranking with FAISS."""

    def __init__(self, faiss_index, metadata):
        self.faiss_index = faiss_index
        self.metadata = metadata

    def retrieve_documents(self, query, top_k=5):
        """Retrieve top K documents based on FAISS similarity."""
        query_embedding = embedding_model.encode(query).astype("float32").reshape(1, -1)
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        retrieved_metadata = [self.metadata[i] for i in indices[0]]
        return retrieved_metadata, distances[0]

    def rank_documents(self, query, retrieved_metadata, docs):
        """Rank retrieved documents based on cosine similarity."""
        query_embedding = embedding_model.encode(query).reshape(1, -1)
        similarities = []
        for result in retrieved_metadata:
            doc = next((doc for doc in docs if doc['movie_name'] == result['movie_name'] and doc['chunk_index'] == result['chunk_index']), None)
            if doc:
                chunk_embedding = np.array(doc['embedding']).reshape(1, -1)
                similarity = cosine_similarity(query_embedding, chunk_embedding)[0][0]
                similarities.append((doc, similarity))
        ranked_results = sorted(similarities, key=lambda x: x[1], reverse=True)
        return ranked_results


class CustomRAGChain(DocumentRetriever):
    """Handles RAG chain operations for generating and editing movie scripts."""

    def __init__(self, faiss_index, metadata, llm, memory):
        # Initialize DocumentRetriever to set up FAISS index and metadata
        super().__init__(faiss_index, metadata)

        self.initial_template = INITIAL_PROMPT_TEMPLATE
        self.edit_template = EDIT_PROMPT_TEMPLATE
        self.llm = llm
        self.memory = memory
        self.generated_script = ""  # Stores the current generated script
        self.current_sources = []  # Stores sources for the current script

    def generate_script(self, user_query: str, retries=3):
        # Retrieve documents for a new script generation
        retrieved_docs, _ = self.retrieve_documents(user_query)

        if not retrieved_docs:
            return "No relevant documents found for script generation."

        # Format retrieved documents into context
        context = "\n".join([f"Movie: {doc['movie_name']}, Chunk: {doc['chunk_index']}" for doc in retrieved_docs])

        # Format the initial prompt with context and user query
        prompt = self.initial_template.format(context=context, query=user_query)

        attempt = 0
        while attempt < retries:
            attempt += 1
            try:
                # Generate a response from the model
                response = self.llm.invoke(prompt)
                if response and response.content:
                    self.generated_script = response.content  # Store the generated script
                    self.memory.save_context({"input": user_query}, {"output": response.content})

                    # Store the sources for this new script
                    self.current_sources = retrieved_docs

                    # Return the generated script along with sources
                    output = response.content + "\n\nSources for generated script:\n"
                    for doc in self.current_sources:
                        output += f"- Movie: {doc['movie_name']}, Chunk: {doc['chunk_index']}\n"

                    return output
            except Exception as e:
                time.sleep(2)  # Wait briefly before retrying

        return "Failed to generate script after multiple attempts. Please try again later."

    def handle_request(self, user_request: str):
        if not self.generated_script and user_request.lower().startswith("edit:"):
            return "No script has been generated yet to edit."

        # Decide whether to use the initial or edit template
        if user_request.lower().startswith("edit:"):
            edit_query = user_request[5:].strip()  # Extract edit instruction after "edit: "
            # Use the edit template
            prompt = self.edit_template.format(
                generated_script=self.generated_script,
                history=self.memory.buffer,
                query=edit_query
            )
        else:
            # Use the initial template to generate a new script
            return self.generate_script(user_request)

        # Pass the prompt to the model for either case
        response = self.llm.invoke(prompt)
        if response and response.content:
            self.generated_script = response.content  # Update or store the script with the response
            return response.content
        else:
            return "Failed to generate response. Please try again."


class StableDiffusionGenerator:
    def __init__(self, model_path, device="cpu"):
        # Load the Stable Diffusion model with low CPU memory usage mode
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        self.pipeline.to(device)

    def generate_images(self, prompt, num_images=1):
        # Generate images as base64-encoded strings to be sent in JSON response
        images_base64 = []

        for _ in range(num_images):
            image = self.pipeline(prompt).images[0]  # Generate image
            buffered = BytesIO()
            image.save(buffered, format="PNG")  # Save image to buffer in PNG format
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")  # Encode as base64
            images_base64.append(img_str)  # Collect base64 image string

        return images_base64



# Instatiate the CustomRAGChain Object
rag_chain = CustomRAGChain(
    faiss_index=faiss_index,
    metadata=metadata,
    llm=llm,
    memory=memory
)

# Instantiating stable diffusion model
model_path = r"C:\Users\Ansh Lulla\PycharmProjects\PlotPoint\venv\stable_diffusion_model"
stable_diff = StableDiffusionGenerator(model_path=model_path)