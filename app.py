from flask import Flask, request, jsonify
from retrieval_pipeline import rag_chain, stable_diff
import json
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def home():
    return 'Welcome to the Movie Script Generator!'

@app.route("/generate_script", methods=["POST"])
def generate_script():
    try:
        user_query = request.json.get("query", "")

        if not user_query:
            logging.error("No 'user_query' provided in the request.")
            return jsonify({"error": "No query provided."}), 400

        logging.debug(f"Received query: {user_query}")

        output = rag_chain.handle_request(user_query)
        logging.debug(f"Generated output: {output}")

        return jsonify({"output": output}), 200
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({"error": "An error occurred while processing the request."}), 500

@app.route('/edit_script', methods=['POST'])
def edit_script():
    user_request = request.json.get('request', '')
    if not user_request:
        return jsonify({"error": "Request not provided"}), 400
    output = rag_chain.handle_request(user_request)  # Handle the edit request
    return jsonify({"editedScript": output}), 200  # Return the edited script

@app.route('/get_sources', methods=['GET'])
def get_sources():
    sources = rag_chain.current_sources  # Get the current sources used in script generation
    if not sources:
        return jsonify({"error": "No sources available"}), 404
    return jsonify({"sources": sources}), 200

@app.route('/generate_images', methods=["POST"])
def generate_images():
    data = request.get_json()
    prompt = data.get("query")
    num_images = data.get("numImages", 1)

    if not prompt:
        return jsonify({"error": "Provide a Prompt to generate images"}), 400

    try:
        images = stable_diff.generate_images(prompt, num_images)
        return jsonify({"images": images}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/test', methods=['GET'])
def test_route():
    return jsonify({"message": "Test route working!"})



if __name__ == "__main__":
    app.run(debug=True)