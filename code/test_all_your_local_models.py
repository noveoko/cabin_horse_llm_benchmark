import ollama
import os
import concurrent.futures
import time

# --- Configuration ---
# List of Ollama models to test
MODELS_TO_TEST = [
    "dolphin-mixtral:latest",
    "llama3.2-vision:11b",
    "mxbai-embed-large:latest",
    "qwq:latest",
    "mistral-small3.1:latest",
    "gemma3:12b",
    "qwen2.5:14b",
    "llama3.2:latest",
    "llama3:latest",
    "llama3:8b",
    "deepseek-r1:14b",
    "qwen2.5:7b",
    "deepseek-r1:32b",
]

# The prompt to be sent to each model
PROMPT = "generate the following scene using the OBJ file format as a text file: a 19th century wooden cabin with a horse standing next to it"

# The directory to save the output files
OUTPUT_DIR = "test_results"

# --- Main Script ---

def create_output_directory():
    """Creates the output directory if it doesn't already exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

def generate_and_save(model_name):
    """
    Sends the prompt to a specific model, gets the response, and saves it to a file.
    Args:
        model_name (str): The name of the model to use.
    Returns:
        str: A message indicating the result of the operation for this model.
    """
    try:
        print(f"[{model_name}] Starting generation...")
        start_time = time.time()

        # Send the prompt to the Ollama API
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': PROMPT}]
        )

        # Extract the content from the response
        generated_content = response['message']['content']
        end_time = time.time()
        duration = end_time - start_time

        # Sanitize the model name for the filename
        safe_model_name = model_name.replace(":", "_")
        file_name = f"{safe_model_name}_July_2025.obj"
        file_path = os.path.join(OUTPUT_DIR, file_name)

        # Save the generated content to the file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(generated_content)

        return f"[{model_name}] SUCCESS: Saved to {file_path} (took {duration:.2f}s)"

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        return f"[{model_name}] FAILED: {e} (after {duration:.2f}s)"

def main():
    """
    Main function to run the concurrent model testing.
    """
    print("--- Starting Concurrent Ollama Model Test ---")
    create_output_directory()

    # Use a ThreadPoolExecutor to run the tests in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map the generate_and_save function to each model in the list
        future_to_model = {executor.submit(generate_and_save, model): model for model in MODELS_TO_TEST}

        # Process the results as they complete
        for future in concurrent.futures.as_completed(future_to_model):
            model = future_to_model[future]
            try:
                result_message = future.result()
                print(result_message)
            except Exception as exc:
                print(f"[{model}] An error occurred during execution: {exc}")

    print("\n--- All tests completed. ---")

if __name__ == "__main__":
    main()
