import ollama
import os
import concurrent.futures
import time
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont

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

# The directories to save the output files
OUTPUT_DIR = "test_results"
RENDER_DIR = "renders"

# --- Main Script ---

def create_directories():
    """Creates the output and render directories if they don't already exist."""
    for directory in [OUTPUT_DIR, RENDER_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def create_error_image(error_message, path, width=800, height=600):
    """
    Creates a white PNG image with the given error message written on it.
    Args:
        error_message (str): The error message to write on the image.
        path (str): The path to save the output .png file.
        width (int): The width of the image.
        height (int): The height of the image.
    """
    try:
        img = Image.new('RGB', (width, height), color = 'white')
        d = ImageDraw.Draw(img)
        
        # Try to use a default font, fall back to a basic one if not found
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()
            
        # Add text wrapping for long messages
        text = f"Render Failed:\n\n{error_message}"
        d.text((10,10), text, fill=(0,0,0), font=font)
        
        img.save(path)
        return f"Created error placeholder image at {path}"
    except Exception as e:
        return f"Failed to create error image: {e}"


def render_obj_to_png(obj_path, render_path):
    """
    Renders an OBJ file to a PNG image using PyVista.
    If rendering fails, it creates a placeholder image with the error.
    Args:
        obj_path (str): The path to the input .obj file.
        render_path (str): The path to save the output .png file.
    Returns:
        tuple: A boolean indicating success, and a message.
    """
    try:
        # Ensure the OBJ file is not empty to avoid errors
        if os.path.getsize(obj_path) < 5: # Check for a minimum size, not just non-zero
            raise ValueError("OBJ file is empty or too small to be valid.")

        # Set up an off-screen plotter
        plotter = pv.Plotter(off_screen=True)
        
        # Read the mesh from the file
        mesh = pv.read(obj_path)
        
        # Add the mesh to the plotter with some basic styling
        plotter.add_mesh(mesh, color='tan', show_edges=True)
        
        # Set a standard isometric camera view
        plotter.camera_position = 'iso'
        
        # Take a screenshot
        plotter.screenshot(render_path)
        plotter.close()
        return True, f"Rendered to {render_path}"

    except Exception as e:
        # Clean up the plotter if it exists
        if 'plotter' in locals() and plotter is not None:
            plotter.close()
        
        # Create a placeholder image with the error message
        error_msg = str(e)
        create_error_image(error_msg, render_path)
        return False, f"Failed to render {obj_path}. {create_error_image(error_msg, render_path)}"

def generate_and_save(model_name):
    """
    Sends a prompt to a model, saves the .obj response, and renders it to .png.
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
        
        # Sanitize the model name for the filename
        safe_model_name = model_name.replace(":", "_")
        file_name = f"{safe_model_name}_July_2025.obj"
        file_path = os.path.join(OUTPUT_DIR, file_name)

        # Save the generated content to the file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(generated_content)
        
        # --- Rendering Step ---
        render_file_name = f"{os.path.splitext(file_name)[0]}.png"
        render_path = os.path.join(RENDER_DIR, render_file_name)
        
        render_success, render_message = render_obj_to_png(file_path, render_path)
        
        duration = time.time() - start_time
        
        if render_success:
            return f"[{model_name}] SUCCESS: Saved OBJ to {file_path} and PNG to {render_path} (took {duration:.2f}s)"
        else:
            return f"[{model_name}] PARTIAL SUCCESS: Saved OBJ to {file_path}, but PNG render failed. {render_message} (took {duration:.2f}s)"

    except Exception as e:
        duration = time.time() - start_time if 'start_time' in locals() else 0
        return f"[{model_name}] FAILED: {e} (after {duration:.2f}s)"

def main():
    """
    Main function to run the concurrent model testing and rendering.
    """
    print("--- Starting Concurrent Ollama Model Test & Render ---")
    create_directories()

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
