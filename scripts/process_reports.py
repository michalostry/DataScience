import os
import ufal.morphodita as morphodita

# Get the current working directory
current_directory = os.getcwd()

# Load the morphological model (adjust the path to your model file)
model_path = os.path.join(current_directory, '..', 'models', 'czech-morfflex-pdt-161115.tagger')  # Example model path
tagger = morphodita.Tagger.load(model_path)

if not tagger:
    print("Failed to load the model")
else:
    print("Model loaded successfully")
