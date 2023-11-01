import nltk
import string
import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk.tokenize import word_tokenize
from flask import Flask, request, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer

data_file = 'dialogs.txt'  # Replace with your file path

conversations = []

with open(data_file, 'r', encoding='utf-8') as file:
    for line in file:
        conversations.append(line.strip())

# Now, the 'conversations' list contains your data.



# Download necessary NLTK resources (if you haven't already)
nltk.download('punkt')



# Define a function to preprocess and format text
def preprocess_text(text):
    # Tokenization: Split the text into words
    tokens = word_tokenize(text)

    # Cleaning: Remove punctuation, convert to lowercase, and remove extra whitespace
    cleaned_tokens = [token.lower() for token in tokens if token not in string.punctuation]
    cleaned_text = ' '.join(cleaned_tokens)

    return cleaned_text

# Read the text dataset from a file
data_file = 'dialogs.txt'  

with open(data_file, 'r', encoding='utf-8') as file:
    dataset = [line.strip() for line in file]

# Preprocess and format the entire dataset
preprocessed_dataset = [preprocess_text(sentence) for sentence in dataset]

# Optionally, you can save the preprocessed dataset to a new file for later use
output_file = 'preprocessed_dataset.txt'

with open(output_file, 'w', encoding='utf-8') as file:
    for sentence in preprocessed_dataset:
        file.write(sentence + '\n')




app = Flask(__name)

# Load the pre-trained chatbot model and tokenizer
model_name = "gpt2" 
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define a function to generate responses
def generate_response(input_text, max_length=100):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    response = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    return tokenizer.decode(response[0], skip_special_tokens=True)

@app.route("/", methods=["GET", "POST"])
def chatbot():
    if request.method == "POST":
        user_input = request.form["user_input"]
        bot_response = generate_response(user_input)
        return render_template("index.html", user_input=user_input, bot_response=bot_response)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)