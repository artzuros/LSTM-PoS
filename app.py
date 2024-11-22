import streamlit as st
import torch
from model import POSModel
import nltk
from nltk.corpus import treebank
from collections import defaultdict

# Download required NLTK resources
nltk.download('treebank')
nltk.download('universal_tagset')

# Load the data and model
def load_model(model_path, vocab_size, tagset_size, embedding_dim, hidden_dim):
    model = POSModel(vocab_size, tagset_size, embedding_dim, hidden_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def test_model(model, sentence, vocab, tagset):
    model.eval()
    with torch.no_grad():
        inputs = [vocab.get(word.lower(), vocab['<unk>']) for word in sentence.split()]
        inputs_tensor = torch.tensor(inputs).unsqueeze(0)
        tag_scores = model(inputs_tensor, torch.tensor([len(inputs)]))
        tag_ids = torch.argmax(tag_scores, dim=2)
        tags = [list(tagset.keys())[tag_id] for tag_id in tag_ids[0]]
        return list(zip(sentence.split(), tags))

# Prepare vocab and tagset
sentences = treebank.tagged_sents(tagset='universal')
train_data = sentences[:3000]

vocab = {word for sent in train_data for word, tag in sent}
vocab = {word: i+2 for i, word in enumerate(sorted(vocab))}
vocab['<pad>'] = 0  # Padding
vocab['<unk>'] = 1  # Unknown words

tagset = {tag for sent in train_data for word, tag in sent}
tagset = {tag: i for i, tag in enumerate(tagset)}

# Load the model
model_loaded = load_model('pos_model.pth', len(vocab), len(tagset), 100, 128)

# Define color mapping for POS tags
TAG_COLORS = {
    'NOUN': 'lightblue',
    'VERB': 'lightgreen',
    'ADJ': 'lightcoral',
    'ADV': 'lightyellow',
    'PRT': 'lightgray',
    'DET': 'lightpink',
    'ADP': 'lightseagreen',
    'NUM': 'lightsalmon',
    'CONJ': 'lightskyblue',
    'X': 'lightsteelblue',
    'PUNCT': 'lightgoldenrodyellow'
}

# Function to display colored words
def display_colored_sentence(sentence, tagged_sentence):
    tagged_words = list(zip(sentence.split(), tagged_sentence))
    html_content = ""
    for word, tag in tagged_words:
        tag_color = TAG_COLORS.get(tag, 'white')  # Default to white if tag not found
        html_content += f'<span style="background-color:{tag_color}; padding: 0.2em; border-radius: 5px;">{word} ({tag})</span> '
    return html_content

# Streamlit App
st.title("POS Tagging with LSTM")
st.write("""
This is a part-of-speech (POS) tagging application built using Long Short Term Memory. 
The model tags each word in a sentence with its corresponding part-of-speech (POS) tag.
""")

# User input
sentence_input = st.text_input("Enter a sentence:", "A very good man")

if sentence_input:
    tagged_sentence = test_model(model_loaded, sentence_input, vocab, tagset)
    
    # Display the results
    st.subheader("Tagged Sentence:")
    colored_sentence = display_colored_sentence(sentence_input, [tag for _, tag in tagged_sentence])
    st.markdown(colored_sentence, unsafe_allow_html=True)

    # Explanation of how things work
    st.subheader("How does this work?")
    st.write("""
    The model uses a custom LSTM architecture to predict the part-of-speech tag for each word in a sentence.
    Here’s a breakdown of the process:
    
    1. **Input Sentence**: The sentence is split into individual words.
    2. **Word to ID Conversion**: Each word is converted into an integer ID using the vocabulary.
    3. **Embedding Layer**: The word IDs are converted into dense vectors (embeddings) representing the words.
    4. **LSTM Layer**: The embeddings pass through a custom LSTM layer that captures the temporal dependencies between words.
    5. **Tag Prediction**: The LSTM output is passed through a fully connected layer to predict the most likely POS tag for each word.
    
    """)
    # The color coding helps visualize the predicted POS tag for each word:
    # st.write("**Color Legend:**")
    # for tag, color in TAG_COLORS.items():
    #     st.write(f"- **{tag}**: {color}")

