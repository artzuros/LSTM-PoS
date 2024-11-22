## LSTM-PoS

 This repository contains a Part-of-Speech (PoS) tagging application built using a custom Long Short-Term Memory (LSTM) model. The application uses the NLTK Treebank dataset for training and testing the model. The model is implemented in PyTorch and can be tested using a Streamlit web application.

 ![alt text](/LSTM-PoS/LSTM-POS%20Streamlit.gif)

### Table of Contents
- Overview
- Mathematical Equations
- - Forward Propagation
- - Backward Propagation
- Installation
- Usage
- - Training the Model
- - Testing the Model
- - Running the Streamlit App
- Files

### Overview
 The LSTM-PoS application tags each word in a sentence with its corresponding part-of-speech (PoS) tag. The model uses a custom LSTM architecture to predict the PoS tags. The application includes a Streamlit web interface for easy interaction.

#### Installation
1. Clone the repository:
   
   ```
   git clone https://github.com/yourusername/LSTM-PoS.git
   
   cd LSTM-PoS
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Download the required NLTK resources:
   ```
   import nltk
   nltk.download('treebank')
   nltk.download('universal_tagset')
   ```

#### Usage
##### Running the Streamlit App
   - Run the Streamlit app
   - `streamlit run app.py`
   - Enter a sentence in the text input field to see the PoS tags predicted by the model.
##### Training the Model
   - Run train.py to train the model on the NLTK Treebank dataset.
   - Save the trained model to a file (pos_model.pth).

##### Testing the Model
   - Load the trained model using the load_model function.
   - Test the model on new sentences using the test_model function.

#### Files
- `app.py`: Streamlit web application for PoS tagging.
- `data_preprocessing.py`: Contains functions for preparing - the dataset and a custom Dataset class.
- `model.py`: Contains the custom LSTM model definition.
- `train.py`: Script for training the model.
- `test.py`: Script for testing the model.
- `playground4.ipynb`: Jupyter Notebook for training and testing the model.
- `README.md`: This file.

#### Mathematical Equations
##### Forward Propagation:
- Concatenated input:
 ( \text{concat_inputs}[q] = [\text{hidden_states}[q - 1]; \text{inputs}[q]] )

- Forget gate:
 ( \text{forget_gates}[q] = \sigma(\text{wf} \times \text{concat_inputs}[q] + \text{bf}) )

- Input gate:
 ( \text{input_gates}[q] = \sigma(\text{wi} \times \text{concat_inputs}[q] + \text{bi}) )

- Candidate gate:
 ( \text{candidate_gates}[q] = \tanh(\text{wc} \times \text{concat_inputs}[q] + \text{bc}) )

- Output gate:
 ( \text{output_gates}[q] = \sigma(\text{wo} \times \text{concat_inputs}[q] + \text{bo}) )

- Cell state:
 ( \text{cell_states}[q] = \text{forget_gates}[q] \times \text{cell_states}[q - 1] + \text{input_gates}[q] \times \text{candidate_gates}[q] )

- Hidden state:
 ( \text{hidden_states}[q] = \text{output_gates}[q] \times \tanh(\text{cell_states}[q]) )

- Output:
 ( \text{outputs}[q] = \text{wy} \times \text{hidden_states}[q] + \text{by} )

##### Backward Propagation:
- Final gate weights errors:
 ( \text{dwy} += \text{error} \times \text{hidden_states}[q]^T )
- Final gate biases errors:
 ( \text{dby} += \text{error} )
