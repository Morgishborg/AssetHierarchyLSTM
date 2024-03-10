# -*- coding: utf-8 -*-
"""Tag_Vec_embed_equip_tokens.ipynb
Course: 2.126: AI and ML for Eng. Des.
Morgen Pronk
This code trains a sequential LSTM model that tries to
predict the parent equipment in an equipment hierarchy
It uses either character-wise sequential prediction or tag-element wise
prediction depending on how it is configured.
Current configuration is element-wise, with the elements being defined by '-' in the tag
"""

"""Importing required Libraries"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, TimeDistributed, Dense

"""Setting up path dependencies"""
path_root = '.' #This needs to change depending on if it is running in a notebook or local computer
file_path = f'{path_root}/FLOC_SFLO_Input.csv'
df_tags = pd.read_csv(file_path)

"""Data Formatting"""
# Turn all the values to strings
df_tags = df_tags.astype(str)
# Replace 'nan' with ''
df_tags = df_tags.replace('nan', '')

# split the tags up into sections using '-'
df_tags = df_tags.applymap(lambda x: x.split('-'))
# For doing this character-wise, we would do the following:
#df_tags = df_tags.applymap(lambdda x: list(x)) # !!have not tested yet!!
# Note: The model will not be able to predict using things that it has not seen before. For characters, this makes
# it variable, but for tag elements, it means if it hasn't seen "WASX" and it goes in for prediction, it will error out

# Rename dataframe columns for easier referencing
# X is the child and y is the parent. if we go this way it's a 1 to many relationship
df_tags.columns = ['X', 'y']

"""Tokenization"""
## We need to make tokens from the characters or elements in order to put into a machine learing model
# We need to get a list of all unique tag elements
def unique_tag_elements(df):
    ## This function goes through the df and finds all of the unique elements in each list in the df
    ## input: dataframe composed of 2 columns and with lists containing strings in each element
    ## output: Set object containing all of the unique elements out of all of the lists
    # Flatten all the lists in the DataFrame into a single list
    all_values = [item for sublist in df.values.flatten() for item in sublist]
    # Find unique elements
    unique_values = set(all_values)
    return unique_values

# get the unique set of elements for the tokenization dictionary
unique_values = unique_tag_elements(df_tags)
# Ensure that '' is included in unique_values
unique_values.add('')

# Create the token dictionary
token_dict = {value: (i if value != '' else 0) for i, value in enumerate(unique_values)}

# tokenize and pad the dataframe
# We need to get the length of the largest a tag can be. This value will be different for
#  character-wise and elements-wise tokenization
max_length = max(len(item) for sublist in df_tags.values.flatten() for item in sublist)

# Now, we will write a function that takes a list or the characters or tag elements representing a tag
#  and tokenize it, and pad it so all the tags are the same length
def tokenize_and_pad_list(lst):
    # This function tokenizes a list representing a tag and returns a list of the token representation of that tag
    # input: character or element-wise list representing the tag
    # output: token represenation of the tag using token_dict
    # Tokenize the list
    tokenized_list = [token_dict[item] for item in lst]
    # Pad the list to the maximum length
    padded_list = tokenized_list + [token_dict['']] * (max_length - len(tokenized_list))
    return padded_list

# apply the token/padding function to the dataframe
df_tokenized_padded = df_tags.applymap(tokenize_and_pad_list)

# We will make a X variable and y variable to put in our machine learning models
# First X:
# take the dataframe column and turn it into a numpy array:
X = np.stack(df_tokenized_padded['X'].to_list())
# We need it to be 3D (tags, tokens, 1) for the specific model (Sequential())
X = X.reshape((X.shape[0], X.shape[1], 1))

# Now y:
# y needs to be different, because we are planning to predict the probabilities for each token.
# We are going to make a 'one-hot' encoding representation of the tags for y:
y = df_tokenized_padded['y']
y_list = y.tolist()
sequence_length = len(y_list[0])  # Length of the first sequence
num_tokens = len(token_dict.items())
# One-hot Encoding
num_samples = len(y_list)  # Number of samples
# Initialize an empty array for the one-hot encoded data
y_train = np.zeros((num_samples, sequence_length, num_tokens), dtype='int')
# One-hot encode each token
for i in range(num_samples):
    for j in range(sequence_length):
        token_index = y_list[i][j]
        y_train[i, j, token_index] = 1
# rename y for clarity
y_onehot = y_train
#y_onehot.shape #shape is (tags, elements_in_tag, possible_tokens)

"""Creating Model"""

## Cutom loss function
# We want an extra penalty for having large probabilities when there should be anything at all was the intent of this
#  custom function, however, it only seems to help slightly
def custom_loss(y_true, y_pred):
    # Define a scaling factor for the extra penalty
    penalty_scale = 15.0

    # Compute the standard categorical cross-entropy loss
    base_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

    # Identify where the true label is zero
    zero_mask = tf.cast(tf.equal(tf.argmax(y_true, axis=-1), 0), tf.float32)

    # Calculate extra penalty for non-zero predictions when true is zero
    non_zero_predictions = tf.reduce_sum(y_pred[:, :, 1:], axis=-1)
    extra_penalty = penalty_scale * zero_mask * non_zero_predictions

    # Return the sum of base loss and extra penalty
    return base_loss + extra_penalty


# # model architecture
Num_of_possible_tokens = vocab_size = len(token_dict.items())
embedding_dim = 15 #my input length is only 15 so a embedding dim of 20 might not be that good

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=15))
model.add(LSTM(units=50, return_sequences=True))  # return_sequences=True for outputting a sequence
model.add(TimeDistributed(Dense(Num_of_possible_tokens, activation='softmax')))

# Assuming custom_loss is defined somewhere in your code
model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy']) # standard loss: loss='categorical_crossentropy'

batch_size = 32
num_epochs = 10

"""Training the model:"""
model.fit(X, y_onehot, batch_size=batch_size, epochs=num_epochs)

"""predictions:"""
# Let's make a function that takes the output of the prediction (matrix representing probabilities of each token
# for each position in the tag and outputs a readable tag
# To do this we will take the most probable token and assume that is the correct one

def pred_2_tag(prediction):
  predicted_sequence = np.argmax(prediction, axis=-1)
  inverse_char_dict = {v: k for k, v in token_dict.items()}
  # Convert each token in predicted_sequences back to tag element
  predicted_sequence = [inverse_char_dict[token] for token in predicted_sequence[0]] #need the zero just because the shape of the numpy array
  pred_tag = "-".join(predicted_sequence)
  return pred_tag

# Let's make a single prediction
def single_prediction(start_tag):
    # This function takes a string, representing a tag, and returns a string that is the predicted parent tag
    start_tag = start_tag.split('-')
    tok_tag = tokenize_and_pad_list(start_tag)
    model_input_tag = np.array(tok_tag).reshape(1,15,1)
    prediction = model.predict(model_input_tag)
    new_tag = pred_2_tag(prediction).rstrip('-')
    return new_tag

start_tag = "KDU-NOFC-W152-08-10-PSHL-1000"
print(f'{start_tag} -> {single_prediction(start_tag)}') #correct

start_tag = "KDU-NOFC-W152-08-10-W-1000"
print(f'{start_tag} -> {single_prediction(start_tag)}') #KDU-NOFC-W152-08-10 not correct

start_tag = 'KDU-W152-0010-0122-WELL'
print(f'{start_tag} -> {single_prediction(start_tag)}') #KDU-W152-0010 not correct

start_tag = 'KDU-W152-0010-0122'
print(f'{start_tag} -> {single_prediction(start_tag)}') #correct

start_tag = 'KDU-W152-0010'
print(f'{start_tag} -> {single_prediction(start_tag)}') #KDU-W152-0010 not correct

start_tag = 'KDU-W152'
print(f'{start_tag} -> {single_prediction(start_tag)}') #KDU-W152 not correct

# multiple chained predictions:
# The predicted tag becomes the tag to predict
start_tag = "KDU-NOFC-W152-08-10-PSHL-1000"
eqp_lst = [start_tag]
for i in range(10):
    start_tag = start_tag.split('-')
    tok_tag = tokenize_and_pad_list(start_tag)
    model_input_tag = np.array(tok_tag).reshape(1,15,1)
    prediction = model.predict(model_input_tag)
    new_tag = pred_2_tag(prediction)
    start_tag = new_tag.rstrip('-')
    eqp_lst.append(start_tag)

eqp_lst
