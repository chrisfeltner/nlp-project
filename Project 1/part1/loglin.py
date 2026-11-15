from scipy.sparse import csr_matrix, issparse, vstack
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle
from tqdm.auto import tqdm

def to_matrix(vectors, encoder: 'Encoder'):
    """
    Stack encoded vectors into a feature matrix.
    Dense encoders (encoder.dense = True, e.g. sentence embeddings) give a numpy array,
    otherwise we build a sparse matrix (good for token indicators / bag of words).
    Encoders that already return sparse rows are stacked directly, without a dense copy.
    """
    if getattr(encoder, 'dense', False):
        return np.stack(vectors).astype(np.float32)
    if issparse(vectors[0]):
        return vstack(vectors, format='csr')
    return csr_matrix(vectors)

def load(path):
    """
    Load a logistic model from {path}.pickle and return it.
    """
    with open(f"{path}.pickle", 'rb') as f:
        logistic_model = pickle.load(f)
    return logistic_model

def fit(training_set, encoder: 'Encoder', save_to=None, max_iter=100, solver=None):
    """
    Constructs, trains and returns a softmax regressor.
    
    training_set: list of text-label pairs (a text is a string, a label is a 0-based integer)
    encoder: an instance of the Encoder class (such as TokenIndicator)    
    save_to: (str) if given, save the model to {save_to}.pickle
    solver: sklearn solver; by default lbfgs for dense encoders and saga for sparse ones
    """
    # Convert the training and validation sets to vectors
    training_vectors = [encoder(text) for text, label in training_set]
    training_labels = [label for text, label in training_set]

    # Sparse matrix for sparse encoders, plain numpy array for dense ones
    X_train = to_matrix(training_vectors, encoder)

    # Train the logistic model
    # saga is the sparse matrix friendly solver, lbfgs converges much faster on dense features
    if solver is None:
        solver = "lbfgs" if getattr(encoder, 'dense', False) else "saga"
    logistic_model = LogisticRegression(solver=solver, max_iter=max_iter)
    logistic_model.fit(X_train, training_labels)

    if save_to is not None:        
        with open(f"{save_to}.pickle", 'wb') as f:
            pickle.dump(logistic_model, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    return logistic_model        

def predict(texts, model, encoder: 'Encoder'):
    """
    Predicts labels (via the most probable class decision rule) for a collection of texts. 
    (scikit-learn hard-codes the most probable class as the decision rule of choice)
    
    texts: a list of texts, each a string
    model: a trained instance of sklearn's LogisticRegression
    encoder: the Encoder used for in the model's training
    """
    validation_vectors = [encoder(text) for text in texts]
    X_val = to_matrix(validation_vectors, encoder)
    predictions = model.predict(X_val)
    return predictions

def predict_probs(texts, model, encoder: 'Encoder'):
    """
    Predicts labels (via the most probable class decision rule) for a collection of texts. 
    (scikit-learn hard-codes the most probable class as the decision rule of choice)
    
    texts: a list of texts, each a string
    model: a trained instance of sklearn's LogisticRegression
    encoder: the Encoder used for in the model's training
    """
    validation_vectors = [encoder(text) for text in texts]
    X_val = to_matrix(validation_vectors, encoder)
    probs = model.predict_proba(X_val)
    return probs