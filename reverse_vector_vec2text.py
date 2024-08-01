""" reverse_vector_vec2text.py

    S. Kurowski, June 2024

    Demo "reversal" of an text-embedding-ada-002 embedding vector into
    text with vec2text module, an iterative embedding dictionary model
    trained (only) upon text-embedding-ada-002 to inform text guesses.

    Example vector text targets and outputs in a few dozen iterations:

    Original source texts: [
        'Be at one with your power, peace and joy.',
        'Godel, Escher, Bach: An Eternal Golden Braid.',
        'Scott Kurowski demonstrates iterative embedding vector reversal at this very moment.',
        'The MongoDB application was modified to be resilient to their Atlas cluster elections by using more-economical $set updates, exponential retry back-off timing, smaller update batches, and a smaller connection pool size.'
    ]
    GPU-inverted text-embedding-ada-002 vectors (regenerated texts): [
        'Be at one with your power, peace and joy.',
        'Bach, Goethe, Bach: An Eternal Golden Braid.',
        'Scott Kurowski demonstrates iterative vector embedding reversal at this very moment.',
        'The MongoDB application was adapted to more economic resiliency by using $Atlas set-back updates, a small set-back pool, and a change-in-connection-timer to reduce the number of retry attempts.'
    ]

    Licensed separately under Software for Open Models License (SOM)
    Version 1.0 dated August 30th, 2023
    https://github.com/jxmorris12/vec2text/blob/master/LICENSE

    No guarantees or warranties are made in this demo.
"""

# pylint: disable-msg=C0301,C0103,W0718,W0621

import pickle
import logging
import torch
import vec2text
from openai import OpenAI
import numpy as np

logging.Formatter.default_msec_format = '%s.%03d'
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("reverse_vector.log",'w'),
        logging.StreamHandler()
    ]
)

# must have EMBEDDING_MODEL = "text-embedding-ada-002"
# for out-of-the-box pre-trained model
EMBEDDING_MODEL = "text-embedding-ada-002"
DEVICE = vec2text.models.model_utils.get_device() # use GPU when available
CORRECTOR = vec2text.load_pretrained_corrector(EMBEDDING_MODEL)

model_client = OpenAI()

def get_embeddings_vectors(text_list, model=EMBEDDING_MODEL) -> list[np.ndarray]:
    """ get embedding vectors and load into a list of numpy arrays. """
    response = model_client.embeddings.create(
        input=text_list,
        model=model,
        encoding_format="float", # match np.dtype = 'float32'
    )
    outputs = []
    outputs.extend([np.array(e.embedding,dtype='float32') for e in response.data])
    return outputs

def get_embeddings_tensor_from_vectors(vectors_list:list) -> torch.Tensor:
    """ Convert embeddings vectors into a float32 2-D Tensor in the GPU. """
    vectors_list = np.array(vectors_list,dtype='float32')
    return torch.Tensor(vectors_list).to(DEVICE)

def get_embeddings_tensor_from_texts(text_list, model=EMBEDDING_MODEL) -> torch.Tensor:
    """ Get embeddings vectors from texts list directly into an embeddings Tensor. """
    return get_embeddings_tensor_from_vectors(
        get_embeddings_vectors(text_list, model)
    )

def reverse_embedding_tensor_to_texts(T_embeddings_GPU:torch.Tensor) -> list[str]:
    """ Reverse an embedding vector 2-D Tensor back into texts. """
    texts = vec2text.invert_embeddings(
        embeddings=T_embeddings_GPU,
        corrector=CORRECTOR,
        num_steps=20,
        sequence_beam_width=4,
    )
    return texts

def reverse_embedding_vectors_to_texts(vectors:list|np.ndarray) -> list[str]:
    """ Reverse an embedding vectors back into texts. """
    return reverse_embedding_tensor_to_texts(
        get_embeddings_tensor_from_vectors(vectors)
    )

if __name__ == "__main__":

    # stand-alone app mode

    # encode the TARGET "mystery" vector to be reversed
    # (otherwise, fetch it from somewhere)
    MYSTERY_TARGET_TEXTS = [
        "Be at one with your power, peace and joy.",
        "Godel, Escher, Bach: An Eternal Golden Braid.",
        "Scott Kurowski demonstrates iterative embedding vector reversal at this very moment.",
        "The MongoDB application was modified to be resilient to their Atlas cluster elections by using more-economical $set updates, exponential retry back-off timing, smaller update batches, and a smaller connection pool size."
    ]

    # get initial "mystery" embeddings list of the MYSTERY_TARGET_TEXTS
    # (as if we didn't know what their texts were)
    # and cache them as if in a Mongo vector db
    v_targets = get_embeddings_vectors(MYSTERY_TARGET_TEXTS)
    f = open('reverse_vectors.bin','wb')
    pickle.dump(v_targets,f)
    f.close()

    # uncache/load the "mystery" vectors to reverse
    # (or load it from a Mongo vector db)
    f = open('reverse_vectors.bin','rb')
    v_targets = pickle.load(f)
    f.close()

    # load list of 1-D "mystery" target vectors into 2-D embeddings Tensor on GPU device
    T_embeddings_GPU = get_embeddings_tensor_from_vectors(v_targets)

    # generatively invert vector's 2-D embeddings Tensor back into to a list of texts
    texts = reverse_embedding_tensor_to_texts(T_embeddings_GPU)

    print(f"\n\nInverted vector texts: '{texts}'")
    print(f"\n\nOriginal source texts: '{MYSTERY_TARGET_TEXTS}'")
