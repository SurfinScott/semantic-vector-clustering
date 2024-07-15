""" load_vector_db.py

    S. Kurowski, July 2024

    Loads demo_db.demo_target_vectors with 26 texts and their embedding vectors.
    Automatically called from demo_vector_clustering.py.

    No guarantees or warranties are made in this demo.
"""
# pylint: disable-msg=C0103,C0301,W0621,W0718

import pymongo
from openai import OpenAI

# local test Mongo server, or any valid user-auth'd URI
vector_mongo_uri = "localhost:27017"
vector_db_name = "demo_db"

# Our demo input vector texts, having 5 main topics "far apart" as semantically distinct,
# where 2 topics (MongoDB items, Bach/Escher/Gödel works) are diverse enough to be
# potentially topically-split on Atlas and Gödel concepts.
# This should allow clustering at MINIMUM_VECTORS_PER_CLUSTER "granularity" of 2 or 3.
# The expected result should be 6 or 7 clusters and 2 or 3 branch/parent clusters.

TEXTS = [
    "Be at one with your power, joy and peace.",
    "Know the flow of the much greater oneness we share.",
    "Let one's superpower be considered choices in the network of life.",

    "MongoDB Ops Manager",
    "MongoDB Cloud Manager",
    "MongoDB Cloud Manager Backups",
    "MongoDB Atlas Database",
    "MongoDB Atlas Stream Processing",
    "MongoDB Atlas Vector Search",
    "MongoDB Atlas Data Lake",
    "MongoDB Enterprise Database Server",

    "Gödel, Escher, Bach: An Eternal Golden Braid",
    "How Gödel's Theorems Shape Quantum Physics as explored by Wheeler and Hawking",
    "Bach, Johann Sebastian - Six Partitas BWV 825-830 for Piano",
    "M.C. Escher, the Graphic Work",
    "Bach's baroque style features recursion or self-referencing iterated functions reminiscent of Escher's nested visuals and Gödelian self-references.",
    "In 1931, Gödel proved the profound duality that formal systems cannot be both self-consistent and complete.",
    "John Von Neumann was able to derive Gödel's 2nd theorem from his 1st before Gödel published it.",

    "My cat is a fun black and white half-sized tuxedo.",
    "Some people prefer the company of dogs instead of cats.",
    "My friend has a large saltwater aquarium with colorful and exotic tropical fish.",
    "My clever dog opens locked windows and doors to follow me.",

    "Mesopotamian tablets record a fantastic version of human history.",
    "North American burial mounds often held deceased local royal families.",
    "Mayan pyramids predated most Aztec pyramids.",
    "The Aztecs Quetzalcoatl closely resembles the Egyptian god Thoth.",
]

VECTOR_COLLECTION = "demo_target_vectors" # destination MongoDB Atlas vector collection
EMBEDDING_MODEL = "text-embedding-ada-002" # embed texts as vectors using this model

mongo_client = pymongo.MongoClient(vector_mongo_uri)
vector_db = mongo_client[vector_db_name]

model_client = OpenAI()


def get_embeddings_vectors(text_list, model=EMBEDDING_MODEL) -> list[list[float]]:
    """ get embedding vectors and load into a list of float lists. """
    response = model_client.embeddings.create(
        input=text_list,
        model=model,
        encoding_format="float",
    )
    outputs = []
    outputs.extend([e.embedding for e in response.data])
    return outputs


def load_demo_vectors(
        vector_db_name=vector_db_name,
        VECTOR_COLLECTION=VECTOR_COLLECTION,
        EMBEDDING_MODEL=EMBEDDING_MODEL):
    """ Load TEXTS and their embedding vectors. """    
    # clean slate
    res = vector_db[VECTOR_COLLECTION].delete_many({})
    print(f"Removed {res.deleted_count} documents in {vector_db_name}.{VECTOR_COLLECTION} before loading demo vectors")
    vectors = get_embeddings_vectors(TEXTS,model=EMBEDDING_MODEL)
    assert len(vectors) == len(TEXTS)
    docs = []
    docs.extend([{ "text": text, "embedding": vectors[ix] } for ix, text in enumerate(TEXTS)])
    res = vector_db[VECTOR_COLLECTION].insert_many(docs)
    assert len(res.inserted_ids) == len(TEXTS)
    print(f"Loaded {vector_db_name}.{VECTOR_COLLECTION} with {len(TEXTS)} demo vectors")


if __name__ == "__main__":

    load_demo_vectors()
