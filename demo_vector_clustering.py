""" demo_vector_clustering.py

    S. Kurowski, July 2024

    Run this code for the entire demonstration.
    Configure CONFIGURE_VEC2TEXT = True to enable Method 2 vector-to-text reversal.

    Demo simple embedding vector clustering semantic structure discovery
    featuring OPTICS density clustering, MongoDB and OpenAI.

    Loads demo_db.demo_target_vectors with 26 texts and their vectors.

    Clusters, identifies centroids and generates their texts,
    charts the semantic clustering hierarchy tree png file,
    and writes the clustered semantic centroids to
    demo_db.demo_target_vectors_fitted_semantic_clusters.

    No guarantees or warranties are made in this demo.
"""

# pylint: disable-msg=C0103,C0301,W0621,W0718

import sys
import time
import ast
import numpy as np
from sklearn.cluster import OPTICS
import pymongo
import tiktoken
from openai import OpenAI

from load_vector_db import load_demo_vectors
from vector_cluster_hierarchy_chart import (
    decode_cluster_hierarchy,
    chart_cluster_hierarchy,
)

# Configure target vector db - local test Mongo server, or any valid user-auth'd URI etc.
VECTOR_MONGO_URI = "localhost:27017"
VECTOR_DB_NAME = "demo_db"

# Configure target MongoDB collection of embedded content vectors to analyze.
VECTOR_COLLECTION = "demo_target_vectors"

# Configure for each vector field in VECTOR_COLLECTION we want to cluster upon.
# Use its source text field name: embedding field name as a {key: value} pair.
# Use "text": "embedding" for demo vectors
V_EMBEDDINGS = {
    "text": "embedding",
}

# Configure namespace of the output clustered centroids data
FITTED_COLLECTION = VECTOR_COLLECTION + "_fitted_semantic_clusters"

# Configure enabling vec2text centroid means to text reversals
# for non-trivial cluster membership
CONFIGURE_VEC2TEXT = False  # True to enable
if CONFIGURE_VEC2TEXT:
    from reverse_vector_vec2text import reverse_embedding_vectors_to_texts

# Configure OpenAI completion model to use for Method 3 centroid texts merging
LLM_MODEL = "gpt-4o"
LLM_MODEL_TOKEN_BUDGET = 16000  # stick to 16 kT upper bound for LLM focus

# Configure embedding model used in VECTOR_COLLECTION vectors
# NOTE: must be "text-embedding-ada-002" for vec2text vector reversal
EMBEDDING_MODEL = "text-embedding-ada-002"  # "text-embedding-3-small"

# Configure grouping "granularity" of cluster vectors;
# each MINIMUM_VECTORS_PER_CLUSTER value is clustered separately.
# Use MINIMUM_VECTORS_PER_CLUSTER = [2] for demo vectors
MINIMUM_VECTORS_PER_CLUSTER = [2]  # [512,256,128,64,32,16]
print(f"Fitting min_vectors_per_clusters: {MINIMUM_VECTORS_PER_CLUSTER}")


mongo_client = pymongo.MongoClient(VECTOR_MONGO_URI)
vector_db = mongo_client[VECTOR_DB_NAME]

print(f"Writing cluster data to {VECTOR_DB_NAME}.{FITTED_COLLECTION}")
# clean slate for fitted output centroids
vector_db[FITTED_COLLECTION].delete_many({})

tokenizer = tiktoken.get_encoding("p50k_base")  # conservative token size
model_client = OpenAI()


def merge_centroid(texts: list[str], model: str = LLM_MODEL) -> tuple[str, list[str]]:
    """Call LLM to merge clustered texts into a centroid text (Method 3)."""

    def tiktoken_len(text):
        """Estimate text length in tokens."""
        return len(tokenizer.encode(text, disallowed_special=()))

    merge_prompt = """Expertly identify what's common between the provided similar statements and write in JSON key `MERGED_STATEMENT_PATTERNS` a unified detailed pattern statement that describes what's common to them all.
- Keep the merged output under 80 words. Write tersely, omit unnecessary words which do not reduce detail nor clarity of meaning.
- Do not enumerate how they differ.
- Use the `Occurrence Frequency` to weight the relative significance of each statement to help determine what's most important to merge into the `MERGED_STATEMENT_PATTERNS` output string.
- Write everything in a single sentence.
- Never include a title or descriptive prefix of the unified statement itself.
Output only in JSON similar to this: `{ "MERGED_STATEMENT_PATTERNS": "<< unified common pattern statement string, here >>" }`
    """
    # allow for ~2x reply length we expect typically
    # (max ~80 words (per prompt) * ~5 chars/word / ~4 chars/token => 100 tokens * 2x => 200 tokens)
    LLM_REPLY_TOKEN_RESERVE = 200

    # basic multi-"line" repeat compression as text : frequency table
    frequencies = {}
    for line in texts:
        if len(line) > 0:
            if line in frequencies:
                frequencies[line] += 1
            else:
                frequencies[line] = 1

    # NOTE: 'Occurrence Frequency' is a designed LLM
    #       merge_prompt focus/reference keyword
    frequency_table = ""
    for line, freq in frequencies.items():
        frequency_table += f"{line} : Occurrence Frequency = {freq}\n\n"

    model_tokens = tiktoken_len(merge_prompt) + LLM_REPLY_TOKEN_RESERVE
    L0 = len(frequency_table)
    while tiktoken_len(frequency_table) + model_tokens >= LLM_MODEL_TOKEN_BUDGET:
        # when frequency table is too large, lop off last ~1 token until better, and punt
        frequency_table = frequency_table[:-4]
    L1 = len(frequency_table)
    if L1 < L0:
        print(f"Trimmed {L0-L1} chars ~{(L0-L1)/4:.1f} tokens from LLM inputs")

    messages = [
        {"role": "system", "content": merge_prompt},
        {"role": "user", "content": f"COMMON_STATEMENTS:\n{frequency_table}"},
    ]
    while True:
        try:
            # set temperature and frequency_penalty
            # for dry, factual statements
            completion = model_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                frequency_penalty=0.5,
                max_tokens=LLM_REPLY_TOKEN_RESERVE,
                timeout=15,
            )
            if hasattr(completion.choices[0].message, "content"):
                break
        except Exception as e_:
            print(e_)
            time.sleep(5)

    reply = completion.choices[0].message.content

    # might need some JSON cleanup
    reply = reply.replace("\n\n---\n\n", "")
    reply = reply.replace("```", "")
    reply = reply.replace("json\n", "")

    json_reply = ast.literal_eval(reply)
    assert "MERGED_STATEMENT_PATTERNS" in json_reply
    return json_reply["MERGED_STATEMENT_PATTERNS"]


def cluster_vectors(min_vectors_per_cluster: int, vectors: np.ndarray) -> tuple:
    """Cluster the vectors using OPTICS."""
    print(f"Clustering vectors with minimum membership = {min_vectors_per_cluster} ...")
    np.seterr(divide="ignore", invalid="ignore")
    t0 = time.time()
    clustering = OPTICS(min_samples=min_vectors_per_cluster).fit(vectors)
    t1 = time.time()
    np.seterr(divide="warn", invalid="warn")
    labels = clustering.labels_
    ordering = clustering.ordering_
    core_dists = clustering.core_distances_[ordering]
    hierarchy = clustering.cluster_hierarchy_
    components = labels.max() + 1
    TF = (t1 - t0) / (x_size * x_size)
    print(f"Elapsed time (sec) = {t1-t0:.1f}, run_time_seconds( N ) = {TF:.6f} * N^2")
    return (clustering, labels, ordering, core_dists, hierarchy, components)


def centroid_vectors_from_clusters(
    min_vectors_per_cluster: int, clusters: list, y_size: int
) -> tuple:
    """Create a centroid mean embedding vector for each cluster found including their parents."""
    c_vectors = np.zeros((len(clusters), y_size), dtype=np.float64)
    c_vector_stderrs = np.zeros((len(clusters)), dtype=np.float64)
    n_fitted, n_parents_traversed = 0, 0
    for k, cluster in enumerate(clusters):
        assert cluster["_id"] == k
        if len(cluster["children"]) == 0:
            # child leaf node cluster, index-adjusted for parent count traversed
            rg_ix = np.argwhere(labels == k - n_parents_traversed).flatten()
            n_fitted += len(rg_ix)
        else:
            # parent branch node cluster members list is
            # assembled from its total child cluster members
            rg_ix = np.array([], dtype=np.int32)
            for child in cluster["children"]:
                rg_ix = np.append(rg_ix, clusters[child]["rg_ix"])
            n_parents_traversed += 1
        clusters[k]["rg_ix"] = rg_ix  # centroid member vectors index list
        c_vectors[k] = vectors[rg_ix].mean(axis=0)  # centroid mean vector
        c_vector_stderrs[k] = vectors[rg_ix].std()  # centroid stdev

    n_branched_parents = len(clusters) - components
    assert n_parents_traversed == n_branched_parents
    print(
        f"Fitted {components} centroid clusters and {n_branched_parents} parent clusters "
        f"having >= {min_vectors_per_cluster} vectors, totaling {n_fitted} vectors of size = {y_size}"
    )
    return (clusters, c_vectors, c_vector_stderrs)


def collect_centroid_member_data(
    cluster: dict, vectors: np.ndarray, texts: list[str]
) -> tuple:
    """Assemble the texts of centroid members. Optionally collect side_data_accumulated."""
    closest_texts, rejected_members = [], 0
    # example collection of non-vector field 'field_to_accumulate' into centroid
    # side_data_accumulated = []
    for iv in cluster["rg_ix"]:
        v, t = vectors[iv], texts[iv]
        # (hyper-)space Pythagorean vector separation
        # distance d from computed mean centroid
        dv = c_vectors[k] - v
        d = np.sqrt(np.sum(np.multiply(dv, dv)))
        if d > 3.0:
            # d > 3 sigmas, 99.7% confidence it's not close enough in vector space distance
            print(f'\tCentroid {k} rejected: vector distance = {d:.3f} > 3.0, "{t}"\n')
            rejected_members += 1
            continue
        if isinstance(t, list):
            # for arrays of strings, concatenate into a single string
            s = "; ".join(t)
        else:
            s = str(t)
        print(f'\tvector distance = {d:.3f}, "{t}"')
        # save the v_field_text of each vector to make a token-friendly
        # frequency table to ask the LLM to merge-summarize as a class.
        closest_texts.append(s)
        # example saving non-vector field 'field_to_accumulate' into centroid
        # side_data_accumulated.append(side_data_accumulation[iv])

    closest_texts.sort()
    return (
        rejected_members,
        closest_texts,
        # side_data_accumulated
    )


##
## SEMANTIC VECTOR CLUSTERING DEMO
## FEATURING OPTICS, MONGODB, OPENAI, VEC2TEXT
##

# load the demo_db with test texts and their vectors
# comment this out to use your own existing vector db collection
load_demo_vectors(VECTOR_DB_NAME, VECTOR_COLLECTION, EMBEDDING_MODEL)

# iterate over embeddings to cluster
for v_field_text, v_field_embed in V_EMBEDDINGS.items():

    # load vectors embedding the text field
    print(f"Fetching {VECTOR_DB_NAME}.{VECTOR_COLLECTION}.{v_field_embed} vectors ...")
    raw_vectors = list(
        vector_db[VECTOR_COLLECTION].find(
            {
                # we don't need to worry about this for demo vectors
                # but it becomes important for any known partial vector set coverage
                # v_field_text: {'$exists':1}, v_field_embed: {'$exists':1},
                # 'field_to_accumulate': {'$exists:1}, # example non-vector field 'field_to_accumulate' to collect in centroids
            },
            {
                v_field_text: 1,
                v_field_embed: 1,
                # example non-vector field 'field_to_accumulate' to collect in centroids
                #'field_to_accumulate': 1,
                "_id": 0,
            },
        )
    )
    assert len(raw_vectors) > 0

    vectors, texts = [], []
    texts.extend([v[v_field_text] for v in raw_vectors])
    vectors.extend([v[v_field_embed] for v in raw_vectors])
    vectors = np.array(vectors, dtype=np.float32)
    x_size = len(vectors)
    y_size = len(vectors[0])

    # example collection of non-vector field 'field_to_accumulate' for centroids
    # side_data_accumulation = []
    # side_data_accumulation.extend([v['field_to_accumulate'] for v in raw_vectors])
    del raw_vectors

    print(
        f"Fetched {x_size} vectors each length {y_size}, total size MB = {sys.getsizeof(vectors)/1024/1024:.3f}"
    )

    # loop over each clustering "granularity" configured;
    # cluster semantic centroids, generate their texts
    for min_vectors_per_cluster in MINIMUM_VECTORS_PER_CLUSTER:

        (clustering, labels, ordering, core_dists, hierarchy, components) = (
            cluster_vectors(min_vectors_per_cluster, vectors)
        )
        if components <= 2:
            print(
                "\nNot enough clusters found as configured. "
                f"Use min_vectors_per_cluster < {min_vectors_per_cluster}\n"
            )
            continue

        # show some stats if not too long
        if len(labels) < 256:
            print(f"\tvector index labels: {labels}")
            print(f"\tordering: {ordering}")
            print(f"\tcore_dists: {core_dists}")
        print(f"\thierarchy: {hierarchy}")

        # convert OPTICS hierarchy into labeled directed node graph dict
        clusters = decode_cluster_hierarchy(hierarchy)

        # save png image of labeled semantic structure
        fn = f"demo_vectors_clustered_{v_field_text}_{min_vectors_per_cluster}minPerCluster"
        print(f"Writing chart of clustered Semantic Structure to {fn}.png")
        chart_cluster_hierarchy(clusters=clusters, save_file_name=fn)

        # show decoded semantic clusters data
        for c in clusters:
            print(c)

        # compute mean vector centroids from clusters
        (clusters, c_vectors, c_vector_stderrs) = centroid_vectors_from_clusters(
            min_vectors_per_cluster, clusters, y_size
        )

        # generate centroid texts for each numbered cluster with their letter-labeled parents;
        # optionally accumulate other centroid metadata in side_data_accumulated
        for cluster in clusters:
            k = cluster["_id"]
            members = len(cluster["rg_ix"])
            if c_vector_stderrs[k] > 3.0:
                print(
                    f"Rejected vector cluster {cluster['node_label']} at stderr = {c_vector_stderrs[k]}\n"
                )
                continue

            print(
                f"{VECTOR_DB_NAME}.{VECTOR_COLLECTION} vector cluster centroid {cluster['node_label']}, member_vectors = {members} ({100*members/x_size:.3f}% of population):"
            )
            (
                rejected_members,
                closest_texts,
                # side_data_accumulated
            ) = collect_centroid_member_data(cluster, vectors, texts)
            if len(closest_texts) == 0:
                print(
                    f"\tCentroid {k} has no vectors having text data to merge/reverse\n"
                )
                continue

            # Method 3 - LLM-merge texts as centroid
            c_vector_text_LLM = merge_centroid(closest_texts)

            # Method 2 - vec2text reverse text from cluster's vector mean centroid
            c_vector_text_v2t = None
            if CONFIGURE_VEC2TEXT:
                c_vector_text_v2t = reverse_embedding_vectors_to_texts([c_vectors[k]])[
                    0
                ]

            # assemble centroid doc and insert
            centroid_doc = {
                "metadata": f"clustered {v_field_text} centroid",
                "min_vectors_per_cluster": min_vectors_per_cluster,
                "centroid_label": cluster["node_label"],
                "centroid_std_error": c_vector_stderrs[k],
                f"centroid_{v_field_text}_LLM_merged": c_vector_text_LLM,  # Method 3
                f"centroid_{v_field_text}_vec2text": c_vector_text_v2t,  # Method 2
                "clustered_members": members,
                "rejected_members": rejected_members,
                "centroid_texts": closest_texts,
                "centroid_embedding": list(c_vectors[k]),
                # example non-vector field 'field_to_accumulate' as-accumulated for this centroid
                # "centroid_field_to_accumulate": side_data_accumulated,
            }
            if not CONFIGURE_VEC2TEXT:
                del centroid_doc[f"centroid_{v_field_text}_vec2text"]

            res = vector_db[FITTED_COLLECTION].insert_one(centroid_doc)
            assert res.inserted_id
            print(
                f"Stored clustered vector centroid {cluster['node_label']}: {c_vector_text_LLM}\n"
            )

print("done")
