import ollama
import numpy as np


MODEL_NAME = "nomic-embed-text"


def get_embedding(text):
    """
    Generate an embedding for the supplied text
    using the local nomic-embed-text model.
    """

    response = ollama.embed(
        model=MODEL_NAME,
        input=text
    )

    return np.array(response["embeddings"][0])


def cosine_similarity(vector_a, vector_b):
    """
    Calculate cosine similarity between two vectors.
    """

    denominator = (
        np.linalg.norm(vector_a)
        * np.linalg.norm(vector_b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(vector_a, vector_b) / denominator
    )


if __name__ == "__main__":

    text1 = """
    Janet has 16 eggs. She uses 3 for breakfast
    and 4 for baking, leaving 9 eggs to sell.
    """

    text2 = """
    Janet sells 9 eggs after using 7 eggs,
    and earns money from selling them.
    """

    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)

    similarity = cosine_similarity(
        embedding1,
        embedding2
    )

    print("Embedding dimension:", len(embedding1))
    print("Cosine similarity:", similarity)