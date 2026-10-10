
import amrlib
import smatch


class AMRClient:
    def __init__(self):
        print("Loading AMR parser...")

        self.stog = amrlib.load_stog_model(
            device="cpu",
            batch_size=1,
            num_beams=4
        )

        print("AMR parser loaded successfully.")

    def parse(self, text):
        if not text or not text.strip():
            return None

        try:
            graphs = self.stog.parse_sents(
                [text],
                add_metadata=False,
                disable_progress=True
            )

            if not graphs:
                return None

            return graphs[0]

        except Exception as e:
            print("AMR parsing error:", e)
            return None

    def similarity(self, text_a, text_b):
        graph_a = self.parse(text_a)
        graph_b = self.parse(text_b)

        if graph_a is None or graph_b is None:
            return 0.0

        try:
            # Smatch returns:
            # matching triples, triples in A, triples in B
            match_count, count_a, count_b = smatch.get_amr_match(
                graph_a,
                graph_b
            )

            if count_a == 0 or count_b == 0:
                return 0.0

            precision = match_count / count_a
            recall = match_count / count_b

            if precision + recall == 0:
                return 0.0

            f1 = 2 * precision * recall / (precision + recall)

            # Numerical safety: similarity must be in [0, 1].
            return max(0.0, min(1.0, float(f1)))

        except Exception as e:
            print("Smatch scoring error:", e)
            return 0.0


if __name__ == "__main__":
    client = AMRClient()

    text_a = (
        "Janet has 16 eggs. "
        "She uses 3 eggs for breakfast. "
        "She uses 4 eggs for baking."
    )

    text_b = (
        "Janet starts with 16 eggs "
        "and uses 7 eggs."
    )

    unrelated_text = (
        "A train travels between two cities "
        "at a constant speed."
    )

    print("\nTesting identical responses...")
    score_identical = client.similarity(text_a, text_a)
    print(f"Identical-text Smatch: {score_identical:.4f}")

    print("\nTesting related responses...")
    score_related = client.similarity(text_a, text_b)
    print(f"Related-text Smatch: {score_related:.4f}")

    print("\nTesting unrelated responses...")
    score_unrelated = client.similarity(text_a, unrelated_text)
    print(f"Unrelated-text Smatch: {score_unrelated:.4f}")
