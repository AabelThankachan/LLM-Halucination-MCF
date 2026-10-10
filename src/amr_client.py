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
            precision, recall, f_score = smatch.get_amr_match(
                graph_a,
                graph_b
            )

            return float(f_score)

        except Exception as e:
            print("Smatch error:", e)
            return 0.0


if __name__ == "__main__":

    client = AMRClient()

    text1 = """
    Janet has 16 eggs.
    She uses 3 eggs for breakfast.
    She uses 4 eggs for baking.
    """

    text2 = """
    Janet starts with 16 eggs and uses 7 eggs.
    """

    print("\n===== GRAPH 1 =====")
    print(client.parse(text1))

    print("\n===== GRAPH 2 =====")
    print(client.parse(text2))

    score = client.similarity(text1, text2)

    print("\n===== SMATCH SCORE =====")
    print(score)