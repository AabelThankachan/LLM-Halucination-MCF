import amrlib

print("Loading AMR parser...")

stog = amrlib.load_stog_model(
    device="cpu",
    batch_size=1,
    num_beams=4
)

print("AMR parser loaded successfully!")

sentences = [
    "Janet has 16 eggs.",
    "She uses 3 eggs for breakfast.",
    "She uses 4 eggs for baking.",
    "She sells the remaining eggs for 2 dollars each."
]

graphs = stog.parse_sents(sentences)

for i, graph in enumerate(graphs, 1):
    print(f"\n===== SENTENCE {i} =====")
    print(sentences[i - 1])

    print("\n===== AMR GRAPH =====")
    print(graph)