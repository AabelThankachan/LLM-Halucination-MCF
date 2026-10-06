from datasets import load_dataset

print("Downloading GSM8K...")

dataset = load_dataset("openai/gsm8k", "main")

print("\nDataset downloaded successfully!")
print(dataset)

print("\nTraining examples:", len(dataset["train"]))
print("Test examples:", len(dataset["test"]))

print("\nFirst question:")
print(dataset["test"][0]["question"])

print("\nCorrect answer:")
print(dataset["test"][0]["answer"])