import torch
import warnings
import numpy as np
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from backend import RAGBackend

warnings.filterwarnings("ignore")

# 1. Setup
print(" Loading Evaluator...")
judge_model = SentenceTransformer("BAAI/bge-m3")
rag_system = RAGBackend()
rag_system.load_resources()

# 2. Golden Dataset
test_questions = [
    # Q1: Compute (Lambda) - Fact
    "What is the maximum execution time for an AWS Lambda function?",

    # Q2: Database (DynamoDB) - Concept
    "What is the difference between a Scan and a Query in DynamoDB?",

    # Q3: Storage (S3) - Procedure
    "How do I create an S3 bucket?",

    # Q4: Compute (EC2) - Fact (Tests your EC2 Ghost Fix)
    "What is the difference between On-Demand and Spot Instances?",

    # Q5: Security (IAM) - Concept
    "What is the AWS root user?"
]

# The "Answer Key"
ground_truths = [
    # A1
    "The maximum execution time for an AWS Lambda function is 15 minutes or 900 seconds.",

    # A2
    "A Query finds items based on primary key values and is efficient. A Scan reads every item in the table and is slower and consumes more throughput.",

    # A3
    "To create an S3 bucket, use the AWS Console, click Create Bucket, choose a unique name, and select a region.",

    # A4
    "On-Demand instances are fixed rate by the second with no commitment. Spot instances use spare EC2 capacity for up to 90% discounts but can be interrupted.",

    # A5
    "The root user is the identity created when you first create your AWS account. It has complete, unrestricted access to all resources and billing."
]

print("\n Running Semantic Evaluation...")
scores = []

for i, q in enumerate(test_questions):
    print(f"   Asking: {q}")
    response, source_docs = rag_system.generate_answer(q)

    # Combine all retrieved text into one big string context
    retrieved_text = " ".join([doc.page_content for doc in source_docs])

    # 1. Vectorize the Ground Truth (The Ideal Answer)
    truth_vec = judge_model.encode([ground_truths[i]])

    # 2. Vectorize the Retrieved Context (What your DB found)
    context_vec = judge_model.encode([retrieved_text])

    # 3. Calculate Similarity (0 to 1)
    similarity_score = cosine_similarity(truth_vec, context_vec)[0][0]
    scores.append(similarity_score)

    status = "PASS" if similarity_score > 0.55 else "FAIL"
    print(f"   -> Semantic Score: {similarity_score:.4f} | {status}")
    print(f"   -> Model Answer: {response[:100]}...")
    print("-" * 50)

avg_score = np.mean(scores)
print("\n" + "=" * 40)
print(f"FINAL SEMANTIC SCORE")
print("=" * 40)
print(f"Average Similarity: {avg_score:.4f} (Target: >0.60)")
if avg_score > 0.6:
    print("RESULT: SYSTEM PASSED")
else:
    print("RESULT: SYSTEM NEEDS TUNING")
print("=" * 40)
