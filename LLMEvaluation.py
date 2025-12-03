import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import re
from collections import Counter
import numpy as np

COMPREHENSIVE_TESTS = {
    "accuracy": [
        {
            "query": "What is the maximum timeout for AWS Lambda?",
            "context": "Lambda timeout: 1 second to 15 minutes (900 seconds). Default: 3 seconds.",
            "required": ["15 minutes", "900 seconds"],
            "wrong": ["30 minutes", "60 minutes"]
        }
    ],
    "instruction_following": [
        {
            "query": "List the steps to create an S3 bucket",
            "context": "To create bucket: 1. Open S3 console 2. Click Create bucket 3. Enter name 4. Select region 5. Click Create",
            "required_format": "numbered_list",
            "should_have": ["1.", "2.", "3."],
            "should_not_have": ["bullet points", "- "]
        }
    ],
    "conciseness": [
        {
            "query": "What is S3?",
            "context": "S3 is object storage service by AWS. Stores data as objects in buckets. Highly scalable and durable.",
            "ideal_length": (20, 50),
            "penalty_if_longer": True
        }
    ],
    "context_adherence": [
        {
            "query": "What are S3 storage classes?",
            "context": "S3 has Standard and Glacier classes. Standard for frequent access. Glacier for archival.",
            "must_not_mention": ["Intelligent-Tiering", "One Zone-IA", "Deep Archive"],
            "must_mention": ["Standard", "Glacier"]
        }
    ],

    "complex_reasoning": [
        {
            "query": "If I need to store 1TB of data accessed once per month, which is cheaper: S3 Standard at $0.023/GB or Glacier at $0.004/GB?",
            "context": "S3 Standard: $0.023 per GB per month. S3 Glacier: $0.004 per GB per month. 1 TB = 1024 GB.",
            "requires_calculation": True,
            "correct_answer": "Glacier",
            "should_show_math": True
        }
    ],
}
MODELS = [
    ("Qwen 2.5 7B", "Qwen/Qwen2.5-7B-Instruct"),
    ("Llama 3.1 8B", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    ("Mistral 7B", "mistralai/Mistral-7B-Instruct-v0.3"),
    ("Gemma 2 9B", "google/gemma-2-9b-it"),
]
class ComprehensiveEvaluator:
    """Multi-dimensional model evaluation."""

    def evaluate_accuracy(self, answer, test):
        """Measure factual accuracy."""
        answer_lower = answer.lower()
        correct = sum(1 for fact in test["required"] if fact.lower() in answer_lower)
        wrong = sum(1 for fact in test["wrong"] if fact.lower() in answer_lower)

        score = (correct / len(test["required"])) * 100 - (wrong * 30)
        return max(0, score), {
            "correct": correct,
            "wrong": wrong,
            "total": len(test["required"])
        }

    def evaluate_instruction_following(self, answer, test):
        """Check if model follows format instructions."""
        score = 100
        issues = []

        if test["required_format"] == "numbered_list":
            has_numbers = any(item in answer for item in test["should_have"])
            if not has_numbers:
                score -= 50
                issues.append("Missing numbered list")

            has_bullets = any(item in answer for item in test.get("should_not_have", []))
            if has_bullets:
                score -= 30
                issues.append("Used bullets instead of numbers")

        return max(0, score), {"issues": issues}

    def evaluate_conciseness(self, answer, test):
        """Measure if answer is appropriately concise."""
        word_count = len(answer.split())
        min_words, max_words = test["ideal_length"]

        if min_words <= word_count <= max_words:
            score = 100
            verdict = "perfect"
        elif word_count < min_words:
            score = 70
            verdict = "too_short"
        elif word_count <= max_words * 1.5:
            score = 80
            verdict = "slightly_verbose"
        else:
            score = 50
            verdict = "too_verbose"

        return score, {
            "word_count": word_count,
            "ideal_range": test["ideal_length"],
            "verdict": verdict
        }

    def evaluate_context_adherence(self, answer, test):
        """Check if model stays within provided context."""
        answer_lower = answer.lower()
        score = 100
        violations = []

        for item in test.get("must_not_mention", []):
            if item.lower() in answer_lower:
                score -= 30
                violations.append(f"Mentioned '{item}' (not in context)")

        missing = []
        for item in test.get("must_mention", []):
            if item.lower() not in answer_lower:
                score -= 20
                missing.append(item)

        return max(0, score), {
            "violations": violations,
            "missing": missing
        }

    def evaluate_reasoning(self, answer, test):
        """Evaluate complex reasoning ability."""
        answer_lower = answer.lower()
        score = 0

        if test["correct_answer"].lower() in answer_lower:
            score += 60

        if test.get("should_show_math"):
            has_numbers = bool(re.search(r'\d+', answer))
            has_calculation = any(word in answer_lower for word in ["cheaper", "cost", "price", "calculate"])
            if has_numbers and has_calculation:
                score += 40

        return score, {
            "correct_conclusion": test["correct_answer"].lower() in answer_lower,
            "shows_reasoning": has_numbers and has_calculation if test.get("should_show_math") else None
        }

    def measure_performance(self, gen_time, input_tokens, output_tokens):
        """Measure generation performance."""
        tokens_per_second = output_tokens / gen_time if gen_time > 0 else 0

        if tokens_per_second > 50:
            speed_score = 100
        elif tokens_per_second > 30:
            speed_score = 80
        elif tokens_per_second > 20:
            speed_score = 60
        else:
            speed_score = 40

        return {
            "gen_time": gen_time,
            "tokens_per_sec": tokens_per_second,
            "speed_score": speed_score
        }


def comprehensive_test(model_name, model_id):
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE EVALUATION: {model_name}")
    print('='*70)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated(0) / 1e9
            print(f"Loaded ({gpu_mem:.1f}GB GPU memory)")

        evaluator = ComprehensiveEvaluator()
        results = {}
        print("1. ACCURACY EVALUATION")
        for test in COMPREHENSIVE_TESTS["accuracy"]:
            answer, gen_time, input_tokens, output_tokens = generate_answer(
                model, tokenizer, test["query"], test["context"]
            )
            score, details = evaluator.evaluate_accuracy(answer, test)
            print(f"Query: {test['query']}")
            print(f"Score: {score:.0f}/100")
            print(f"Details: {details}")
            results["accuracy"] = {"score": score, "details": details, "gen_time": gen_time}

        print("2. INSTRUCTION FOLLOWING")
        for test in COMPREHENSIVE_TESTS["instruction_following"]:
            answer, gen_time, _, _ = generate_answer(
                model, tokenizer, test["query"], test["context"]
            )
            score, details = evaluator.evaluate_instruction_following(answer, test)
            print(f"Query: {test['query']}")
            print(f"Score: {score:.0f}/100")
            print(f"Issues: {details['issues'] if details['issues'] else 'None'}")
            results["instruction_following"] = {"score": score, "details": details, "gen_time": gen_time}

        print("3. CONCISENESS EVALUATION")
        for test in COMPREHENSIVE_TESTS["conciseness"]:
            answer, gen_time, _, _ = generate_answer(
                model, tokenizer, test["query"], test["context"]
            )
            score, details = evaluator.evaluate_conciseness(answer, test)
            print(f"Query: {test['query']}")
            print(f"Score: {score:.0f}/100")
            print(f"Word count: {details['word_count']} (ideal: {details['ideal_range'][0]}-{details['ideal_range'][1]})")
            print(f"Verdict: {details['verdict']}")
            results["conciseness"] = {"score": score, "details": details, "gen_time": gen_time}
        print("4. CONTEXT ADHERENCE")
        for test in COMPREHENSIVE_TESTS["context_adherence"]:
            answer, gen_time, _, _ = generate_answer(
                model, tokenizer, test["query"], test["context"]
            )
            score, details = evaluator.evaluate_context_adherence(answer, test)
            print(f"Query: {test['query']}")
            print(f"Score: {score:.0f}/100")
            if details["violations"]:
                print(f" Violations: {details['violations']}")
            results["context_adherence"] = {"score": score, "details": details, "gen_time": gen_time}

        print("5. COMPLEX REASONING")
        for test in COMPREHENSIVE_TESTS["complex_reasoning"]:
            answer, gen_time, _, _ = generate_answer(
                model, tokenizer, test["query"], test["context"]
            )
            score, details = evaluator.evaluate_reasoning(answer, test)
            print(f"Query: {test['query'][:60]}...")
            print(f"Score: {score:.0f}/100")
            print(f"Correct conclusion: {details['correct_conclusion']}")
            results["reasoning"] = {"score": score, "details": details, "gen_time": gen_time}
        print("6. PERFORMANCE METRICS")
        avg_time = np.mean([r.get("gen_time", 5) for r in results.values() if "gen_time" in r]) if results else 5
        perf = evaluator.measure_performance(avg_time, 100, 100)
        print(f"Avg generation time: {perf['gen_time']:.2f}s")
        print(f"Tokens per second: {perf['tokens_per_sec']:.1f}")
        print(f"Speed score: {perf['speed_score']:.0f}/100")

        # Store with consistent 'score' key like other dimensions
        results["performance"] = {
            "score": perf['speed_score'],
            "details": {
                "gen_time": perf['gen_time'],
                "tokens_per_sec": perf['tokens_per_sec']
            }
        }
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_answer(model, tokenizer, query, context):
    """Generate answer and measure performance."""
    messages = [
        {"role": "system", "content": f"Answer based on context:\n\n{context}"},
        {"role": "user", "content": query}
    ]

    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        # If system role not supported (like Gemma), combine into user message
        if "system" in str(e).lower():
            messages = [
                {"role": "user", "content": f"Answer based on the following context:\n\n{context}\n\nQuestion: {query}"}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            raise e

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_time = time.time() - start

    answer = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    input_tokens = inputs['input_ids'].shape[1]
    output_tokens = outputs.shape[1] - input_tokens

    return answer, gen_time, input_tokens, output_tokens


if __name__ == "__main__":

    all_results = {}
    for model_name, model_id in MODELS:
        result = comprehensive_test(model_name, model_id)
        if result:
            all_results[model_name] = result

    print("  MODEL COMPARISON - 6 DIMENSIONS")

    model_names = list(all_results.keys())
    print(f"\n{'Dimension':<25}", end="")
    for name in model_names:
        print(f"{name:<18}", end="")
    print()
    dimensions = ["accuracy", "instruction_following", "context_adherence",
                  "conciseness", "reasoning", "performance"]

    for dim in dimensions:
        print(f"{dim.replace('_', ' ').title():<25}", end="")
        for name in model_names:
            score = all_results.get(name, {}).get(dim, {}).get("score", 0)
            print(f"{score:>6.0f}/100       ", end="")
        print()
