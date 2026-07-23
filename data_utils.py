# data_utils.py

import os
import json
from datasets import load_dataset
from utils import create_demo_text, extract_supporting_facts, create_summarization_demo_text


def load_dataset_first_available(*dataset_specs):
    """
    Written by Rishikesh Raj: try multiple Hugging Face dataset identifiers.

    OLD CODE:
    # dataset_dict = load_dataset("single/name", "single_config")
    #
    # That can fail when a dataset moves namespace or a mirror exposes a
    # slightly different config name.

    NEW CODE:
    Keep fallbacks close to each loader so experiments fail only after all
    known compatible dataset IDs have been tried.
    """
    errors = []
    for spec in dataset_specs:
        if isinstance(spec, str):
            args = (spec,)
        else:
            args = tuple(spec)
        try:
            return load_dataset(*args)
        except Exception as exc:
            errors.append(f"{args}: {exc}")
    raise RuntimeError("All dataset loaders failed:\n" + "\n".join(errors))


def _limit_dataset(dataset, default_size: int, args):
    max_examples = getattr(args, "max_examples", None)
    size = default_size if max_examples is None else min(int(max_examples), default_size)
    size = min(size, len(dataset))
    return dataset.select(range(size))


def load_data_and_create_prompts(args):
    """
    Loads the specified dataset and creates prompts (with and without context)
    along with the answers/summaries.
    Returns: (prompts_with_context, prompts_without_context, answers/summaries)
    """
    task_type = getattr(args, 'task_type', 'qa')  # Default to 'qa' for backward compatibility
    
    if task_type == 'summarization':
        # Summarization datasets
        if args.dataset == 'cnn_dailymail':
            dataset_dict = load_dataset("cnn_dailymail", "3.0.0")
            # Problem 11 fix - modified by rishikesh.
            # OLD CODE:
            # validation_dataset = dataset_dict["validation"].select(range(50))
            #
            # NEW CODE:
            # Respect --max-examples for summarization also, so CNN/DailyMail,
            # XSum, and SAMSum can run separate small/large experiments fairly.
            validation_dataset = _limit_dataset(dataset_dict["validation"], 50, args)
            prompts_with_context, summaries = create_prompts_from_cnn_dailymail(validation_dataset, include_context=True)
            prompts_without_context, _ = create_prompts_from_cnn_dailymail(validation_dataset, include_context=False)
        
        elif args.dataset == 'xsum':
            dataset_dict = load_dataset("xsum")
            # Problem 11 fix - modified by rishikesh.
            # OLD CODE:
            # validation_dataset = dataset_dict["validation"].select(range(500))
            #
            # NEW CODE:
            validation_dataset = _limit_dataset(dataset_dict["validation"], 500, args)
            prompts_with_context, summaries = create_prompts_from_xsum(validation_dataset, include_context=True)
            prompts_without_context, _ = create_prompts_from_xsum(validation_dataset, include_context=False)
        
        elif args.dataset == 'samsum':
            dataset_dict = load_dataset("samsum")
            # Problem 11 fix - modified by rishikesh.
            # OLD CODE:
            # validation_dataset = dataset_dict["validation"].select(range(500))
            #
            # NEW CODE:
            validation_dataset = _limit_dataset(dataset_dict["validation"], 500, args)
            prompts_with_context, summaries = create_prompts_from_samsum(validation_dataset, include_context=True)
            prompts_without_context, _ = create_prompts_from_samsum(validation_dataset, include_context=False)
        
        else:
            raise ValueError(f"Invalid summarization dataset name. Choose from 'cnn_dailymail', 'xsum', or 'samsum'.")
        
        return prompts_with_context, prompts_without_context, summaries
    
    else:
        # QA datasets (original functionality)
        if args.dataset == 'hotpot_qa':
            dataset_dict = load_dataset("hotpot_qa", "distractor")
            validation_dataset = _limit_dataset(dataset_dict["validation"], 1000, args)
            prompts_with_context, answers = create_prompts_from_hotpot(validation_dataset, include_context=True)
            prompts_without_context, _ = create_prompts_from_hotpot(validation_dataset, include_context=False)

        elif args.dataset == 'squad':
            dataset_dict = load_dataset("rajpurkar/squad", "plain_text")
            validation_dataset = _limit_dataset(dataset_dict["validation"], 500, args)
            prompts_with_context, answers = create_prompts_from_squad(validation_dataset, include_context=True)
            prompts_without_context, _ = create_prompts_from_squad(validation_dataset, include_context=False)

        # Problem 4 / 5 fix - modified by rishikesh.
        # OLD CODE:
        # else:
        #     raise ValueError("Invalid dataset name. Choose from 'hotpot_qa', 'squad', or 'strategyqa'.")
        #
        # NEW CODE:
        # Add SQuAD v2 and hallucination-specific QA datasets requested by the
        # PDF so the evaluation runner does not fail before experiments start.
        elif args.dataset == 'squad_v2':
            dataset_dict = load_dataset("rajpurkar/squad_v2")
            validation_dataset = _limit_dataset(dataset_dict["validation"], 500, args)
            prompts_with_context, answers = create_prompts_from_squad(validation_dataset, include_context=True)
            prompts_without_context, _ = create_prompts_from_squad(validation_dataset, include_context=False)

        elif args.dataset == 'boolq':
            dataset_dict = load_dataset("google/boolq")
            validation_dataset = _limit_dataset(dataset_dict["validation"], 500, args)
            prompts_with_context, answers = create_prompts_from_boolq(validation_dataset, include_context=True)
            prompts_without_context, _ = create_prompts_from_boolq(validation_dataset, include_context=False)

        elif args.dataset == 'triviaqa':
            dataset_dict = load_dataset("trivia_qa", "rc.nocontext")
            validation_dataset = _limit_dataset(dataset_dict["validation"], 500, args)
            prompts_with_context, answers = create_prompts_from_triviaqa(validation_dataset, include_context=True)
            prompts_without_context, _ = create_prompts_from_triviaqa(validation_dataset, include_context=False)

        elif args.dataset == 'natural_questions':
            dataset_dict = load_dataset("google-research-datasets/natural_questions", "default")
            validation_dataset = _limit_dataset(dataset_dict["validation"], 250, args)
            prompts_with_context, answers = create_prompts_from_natural_questions(validation_dataset, include_context=True)
            prompts_without_context, _ = create_prompts_from_natural_questions(validation_dataset, include_context=False)

        elif args.dataset == 'openbookqa':
            dataset_dict = load_dataset("allenai/openbookqa", "additional")
            validation_dataset = _limit_dataset(dataset_dict["validation"], 500, args)
            prompts_with_context, answers = create_prompts_from_openbookqa(validation_dataset, include_context=True)
            prompts_without_context, _ = create_prompts_from_openbookqa(validation_dataset, include_context=False)

        elif args.dataset == 'arc_challenge':
            dataset_dict = load_dataset("allenai/ai2_arc", "ARC-Challenge")
            validation_dataset = _limit_dataset(dataset_dict["validation"], 500, args)
            prompts_with_context, answers = create_prompts_from_multiple_choice(validation_dataset, include_context=True)
            prompts_without_context, _ = create_prompts_from_multiple_choice(validation_dataset, include_context=False)

        elif args.dataset == 'commonsense_qa':
            dataset_dict = load_dataset("tau/commonsense_qa")
            validation_dataset = _limit_dataset(dataset_dict["validation"], 500, args)
            prompts_with_context, answers = create_prompts_from_multiple_choice(validation_dataset, include_context=True)
            prompts_without_context, _ = create_prompts_from_multiple_choice(validation_dataset, include_context=False)

        elif args.dataset == 'truthfulqa':
            dataset_dict = load_dataset_first_available(
                ("truthful_qa", "generation"),
                ("truthfulqa/truthful_qa", "generation"),
            )
            validation_dataset = _limit_dataset(dataset_dict["validation"], 500, args)
            prompts_with_context, answers = create_prompts_from_truthfulqa(validation_dataset, include_context=True)
            prompts_without_context, _ = create_prompts_from_truthfulqa(validation_dataset, include_context=False)

        elif args.dataset == 'fever':
            dataset_dict = load_dataset_first_available(
                ("fever", "v1.0"),
                "fever",
            )
            validation_dataset = _limit_dataset(dataset_dict["labelled_dev"], 500, args)
            prompts_with_context, answers = create_prompts_from_fever(validation_dataset, include_context=True)
            prompts_without_context, _ = create_prompts_from_fever(validation_dataset, include_context=False)

        elif args.dataset == 'halueval':
            dataset_dict = load_dataset_first_available(
                ("pminervini/HaluEval", "qa"),
                ("HaluEval/HaluEval", "qa"),
                "pminervini/HaluEval",
            )
            split_name = "data" if "data" in dataset_dict else list(dataset_dict.keys())[0]
            validation_dataset = _limit_dataset(dataset_dict[split_name], 500, args)
            prompts_with_context, answers = create_prompts_from_halueval(validation_dataset, include_context=True)
            prompts_without_context, _ = create_prompts_from_halueval(validation_dataset, include_context=False)

        elif args.dataset == 'strategyqa':
            # StrategyQA: load from a local JSON file only (no HTTP calls inside the code).
            # Preferred path order: explicit --strategyqa-path, then default 'data/strategyqa/dev.json'
            # from the official repo: https://github.com/eladsegal/strategyqa/tree/main/data/strategyqa
            data_list = None
            last_err = None

            # 1) Explicit path from arguments
            local_path = getattr(args, "strategyqa_path", None)
            if not local_path:
                # 2) Default conventional location
                local_path = os.path.join("data", "strategyqa", "dev.json")

            try:
                if not os.path.exists(local_path):
                    raise FileNotFoundError(
                        f"StrategyQA file not found at '{local_path}'. "
                        "Download 'dev.json' from the official repo "
                        "https://github.com/eladsegal/strategyqa (data/strategyqa/dev.json) "
                        "and place it there, or pass --strategyqa-path to its location."
                    )
                with open(local_path, "r", encoding="utf-8") as f:
                    data_list = json.load(f)
            except Exception as e:
                last_err = e
                data_list = None

            if data_list is None:
                raise ValueError(
                    f"StrategyQA failed to load from local file. Last error: {last_err}"
                )

            # Normalize possible formats: expect a list of QA dicts
            if isinstance(data_list, dict):
                # Common keys in StrategyQA repo
                for key in ["data", "examples", "questions"]:
                    if key in data_list:
                        data_list = data_list[key]
                        break

            if not isinstance(data_list, list):
                raise ValueError("StrategyQA JSON format not understood: expected a list of examples.")

            # Use up to 400 examples for evaluation
            eval_data = data_list[:400]
            prompts_with_context, answers = create_prompts_from_strategyqa(eval_data, include_context=True)
            prompts_without_context, _ = create_prompts_from_strategyqa(eval_data, include_context=False)

        else:
            raise ValueError(
                "Invalid dataset name. Choose from 'hotpot_qa', 'squad', 'squad_v2', "
                "'strategyqa', 'boolq', 'triviaqa', 'natural_questions', 'openbookqa', "
                "'arc_challenge', 'commonsense_qa', 'truthfulqa', 'fever', or 'halueval'."
            )

        return prompts_with_context, prompts_without_context, answers


def create_prompts_from_hotpot(dataset, include_context: bool):
    """
    Creates prompts and answers for HotpotQA.
    If include_context=True, attaches supporting facts to the prompts.
    """
    prompts = []
    answers = []

    for item in dataset:
        question = item['question']
        answer = item['answer']
        context = item.get('context', {})
        supporting_facts = item.get('supporting_facts', {})

        # Build prompt with an explicit instruction to return only the short answer span.
        instruction = create_demo_text() + " Answer with only the short answer (a single word or short phrase), no explanation.\n"
        if include_context:
            facts = extract_supporting_facts(context, supporting_facts)
            if facts:
                facts_str = ' '.join(facts)
                prompt = f"{instruction}Supporting information: {facts_str}\n\nQ: {question}\nA: "
            else:
                prompt = f"{instruction}\n\nQ: {question}\nA: "
        else:
            prompt = f"{instruction}\n\nQ: {question}\nA: "

        prompts.append(prompt)
        answers.append(answer)

    return prompts, answers


def create_prompts_from_squad(dataset, include_context: bool):
    """
    Creates prompts and answers for SQuAD.
    If include_context=True, attaches the paragraph context to the prompts.
    """
    prompts = []
    answers = []

    for item in dataset:
        question = item['question']
        # OLD CODE:
        # answer = item['answers']['text'][0]
        #
        # NEW CODE by Rishikesh Raj - modified by rishikesh:
        # Keep all SQuAD aliases. Exact Match should compare against every
        # accepted answer, not only the first string in the dataset.
        answer = item.get('answers', {}).get('text', [])
        if not answer:
            answer = ["unanswerable"]
        context = item['context']

        instruction = create_demo_text()
        if include_context:
            prompt = f"{instruction}Supporting information: {context}\n\nQ: {question}\nA: "
        else:
            prompt = f"{instruction}\n\nQ: {question}\nA: "

        prompts.append(prompt)
        answers.append(answer)

    return prompts, answers


def create_prompts_from_boolq(dataset, include_context: bool):
    prompts, answers = [], []
    for item in dataset:
        question = item.get("question", "")
        passage = item.get("passage", "")
        answer = "yes" if bool(item.get("answer", False)) else "no"
        instruction = create_demo_text()
        prompt = (
            f"{instruction}Supporting information: {passage}\n\nQ: {question}\nA: "
            if include_context else f"{instruction}\n\nQ: {question}\nA: "
        )
        prompts.append(prompt)
        answers.append(answer)
    return prompts, answers


def create_prompts_from_triviaqa(dataset, include_context: bool):
    prompts, answers = [], []
    for item in dataset:
        question = item.get("question", "")
        answer_obj = item.get("answer", {})
        aliases = answer_obj.get("aliases", []) if isinstance(answer_obj, dict) else [answer_obj]
        context = " ".join(item.get("search_results", {}).get("search_context", [])[:3]) if isinstance(item.get("search_results"), dict) else ""
        instruction = create_demo_text()
        prompt = (
            f"{instruction}Supporting information: {context}\n\nQ: {question}\nA: "
            if include_context and context else f"{instruction}\n\nQ: {question}\nA: "
        )
        prompts.append(prompt)
        answers.append(aliases)
    return prompts, answers


def create_prompts_from_natural_questions(dataset, include_context: bool):
    prompts, answers = [], []
    for item in dataset:
        # Problem 4 / 5 fix - modified by rishikesh.
        # OLD CODE:
        # question = item.get("question", {}).get("text", item.get("question", ""))
        # annotations = item.get("annotations", {})
        # short_answers = annotations.get("short_answers", []) if isinstance(annotations, dict) else []
        #
        # NEW CODE:
        # Natural Questions appears in more than one schema. Handle dict,
        # string, and list annotation shapes without crashing before scoring.
        raw_question = item.get("question", "")
        question = raw_question.get("text", "") if isinstance(raw_question, dict) else str(raw_question)
        annotations = item.get("annotations", {})
        if isinstance(annotations, list) and annotations:
            annotations = annotations[0]
        short_answers = annotations.get("short_answers", []) if isinstance(annotations, dict) else []
        answer_texts = []
        if isinstance(short_answers, list):
            for answer in short_answers:
                if isinstance(answer, dict) and answer.get("text"):
                    answer_texts.append(answer["text"])
                elif isinstance(answer, str) and answer:
                    answer_texts.append(answer)
        if not answer_texts:
            answer_texts = ["unanswerable"]
        document = item.get("document", {})
        context = document.get("text", "") if isinstance(document, dict) else ""
        instruction = create_demo_text()
        prompt = (
            f"{instruction}Supporting information: {context[:3000]}\n\nQ: {question}\nA: "
            if include_context and context else f"{instruction}\n\nQ: {question}\nA: "
        )
        prompts.append(prompt)
        answers.append(answer_texts)
    return prompts, answers


def create_prompts_from_openbookqa(dataset, include_context: bool):
    return create_prompts_from_multiple_choice(dataset, include_context)


def create_prompts_from_multiple_choice(dataset, include_context: bool):
    prompts, answers = [], []
    for item in dataset:
        question = item.get("question", item.get("question_stem", ""))
        choices = item.get("choices", {})
        labels = choices.get("label", []) if isinstance(choices, dict) else []
        texts = choices.get("text", []) if isinstance(choices, dict) else []
        choice_text = " ".join(f"{label}. {text}" for label, text in zip(labels, texts))
        answer_key = item.get("answerKey", item.get("answer_key", ""))
        answer = answer_key
        if answer_key in labels:
            answer = texts[labels.index(answer_key)]
        instruction = create_demo_text()
        prompt_body = f"{question}\nChoices: {choice_text}" if choice_text else question
        prompt = (
            f"{instruction}Supporting information: {choice_text}\n\nQ: {prompt_body}\nA: "
            if include_context else f"{instruction}\n\nQ: {prompt_body}\nA: "
        )
        prompts.append(prompt)
        answers.append([answer, answer_key])
    return prompts, answers


def create_prompts_from_truthfulqa(dataset, include_context: bool):
    prompts, answers = [], []
    for item in dataset:
        question = item.get("question", "")
        correct = item.get("correct_answers", [])
        best = item.get("best_answer", "")
        aliases = correct if isinstance(correct, list) and correct else [best]
        instruction = create_demo_text()
        prompt = f"{instruction}\n\nQ: {question}\nA: "
        prompts.append(prompt)
        answers.append(aliases)
    return prompts, answers


def create_prompts_from_fever(dataset, include_context: bool):
    prompts, answers = [], []
    for item in dataset:
        claim = item.get("claim", "")
        label = str(item.get("label", "")).lower()
        answer = "yes" if label == "supports" else "no" if label == "refutes" else "not enough information"
        instruction = create_demo_text()
        prompt = f"{instruction}\n\nQ: Is this claim supported by evidence? {claim}\nA: "
        prompts.append(prompt)
        answers.append(answer)
    return prompts, answers


def create_prompts_from_halueval(dataset, include_context: bool):
    prompts, answers = [], []
    for item in dataset:
        question = item.get("question", item.get("user_query", ""))
        context = item.get("knowledge", item.get("document", ""))
        answer = item.get("right_answer", item.get("answer", item.get("label", "")))
        instruction = create_demo_text()
        prompt = (
            f"{instruction}Supporting information: {context}\n\nQ: {question}\nA: "
            if include_context and context else f"{instruction}\n\nQ: {question}\nA: "
        )
        prompts.append(prompt)
        answers.append(answer)
    return prompts, answers


def create_prompts_from_strategyqa(dataset, include_context: bool):
    """
    Creates prompts and answers for StrategyQA (yes/no).
    If include_context=True, attaches provided facts as supporting information.
    """
    prompts = []
    answers = []

    for item in dataset:
        question = item.get("question", "")
        facts = item.get("facts", [])
        # StrategyQA answers are booleans; normalize to yes/no strings
        raw_answer = item.get("answer", False)
        if isinstance(raw_answer, bool):
            answer = "yes" if raw_answer else "no"
        elif isinstance(raw_answer, str):
            answer = "yes" if raw_answer.strip().lower() in {"yes", "true", "1"} else "no"
        else:
            answer = "no"

        instruction = create_demo_text()
        if include_context and facts:
            facts_str = " ".join(facts) if isinstance(facts, list) else str(facts)
            prompt = f"{instruction}Supporting information: {facts_str}\n\nQ: {question}\nA: "
        else:
            prompt = f"{instruction}\n\nQ: {question}\nA: "

        prompts.append(prompt)
        answers.append(answer)

    return prompts, answers


# Summarization dataset loaders

def create_prompts_from_cnn_dailymail(dataset, include_context: bool):
    """
    Creates prompts and summaries for CNN/DailyMail.
    If include_context=True, attaches the article to the prompts.
    """
    prompts = []
    summaries = []

    for item in dataset:
        article = item['article']
        highlights = item['highlights']  # This is the reference summary
        
        instruction = create_summarization_demo_text()
        if include_context:
            prompt = f"{instruction}Article: {article}\n\nSummary: "
        else:
            prompt = f"{instruction}Summary: "

        prompts.append(prompt)
        summaries.append(highlights)

    return prompts, summaries


def create_prompts_from_xsum(dataset, include_context: bool):
    """
    Creates prompts and summaries for XSum.
    If include_context=True, attaches the document to the prompts.
    """
    prompts = []
    summaries = []

    for item in dataset:
        document = item['document']
        summary = item['summary']
        
        instruction = create_summarization_demo_text()
        if include_context:
            prompt = f"{instruction}Document: {document}\n\nSummary: "
        else:
            prompt = f"{instruction}Summary: "

        prompts.append(prompt)
        summaries.append(summary)

    return prompts, summaries


def create_prompts_from_samsum(dataset, include_context: bool):
    """
    Creates prompts and summaries for SamSum (dialogue summarization).
    If include_context=True, attaches the dialogue to the prompts.
    """
    prompts = []
    summaries = []

    for item in dataset:
        dialogue = item['dialogue']
        summary = item['summary']
        
        instruction = create_summarization_demo_text()
        if include_context:
            prompt = f"{instruction}Dialogue: {dialogue}\n\nSummary: "
        else:
            prompt = f"{instruction}Summary: "

        prompts.append(prompt)
        summaries.append(summary)

    return prompts, summaries
