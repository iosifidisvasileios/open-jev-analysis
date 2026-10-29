"""DSPy instruction optimization with Open-Jev as a non-generative target.

The optimizer mutates a registered Predict signature. The forward pass reads that exact
signature and sends its instructions to Client.ask, never to a text-generation LM.
Only proposal calls use the separately configured generative prompt LM.
"""

from copy import deepcopy
import hashlib
import json
from collections import Counter

import dspy

from .experiments import assert_disjoint_splits, make_client_choice_predictor, multiclass_metrics


class OpenJevPromptProgram(dspy.Module):
    """Open-Jev classifier whose rubric is stored in a DSPy signature. See __init__ for configuration parameters."""

    def __init__(self, client, question, *, model_alias="open-jev", expected_model=None,
                 positive_classes=("offensive", "hatespeech"), negative_class="normal"):
        """Configure this object without running inference.
        
        Args:
            client: Configured Open-Jev client exposing ask(); reused for classification requests.
            question: Choice rubric dictionary with type, instructions, and criteria mapping class names to descriptions; copied before editing.
            model_alias: Model name sent to the server (default "open-jev"); this does not load a checkpoint.
            expected_model: Expected returned checkpoint identifier, usually from /health; None disables the identity check.
            positive_classes: Class names grouped as binary label 1; their probabilities are summed into p_positive.
            negative_class: Class name mapped to binary label 0, usually "normal".
        """
        super().__init__()
        self.client = client
        self.base_question = deepcopy(question)
        self.model_alias, self.expected_model = model_alias, expected_model
        self.positive_classes, self.negative_class = tuple(positive_classes), negative_class
        signature = dspy.Signature("text -> label", instructions=question["instructions"])
        self.classify = dspy.Predict(signature)
        self.initial_prefix = signature.fields["label"].json_schema_extra["prefix"]

    def question(self):
        """Return a fresh rubric using the current DSPy instructions.
        """
        question = deepcopy(self.base_question)
        signature = self.classify.signature
        question["instructions"] = signature.instructions
        prefix = signature.fields["label"].json_schema_extra["prefix"]
        # COPRO also optimizes the output cue. Apply it to the server criterion
        # only when changed, so the untouched program matches the original rubric.
        if prefix != self.initial_prefix:
            question["instructions"] += f"\nOutput cue: {prefix}"
        return question

    def predict_record(self, text):
        """Classify one post through Open-Jev and return a normalized prediction record.
        
        Args:
            text: Input post as a string, without reference labels. Treated as untrusted evidence, not classifier instructions.
        """
        predict = make_client_choice_predictor(
            self.client, self.question(), positive_class=self.positive_classes,
            negative_class=self.negative_class, model_alias=self.model_alias,
            expected_model=self.expected_model,
        )
        return predict(text)

    def forward(self, text):
        """Classify one post and return a DSPy Prediction with native class and binary decision.
        
        Args:
            text: Input post as a string, without reference labels. Treated as untrusted evidence, not classifier instructions.
        """
        row = self.predict_record(text)
        return dspy.Prediction(label=row["predicted_class"], **row)


def native_binary_accuracy(example, prediction, trace=None):
    """Optimize native argmax mapped to the positive group; no score threshold.
    
    Args:
        example: One labeled Example used for inference or scoring; reference labels must not be sent as model input.
        prediction: DSPy prediction with predicted_label containing the native binary decision.
        trace: Optional DSPy trace argument, accepted for metric compatibility and ignored.
    """
    return float(prediction.predicted_label == example.binary_label)


def make_prompt_search_sets(train, validation, test, *, train_limit=96, validation_limit=128,
                            balanced=False, classes=("normal", "offensive", "hatespeech")):
    """Deterministic disjoint search/validation sets, keeping shared test intact.
    
    Drop validation copies of training/test texts before selecting its prefix.
    Test labels are never used for filtering or returned to the optimizer.
    
    Args:
        train: Prepared training dataset; must not share trimmed text with test. DSPy selection expects id, text, label, and label_multiclass columns.
        validation: Official prepared validation dataset with id, text, label, and label_multiclass. Copies of train/test text are removed before selection.
        test: Prepared held-out test dataset; membership is preserved. Text is used to detect leakage, not to tune prompts or thresholds.
        train_limit: Maximum search-training examples (default 96), not the later threshold-fitting size. Balanced mode requires enough examples to meet each class quota.
        validation_limit: Maximum validation examples (helper default 128; notebook uses 126). Balanced mode divides the quota across classes; earlier classes receive any remainder.
        balanced: If True, select near-equal per-class quotas deterministically; otherwise take the original-order prefix. No shuffle is performed.
        classes: Ordered class names, e.g. ["normal", "offensive", "hatespeech"]; order determines confusion-matrix rows and columns.
    
    Returns:
        (search_examples, validation_examples, audit). Only text is marked as model
        input; labels remain available to scoring. The shared test set is unchanged.
    """
    for limit in (train_limit, validation_limit):
        if type(limit) is not int or limit < 1:
            raise ValueError("Search limits must be positive integers")
    assert_disjoint_splits(train, test)
    forbidden = {row["text"].strip() for row in train} | {row["text"].strip() for row in test}
    validation_indices = [i for i, row in enumerate(validation) if row["text"].strip() not in forbidden]
    clean_validation = validation.select(validation_indices)
    if not len(clean_validation):
        raise ValueError("No disjoint validation examples remain")
    def select(table, limit):
        """Select.
        
        Args:
            table: Prepared Hugging Face table to select or convert to DSPy examples.
            limit: Maximum rows to select from this table.
        """
        if not balanced:
            return table.select(range(min(limit, len(table))))
        if limit < len(classes):
            raise ValueError("Balanced selection needs at least one example per class")
        indices = []
        for i, name in enumerate(classes):
            quota = limit // len(classes) + int(i < limit % len(classes))
            candidates = [j for j, row in enumerate(table) if row["label_multiclass"] == name]
            if len(candidates) < quota:
                raise ValueError(f"Not enough {name} examples for balanced selection: need {quota}, have {len(candidates)}")
            indices.extend(candidates[:quota])
        return table.select(sorted(indices))
    search_train = select(train, train_limit)
    selection = select(clean_validation, validation_limit)
    assert_disjoint_splits(search_train, selection)
    assert_disjoint_splits(selection, test)

    def to_examples(table):
        """To examples.
        
        Args:
            table: Prepared Hugging Face table to select or convert to DSPy examples.
        """
        return [dspy.Example(text=r["text"], label=r["label_multiclass"], binary_label=r["label"]).with_inputs("text") for r in table]

    audit = {"method": "deterministic per-class quotas; original order" if balanced else "official split prefixes; no shuffle",
             "balanced": balanced, "search_train_class_counts": dict(Counter(search_train["label_multiclass"])),
             "validation_class_counts": dict(Counter(selection["label_multiclass"])),
             "search_train_ids": list(search_train["id"]),
             "validation_ids": list(selection["id"]), "test_ids": list(test["id"]),
             "validation_overlap_rows_removed": len(validation) - len(clean_validation)}
    audit["selection_sha256"] = hashlib.sha256(json.dumps(audit, sort_keys=True).encode()).hexdigest()
    return to_examples(search_train), to_examples(selection), audit


def evaluate_class_balanced(program, examples, *, classes, worst_class_weight=.25):
    """Exact dataset-level macro-F1 objective, not an average of per-row F1s.
    
    Args:
        program: Callable DSPy classifier accepting text and returning label (class name) and predicted_label (binary native decision).
        examples: Nonempty sequence of dspy.Example objects with text input, label (class name), and binary_label (0/1); use a single intended evaluation split.
        classes: Ordered class names, e.g. ["normal", "offensive", "hatespeech"]; order determines confusion-matrix rows and columns.
        worst_class_weight: Number in [0, 1] weighting the weakest class F1: objective = (1-w)*macro_F1 + w*minimum_class_F1. Default 0.25; 0 uses macro-F1 only, 1 uses only the weakest class. This is not a probability threshold.
    
    Returns:
        Three-class metrics, weakest-class F1, weighted objective, and native binary
        accuracy. No probability threshold is applied; prediction failures abort.
    """
    if not 0 <= worst_class_weight <= 1:
        raise ValueError("worst_class_weight must be between 0 and 1")
    predictions = [program(**example.inputs()) for example in examples]
    rows = [{"predicted_class": p.label, "error": None} for p in predictions]
    report = multiclass_metrics(rows, [e.label for e in examples], classes)
    report["min_class_f1"] = min(v["f1"] for v in report["per_class"].values())
    report["objective"] = (1 - worst_class_weight) * report["macro_f1"] + worst_class_weight * report["min_class_f1"]
    report["worst_class_weight"] = worst_class_weight
    report["binary_accuracy"] = sum(p.predicted_label == e.binary_label for p, e in zip(predictions, examples)) / len(examples)
    return report


class BalancedInstructionProposal(dspy.Signature):
    """Improve all classes rather than sacrificing a class for overall accuracy.

    Rewrite the instruction to distinguish every fixed class, guided by per-class
    precision, recall, F1, and confusion counts. Preserve the supplied class policy.
    """
    current_instruction = dspy.InputField()
    fixed_class_definitions = dspy.InputField()
    feedback = dspy.InputField(desc="Aggregate training metrics and confusion matrix; no test results")
    objective = dspy.InputField()
    proposed_instruction = dspy.OutputField()


class ClassBalancedPromptOptimizer(dspy.teleprompt.Teleprompter):
    """Small DSPy instruction search supporting an exact aggregate objective.

    Stock COPRO averages example scores, which cannot express exact macro-F1.
    This optimizer uses DSPy proposal generation but scores each complete candidate
    with dataset-level macro-F1 plus a configurable worst-class-F1 term.
    """
    def __init__(self, prompt_model, *, classes, breadth=3, depth=2, worst_class_weight=.25):
        """Configure this object without running inference.
        
        Args:
            prompt_model: Configured generative dspy.LM that proposes rewritten instructions (e.g. Azure DeepSeek). Open-Jev still classifies; proposal calls may incur charges.
            classes: Ordered class names, e.g. ["normal", "offensive", "hatespeech"]; order determines confusion-matrix rows and columns.
            breadth: Number of instruction candidates proposed per search round (default 3). Larger values explore more alternatives and increase proposer and classifier calls.
            depth: Number of sequential search rounds (default 2). Each round starts from the best instruction so far. Approximately (1 + breadth*depth)*len(trainset) classification calls are scored.
            worst_class_weight: Number in [0, 1] weighting the weakest class F1: objective = (1-w)*macro_F1 + w*minimum_class_F1. Default 0.25; 0 uses macro-F1 only, 1 uses only the weakest class. This is not a probability threshold.
        """
        if breadth < 1 or depth < 1:
            raise ValueError("breadth and depth must be positive")
        self.prompt_model, self.classes = prompt_model, list(classes)
        self.breadth, self.depth, self.worst_class_weight = breadth, depth, worst_class_weight

    def compile(self, student, *, trainset):
        """Search training instructions and return the best program, retaining the baseline on ties.
        
        Args:
            student: OpenJevPromptProgram to improve. Deep-copied so the original baseline remains unchanged; class definitions stay fixed.
            trainset: Nonempty training-only sequence of dspy.Example objects with text input, label (class name), and binary_label (0/1). Never supply validation or test data.
        
        Returns:
            A copied program with candidate_history and total_calls (number of complete
            candidate evaluations, not HTTP calls). Threshold fitting is a later step.
        """
        best = student.deepcopy()
        history = []
        def assess(program, depth):
            """Assess.
            
            Args:
                program: Callable DSPy classifier accepting text and returning label (class name) and predicted_label (binary native decision).
                depth: Search-round index recorded in candidate history; 0 denotes the baseline.
            """
            report = evaluate_class_balanced(program, trainset, classes=self.classes,
                                             worst_class_weight=self.worst_class_weight)
            history.append({"instruction": program.question()["instructions"], "depth": depth,
                            "score": report["objective"], **report})
            return report
        best_report = assess(best, 0)
        for depth in range(1, self.depth + 1):
            feedback = {"current": best_report, "recent_candidates": history[-self.breadth:]}
            with dspy.context(lm=self.prompt_model, adapter=dspy.ChatAdapter()):
                proposals = dspy.Predict(BalancedInstructionProposal, n=self.breadth)(
                    current_instruction=best.question()["instructions"],
                    fixed_class_definitions=json.dumps(student.base_question["criteria"]),
                    feedback=json.dumps(feedback),
                    objective=f"Maximize {(1-self.worst_class_weight):.2f} * macro_F1 + {self.worst_class_weight:.2f} * minimum_class_F1 across {self.classes}; improve the weakest class without collapsing others.",
                )
            # Hold the current best constant while comparing this batch.
            batch_base = best.deepcopy()
            for instruction in proposals.completions.proposed_instruction:
                if not isinstance(instruction, str) or not instruction.strip():
                    raise ValueError("Proposer returned an empty instruction")
                candidate = batch_base.deepcopy()
                candidate.classify.signature = candidate.classify.signature.with_instructions(instruction.strip())
                report = assess(candidate, depth)
                print(f"Depth {depth}: macro-F1={report['macro_f1']:.3f}, weakest F1={report['min_class_f1']:.3f}, objective={report['objective']:.3f}")
                if report["objective"] > best_report["objective"]:
                    best, best_report = candidate, report
        best.candidate_history = history
        best.total_calls = len(history)
        return best


def choose_class_balanced_prompt(baseline, candidate, *, max_class_f1_drop=0.):
    """Recommend a candidate only with a better objective and no large class drop.
    
    Args:
        baseline: Baseline validation report returned by evaluate_class_balanced, including objective and per_class F1.
        candidate: Optimized-prompt validation report with the same classes and objective definition as baseline.
        max_class_f1_drop: Largest allowed absolute per-class validation F1 decrease. Default 0 forbids any decrease; 0.02 permits two percentage points. A strictly higher aggregate objective is still required.
    
    Returns:
        Selection name (baseline or optimized), per-class F1 differences, and gate
        status. This function does not fit or compare binary thresholds.
    """
    if max_class_f1_drop < 0:
        raise ValueError("max_class_f1_drop must be nonnegative")
    drops = {name: candidate["per_class"][name]["f1"] - scores["f1"] for name, scores in baseline["per_class"].items()}
    allowed = all(delta >= -max_class_f1_drop - 1e-12 for delta in drops.values())
    return {"selected_for_deployment": "optimized" if candidate["objective"] > baseline["objective"] and allowed else "baseline",
            "per_class_f1_change": drops, "per_class_guard_passed": allowed,
            "max_class_f1_drop": max_class_f1_drop}
