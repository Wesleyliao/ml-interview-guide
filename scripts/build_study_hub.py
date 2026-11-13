"""Build the self-contained GitHub Pages site in docs/ using standard-library Python."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FRAMEWORKS = {'ml': {'theme': '',
        'flow': [{'name': 'Formulation', 'note': 'goals & SLAs'},
                 {'name': 'Metrics', 'note': 'online + offline'},
                 {'name': 'Retrieval', 'note': 'candidates'},
                 {'name': 'Ranking', 'note': 'light → heavy'},
                 {'name': 'Policy Layer', 'note': 'diversity / rules'},
                 {'name': 'Monitoring', 'note': 'drift & retrain'}],
        'steps': [{'title': 'Problem Formulation',
                   'items': ['<strong>Goals:</strong> business objective → ML '
                             'objective',
                             '<strong>Functional:</strong> what is in scope? '
                             'multi-modal? live or batch?',
                             '<strong>Non-functional:</strong> latency, number '
                             'of users, items, QPS'],
                   'tags': ['Scope', 'Live vs Batch', 'Latency']},
                  {'title': 'Metrics',
                   'items': ['<strong>Online:</strong> CTR / conversion / '
                             'engagement + guardrails (latency, hide rate)',
                             '<strong>Offline:</strong> <code>Recall@K</code>, '
                             '<code>NDCG@K</code>, <code>LogLoss / NE</code>, '
                             'calibration (<code>ECE</code>)'],
                   'tags': ['Online', 'Offline', 'Guardrails']},
                  {'title': 'High-Level Components',
                   'items': ['Major pipeline stages end-to-end',
                             'Candidate retrieval → Light ranker → Heavy '
                             'ranker → Re-ranker / Policy'],
                   'tags': ['Two-Stage Funnel']},
                  {'title': 'Data',
                   'items': ['<strong>Labels &amp; signals:</strong> explicit '
                             'vs implicit, negatives (in-batch + log-Q, hard)',
                             '<strong>Features:</strong> scaling, missing '
                             'values, ID / sequence / multimodal encoding',
                             '<strong>Splits:</strong> sampling, temporal / '
                             'user-stratified train-test split (zero leakage)'],
                   'tags': ['Labels', 'Features', 'Temporal Split']},
                  {'title': 'Modeling',
                   'items': ['<strong>Progression:</strong> heuristic / GBDT '
                             'baseline → neural / sequence / multi-task '
                             'architecture',
                             '<strong>Loss &amp; eval:</strong> pointwise BCE '
                             'vs pairwise/listwise; offline slice checks'],
                   'tags': ['Baseline First', 'Loss', 'Calibration']},
                  {'title': 'Deployment & Scaling',
                   'items': ['<strong>Serving:</strong> end-to-end prediction '
                             'service + non-ML policy/filtering layers',
                             '<strong>Operations:</strong> monitoring, alerts, '
                             'feature/label drift, retraining cadence'],
                   'tags': ['Serving', 'Policy Layer', 'Drift']}]}}


LC_GROUPS = [
    {
        "category": "Arrays, Hashing & Prefix Sums",
        "problems": [
            {"n": 1, "title": "Two Sum", "slug": "two-sum", "diff": "E", "tier": "core"},
            {"n": 49, "title": "Group Anagrams", "slug": "group-anagrams", "diff": "M", "tier": "core"},
            {"n": 238, "title": "Product of Array Except Self", "slug": "product-of-array-except-self", "diff": "M", "tier": "core"},
            {"n": 128, "title": "Longest Consecutive Sequence", "slug": "longest-consecutive-sequence", "diff": "M", "tier": "core"},
            {"n": 347, "title": "Top K Frequent Elements", "slug": "top-k-frequent-elements", "diff": "M", "tier": "core"},
            {"n": 560, "title": "Subarray Sum Equals K", "slug": "subarray-sum-equals-k", "diff": "M", "tier": "core"},
            {"n": 36, "title": "Valid Sudoku", "slug": "valid-sudoku", "diff": "M", "tier": "core"},
        ],
    },
    {
        "category": "Two Pointers & Sliding Window",
        "problems": [
            {"n": 125, "title": "Valid Palindrome", "slug": "valid-palindrome", "diff": "E", "tier": "core"},
            {"n": 15, "title": "3Sum", "slug": "3sum", "diff": "M", "tier": "core"},
            {"n": 11, "title": "Container With Most Water", "slug": "container-with-most-water", "diff": "M", "tier": "core"},
            {"n": 3, "title": "Longest Substring Without Repeating Characters", "slug": "longest-substring-without-repeating-characters", "diff": "M", "tier": "core"},
            {"n": 424, "title": "Longest Repeating Character Replacement", "slug": "longest-repeating-character-replacement", "diff": "M", "tier": "core"},
            {"n": 567, "title": "Permutation in String", "slug": "permutation-in-string", "diff": "M", "tier": "core"},
            {"n": 76, "title": "Minimum Window Substring", "slug": "minimum-window-substring", "diff": "H", "tier": "core"},
        ],
    },
    {
        "category": "Binary Search",
        "problems": [
            {"n": 704, "title": "Binary Search", "slug": "binary-search", "diff": "E", "tier": "core"},
            {"n": 33, "title": "Search in Rotated Sorted Array", "slug": "search-in-rotated-sorted-array", "diff": "M", "tier": "core"},
            {"n": 153, "title": "Find Minimum in Rotated Sorted Array", "slug": "find-minimum-in-rotated-sorted-array", "diff": "M", "tier": "core"},
            {"n": 875, "title": "Koko Eating Bananas", "slug": "koko-eating-bananas", "diff": "M", "tier": "core"},
            {"n": 34, "title": "Find First and Last Position of Element", "slug": "find-first-and-last-position-of-element-in-sorted-array", "diff": "M", "tier": "core"},
        ],
    },
    {
        "category": "Stacks & Intervals",
        "problems": [
            {"n": 20, "title": "Valid Parentheses", "slug": "valid-parentheses", "diff": "E", "tier": "core"},
            {"n": 155, "title": "Min Stack", "slug": "min-stack", "diff": "M", "tier": "core"},
            {"n": 739, "title": "Daily Temperatures", "slug": "daily-temperatures", "diff": "M", "tier": "core"},
            {"n": 56, "title": "Merge Intervals", "slug": "merge-intervals", "diff": "M", "tier": "core"},
            {"n": 57, "title": "Insert Interval", "slug": "insert-interval", "diff": "M", "tier": "core"},
            {"n": 435, "title": "Non-overlapping Intervals", "slug": "non-overlapping-intervals", "diff": "M", "tier": "core"},
        ],
    },
    {
        "category": "Linked Lists & LRU",
        "problems": [
            {"n": 206, "title": "Reverse Linked List", "slug": "reverse-linked-list", "diff": "E", "tier": "core"},
            {"n": 21, "title": "Merge Two Sorted Lists", "slug": "merge-two-sorted-lists", "diff": "E", "tier": "core"},
            {"n": 141, "title": "Linked List Cycle", "slug": "linked-list-cycle", "diff": "E", "tier": "core"},
            {"n": 143, "title": "Reorder List", "slug": "reorder-list", "diff": "M", "tier": "core"},
            {"n": 146, "title": "LRU Cache", "slug": "lru-cache", "diff": "M", "tier": "core"},
        ],
    },
    {
        "category": "Trees",
        "problems": [
            {"n": 104, "title": "Maximum Depth of Binary Tree", "slug": "maximum-depth-of-binary-tree", "diff": "E", "tier": "core"},
            {"n": 102, "title": "Binary Tree Level Order Traversal", "slug": "binary-tree-level-order-traversal", "diff": "M", "tier": "core"},
            {"n": 98, "title": "Validate Binary Search Tree", "slug": "validate-binary-search-tree", "diff": "M", "tier": "core"},
            {"n": 236, "title": "Lowest Common Ancestor of a Binary Tree", "slug": "lowest-common-ancestor-of-a-binary-tree", "diff": "M", "tier": "core"},
            {"n": 543, "title": "Diameter of Binary Tree", "slug": "diameter-of-binary-tree", "diff": "E", "tier": "core"},
            {"n": 230, "title": "Kth Smallest Element in a BST", "slug": "kth-smallest-element-in-a-bst", "diff": "M", "tier": "core"},
            {"n": 105, "title": "Construct Binary Tree from Preorder and Inorder", "slug": "construct-binary-tree-from-preorder-and-inorder-traversal", "diff": "M", "tier": "core"},
        ],
    },
    {
        "category": "Graphs, BFS & DFS",
        "problems": [
            {"n": 200, "title": "Number of Islands", "slug": "number-of-islands", "diff": "M", "tier": "core"},
            {"n": 133, "title": "Clone Graph", "slug": "clone-graph", "diff": "M", "tier": "core"},
            {"n": 207, "title": "Course Schedule", "slug": "course-schedule", "diff": "M", "tier": "core"},
            {"n": 994, "title": "Rotting Oranges", "slug": "rotting-oranges", "diff": "M", "tier": "core"},
            {"n": 417, "title": "Pacific Atlantic Water Flow", "slug": "pacific-atlantic-water-flow", "diff": "M", "tier": "core"},
            {"n": 721, "title": "Accounts Merge", "slug": "accounts-merge", "diff": "M", "tier": "core"},
            {"n": 743, "title": "Network Delay Time", "slug": "network-delay-time", "diff": "M", "tier": "core"},
        ],
    },
    {
        "category": "Heaps & Backtracking",
        "problems": [
            {"n": 215, "title": "Kth Largest Element in an Array", "slug": "kth-largest-element-in-an-array", "diff": "M", "tier": "core"},
            {"n": 973, "title": "K Closest Points to Origin", "slug": "k-closest-points-to-origin", "diff": "M", "tier": "core"},
            {"n": 621, "title": "Task Scheduler", "slug": "task-scheduler", "diff": "M", "tier": "core"},
            {"n": 78, "title": "Subsets", "slug": "subsets", "diff": "M", "tier": "core"},
            {"n": 39, "title": "Combination Sum", "slug": "combination-sum", "diff": "M", "tier": "core"},
            {"n": 79, "title": "Word Search", "slug": "word-search", "diff": "M", "tier": "core"},
        ],
    },
    {
        "category": "Dynamic Programming",
        "problems": [
            {"n": 198, "title": "House Robber", "slug": "house-robber", "diff": "M", "tier": "core"},
            {"n": 322, "title": "Coin Change", "slug": "coin-change", "diff": "M", "tier": "core"},
            {"n": 139, "title": "Word Break", "slug": "word-break", "diff": "M", "tier": "core"},
            {"n": 300, "title": "Longest Increasing Subsequence", "slug": "longest-increasing-subsequence", "diff": "M", "tier": "core"},
            {"n": 1143, "title": "Longest Common Subsequence", "slug": "longest-common-subsequence", "diff": "M", "tier": "core"},
        ],
    },
    {
        "category": "Additional: Systems & ML Data Structures",
        "problems": [
            {"n": 981, "title": "Time Based Key-Value Store", "slug": "time-based-key-value-store", "diff": "M", "tier": "ext", "local": True},
            {"n": 380, "title": "Insert Delete GetRandom O(1)", "slug": "insert-delete-getrandom-o1", "diff": "M", "tier": "ext"},
            {"n": 208, "title": "Implement Trie (Prefix Tree)", "slug": "implement-trie-prefix-tree", "diff": "M", "tier": "ext"},
            {"n": 295, "title": "Find Median from Data Stream", "slug": "find-median-from-data-stream", "diff": "H", "tier": "ext"},
            {"n": 1570, "title": "Dot Product of Two Sparse Vectors", "slug": "dot-product-of-two-sparse-vectors", "diff": "M", "tier": "ext", "local": True},
            {"n": 311, "title": "Sparse Matrix Multiplication", "slug": "sparse-matrix-multiplication", "diff": "M", "tier": "ext", "local": True},
            {"n": 588, "title": "Design In-Memory File System", "slug": "design-in-memory-file-system", "diff": "H", "tier": "ext", "local": True},
        ],
    },
    {
        "category": "Additional: Advanced Graphs",
        "problems": [
            {"n": 210, "title": "Course Schedule II", "slug": "course-schedule-ii", "diff": "M", "tier": "ext"},
            {"n": 127, "title": "Word Ladder", "slug": "word-ladder", "diff": "H", "tier": "ext"},
            {"n": 332, "title": "Reconstruct Itinerary", "slug": "reconstruct-itinerary", "diff": "H", "tier": "ext"},
            {"n": 684, "title": "Redundant Connection", "slug": "redundant-connection", "diff": "M", "tier": "ext"},
            {"n": 1584, "title": "Min Cost to Connect All Points", "slug": "min-cost-to-connect-all-points", "diff": "M", "tier": "ext"},
        ],
    },
    {
        "category": "Additional: DP, Greedy & Selected Hards",
        "problems": [
            {"n": 416, "title": "Partition Equal Subset Sum", "slug": "partition-equal-subset-sum", "diff": "M", "tier": "ext"},
            {"n": 72, "title": "Edit Distance", "slug": "edit-distance", "diff": "M", "tier": "ext"},
            {"n": 45, "title": "Jump Game II", "slug": "jump-game-ii", "diff": "M", "tier": "ext"},
            {"n": 239, "title": "Sliding Window Maximum", "slug": "sliding-window-maximum", "diff": "H", "tier": "ext"},
            {"n": 23, "title": "Merge k Sorted Lists", "slug": "merge-k-sorted-lists", "diff": "H", "tier": "ext"},
            {"n": 297, "title": "Serialize and Deserialize Binary Tree", "slug": "serialize-and-deserialize-binary-tree", "diff": "H", "tier": "ext"},
            {"n": 42, "title": "Trapping Rain Water", "slug": "trapping-rain-water", "diff": "H", "tier": "ext"},
            {"n": 124, "title": "Binary Tree Maximum Path Sum", "slug": "binary-tree-maximum-path-sum", "diff": "H", "tier": "ext"},
        ],
    },
]



ML_FILES = [
    ("PyTorch Lab 1: Losses, K-Means & Naive Bayes", "ml_coding/pytorch_transformers/ex1_classical_and_losses.py"),
    ("PyTorch Lab 2: RoPE, Causal GQA & KV Cache", "ml_coding/pytorch_transformers/ex2_attention_and_kvcache.py"),
    ("PyTorch Lab 3: LoRA, DPO & GRPO", "ml_coding/pytorch_transformers/ex3_lora_dpo_grpo.py"),
    ("PyTorch Lab 4: Buggy Trainer Audit", "ml_coding/pytorch_transformers/ex4_buggy_pytorch_trainer.py"),
    ("Applied AI: Async RAG Review & Tests", "ml_coding/applied_ai_live/rag.py"),
    ("Applied AI: Permissioned Cosine Retrieval", "ml_coding/applied_ai_live/retrieval.py"),
    ("Applied AI: Bounded Tool-Calling Loop", "ml_coding/applied_ai_live/agent.py"),
    ("Applied AI: Macro Retrieval Eval Metrics", "ml_coding/applied_ai_live/evaluation.py"),
    ("PyTorch Reference Solutions", "ml_coding/pytorch_transformers/solutions_pytorch.py"),
    ("Applied AI Reference Solutions", "ml_coding/foundational_labs/ex6_ai_live/solutions.py"),
]



def load_files(specs: list[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"label": label, "path": rel, "code": (ROOT / rel).read_text()} for label, rel in specs]


def build() -> None:
    total_lc = sum(len(g["problems"]) for g in LC_GROUPS)
    assert total_lc == 75, f"Expected 75 LeetCode problems, found {total_lc}"

    payload = {
        "frameworks": {"ml": FRAMEWORKS["ml"]},
        "theory": json.loads((ROOT / "curriculum/ml_theory.json").read_text()),
        "diagrams": {name: (ROOT / f"study-hub/diagrams/{name}.svg").read_text() for name in ["recommender", "rag"]},
        "classics": load_files([(name.replace('_',' ').title(), f"ml_coding/classics/{name}.py") for name in ["linear_regression","kmeans","naive_bayes","self_attention","self_attention_numpy"]]),
        "designReferences": (ROOT / "study-hub/design-references.html").read_text().replace("{{recommender}}", (ROOT / "study-hub/diagrams/recommender.svg").read_text()).replace("{{rag}}", (ROOT / "study-hub/diagrams/rag.svg").read_text()),
        "lcGroups": LC_GROUPS,
        
        
        "mlFiles": load_files(ML_FILES),
        
        "industryTasks": [{**task, "code": (ROOT / task["path"]).read_text()} for task in json.loads((ROOT / "curriculum/industry_assessment.json").read_text())],
    }

    shell = (ROOT / "study-hub/shell.html").read_text()
    css = (ROOT / "study-hub/style.css").read_text()
    js = (ROOT / "study-hub/app.js").read_text()
    data_str = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")

    out_html = (
        shell.replace("/* COURSE_CSS */", css)
        .replace("/* COURSE_DATA */", f"const COURSE = {data_str};")
        .replace("/* COURSE_JS */", js)
    )
    output = ROOT / "docs"
    output.mkdir(exist_ok=True)
    (output / "index.html").write_text(out_html)
    (output / ".nojekyll").touch()
    for item in payload["classics"] + payload["mlFiles"] + payload["industryTasks"]:
        target = output / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / item["path"], target)
    print(f"Built docs/index.html ({len(out_html):,} chars, {total_lc} LC problems).")



if __name__ == "__main__":
    build()
