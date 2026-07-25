import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


CHECKPOINT_PATTERN = re.compile(
    r"(?:checkpoint|global_step)[-_]?(\d+)",
    re.IGNORECASE,
)

METRIC_ALIASES = {
    "prompt_level_strict_acc": [
        "prompt_level_strict_acc",
        "prompt_level_strict_acc,none",
    ],
    "inst_level_strict_acc": [
        "inst_level_strict_acc",
        "inst_level_strict_acc,none",
        "instruction_level_strict_acc",
        "instruction_level_strict_acc,none",
    ],
    "prompt_level_loose_acc": [
        "prompt_level_loose_acc",
        "prompt_level_loose_acc,none",
    ],
    "inst_level_loose_acc": [
        "inst_level_loose_acc",
        "inst_level_loose_acc,none",
        "instruction_level_loose_acc",
        "instruction_level_loose_acc,none",
    ],
}

IF_RL_BINARY_SCORES = [
    0.3253,
    0.3124,
    0.3383,
    0.3272,
    0.3161,
    0.3087,
    0.3346,
    0.3364,
    0.3346,
    0.3420,
    0.3457,
    0.3290,
    0.3475,
    0.3420,
    0.3475,
    0.3641,
    0.3567,
    0.3752,
]
IF_RL_BINARY_STEPS = [i * 10 for i in range(1, len(IF_RL_BINARY_SCORES) + 1)]


def resolve_eval_directory(root_directory: Path, preferred_subdir: str | None) -> Path:
    """
    Resolve the evaluation directory to use for a run.

    Prefers a specific subdirectory when it exists, but falls back to the
    root directory or another common IFEval directory if needed.
    """
    candidates: list[Path] = []

    if preferred_subdir:
        candidates.append(root_directory / preferred_subdir)

    candidates.extend(
        [
            root_directory / "ifeval_binary",
            root_directory / "ifeval",
            root_directory,
        ]
    )

    for candidate in candidates:
        if candidate.exists() and any(candidate.rglob("*.json")):
            return candidate

    raise FileNotFoundError(
        f"No evaluation JSON files found under {root_directory}"
    )


def find_checkpoint_step(json_path: Path, data: dict) -> int | None:
    """
    Find the checkpoint number from either:
      - the path: checkpoint-20/results.json
      - the JSON metadata/model path
    """
    match = CHECKPOINT_PATTERN.search(str(json_path))

    if match:
        return int(match.group(1))

    metadata = json.dumps(data)
    match = CHECKPOINT_PATTERN.search(metadata)

    if match:
        return int(match.group(1))

    return None


def find_metric_recursively(
    value,
    aliases: list[str],
) -> float | None:
    """
    Recursively search a JSON object for one of the metric aliases.
    """
    if isinstance(value, dict):
        for key, nested_value in value.items():
            if key in aliases:
                try:
                    return float(nested_value)
                except (TypeError, ValueError):
                    pass

        for nested_value in value.values():
            result = find_metric_recursively(nested_value, aliases)

            if result is not None:
                return result

    elif isinstance(value, list):
        for item in value:
            result = find_metric_recursively(item, aliases)

            if result is not None:
                return result

    return None


def load_evaluations(
    eval_directory: Path,
    metric: str,
) -> dict[int, tuple[float, Path]]:
    """
    Return:
        checkpoint_step -> (metric_value, result_file)
    """
    if not eval_directory.exists():
        raise FileNotFoundError(
            f"Evaluation folder does not exist: {eval_directory}"
        )

    aliases = METRIC_ALIASES.get(
        metric,
        [metric, f"{metric},none"],
    )

    evaluations: dict[int, tuple[float, Path]] = {}

    for json_path in eval_directory.rglob("*.json"):
        try:
            with json_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue

        score = find_metric_recursively(data, aliases)

        if score is None:
            continue

        step = find_checkpoint_step(json_path, data)

        if step is None:
            print(
                f"[WARNING] Found an IFEval result but could not determine "
                f"its checkpoint:\n  {json_path}"
            )
            continue

        previous = evaluations.get(step)

        # If multiple result files exist for one checkpoint,
        # retain the most recently modified one.
        if (
            previous is None
            or json_path.stat().st_mtime > previous[1].stat().st_mtime
        ):
            evaluations[step] = (score, json_path)

    return evaluations


def print_evaluations(
    label: str,
    evaluations: dict[int, tuple[float, Path]],
) -> None:
    print(f"\n{label}")
    print("-" * len(label))

    for step in sorted(evaluations):
        score, path = evaluations[step]
        print(f"checkpoint-{step:<5} {score:.4f}  {path}")


def load_single_evaluation(
    evaluation_path: Path,
    metric: str,
) -> tuple[float, Path] | None:
    """
    Load a single metric value from either a JSON file or a directory.
    """
    if not evaluation_path.exists():
        raise FileNotFoundError(
            f"Evaluation path does not exist: {evaluation_path}"
        )

    aliases = METRIC_ALIASES.get(
        metric,
        [metric, f"{metric},none"],
    )

    candidate_paths = (
        [evaluation_path]
        if evaluation_path.is_file()
        else sorted(evaluation_path.rglob("*.json"))
    )

    for json_path in candidate_paths:
        try:
            with json_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue

        score = find_metric_recursively(data, aliases)
        if score is not None:
            return score, json_path

    return None


def plot_run(
    evaluations: dict[int, tuple[float, Path]],
    label: str,
) -> None:
    steps = sorted(evaluations)
    scores = [evaluations[step][0] for step in steps]

    plt.plot(
        steps,
        scores,
        marker="o",
        linewidth=2,
        label=label,
    )

    for step, score in zip(steps, scores):
        plt.annotate(
            f"{score:.3f}",
            xy=(step, score),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )


def main() -> None:
    project_directory = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Compare IFEval results from CASCADE2/eval and "
            "CASCADE2/TRL-version/eval."
        )
    )

    parser.add_argument(
        "--cascade-eval",
        type=Path,
        default=project_directory / "eval",
        help="Evaluation directory for the Cascade2 run",
    )

    parser.add_argument(
        "--cascade-label",
        default="Cascade2 eval",
    )

    parser.add_argument(
        "--trl-eval",
        type=Path,
        default=None,
        help="Evaluation directory for the TRL run",
    )

    parser.add_argument(
        "--trl-label",
        default="TRL - ds",
    )

    parser.add_argument(
        "--verl-binary-eval",
        type=Path,
        default=None,
        help="Evaluation directory for the VeRL binary run (temperature 0)",
    )

    parser.add_argument(
        "--verl-binary-label",
        default="VeRL bfloat16 + ds",
    )

    parser.add_argument(
        "--verl-nods-eval",
        type=Path,
        default=None,
        help="Evaluation directory for the VeRL fraction run (temperature 0)",
    )

    parser.add_argument(
        "--verl-withds-eval",
        type=Path,
        default=None,
        help="Evaluation directory for the VeRL fraction run (temperature 0)",
    )

    parser.add_argument(
        "--verl-nods-label",
        default="VeRL fp32 - ds",
    )

    parser.add_argument(
        "--verl-withds-label",
        default="VeRL fp32 + ds",
    )

    parser.add_argument(
        "--base-eval",
        type=Path,
        default=(
            project_directory
            / "verl-version"
            / "Meluxina"
            / "checkpoints"
            / "merged_checkpoints"
            / "eval"
            / "ifeval"
            / "Qwen__Qwen2.5-1.5B-Instruct"
        ),
        help="Single evaluation JSON or directory for the base model",
    )

    parser.add_argument(
        "--base-label",
        default="Base model",
    )

    parser.add_argument(
        "--metric",
        default="prompt_level_strict_acc",
        choices=list(METRIC_ALIASES),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=project_directory / "if_rl_comparison.png",
    )

    args = parser.parse_args()

    # cascade_eval_dir = resolve_eval_directory(
    #     args.cascade_eval,
    #     preferred_subdir="ifeval",
    # )

    trl_eval_dir = resolve_eval_directory(
        args.trl_eval or project_directory / "TRL-version" / "IF-RL-Binary_checkpoints",
        preferred_subdir="ifeval_binary",
    )

    verl_binary_eval_dir = resolve_eval_directory(
        args.verl_binary_eval or project_directory / "verl-version" / "Meluxina" / "IF_RL" / "if_rl_verl_binary_checkpoints" / "merged_checkpoints" / "2_eval",
        preferred_subdir="ifeval",
    )

    verl_nods_eval_dir = resolve_eval_directory(
        args.verl_nods_eval or project_directory / "verl-version" / "Meluxina" / "IF_RL_Binary" / "no_ds" / "checkpoints" / "merged_checkpoints" / "eval",
        preferred_subdir="ifeval",
    )

    verl_withds_eval_dir = resolve_eval_directory(
        args.verl_withds_eval or project_directory / "verl-version" / "Meluxina" / "IF_RL_Binary" / "with_ds" / "checkpoints" / "merged_checkpoints" / "eval",
        preferred_subdir="ifeval",
    )

    # print(f"Using Cascade2 eval directory: {cascade_eval_dir}")
    print(f"Using TRL eval directory: {trl_eval_dir}")
    print(f"Using VeRL bfloat16 + ds eval directory: {verl_binary_eval_dir}")
    print(f"Using VeRL fp32 - ds eval directory: {verl_nods_eval_dir}")
    print(f"Using VeRL fp32 + ds eval directory: {verl_withds_eval_dir}")

    # cascade_results = load_evaluations(
    #     cascade_eval_dir,
    #     args.metric,
    # )

    trl_results = load_evaluations(
        trl_eval_dir,
        args.metric,
    )

    verl_binary_results = load_evaluations(
        verl_binary_eval_dir,
        args.metric,
    )

    verl_nods_results = load_evaluations(
        verl_nods_eval_dir,
        args.metric,
    )

    verl_withds_results = load_evaluations(
        verl_withds_eval_dir,
        args.metric,
    )

    # if not cascade_results:
    #     raise RuntimeError(
    #         f"No {args.metric!r} evaluations found under "
    #         f"{args.cascade_eval}"
    #     )

    if not trl_results:
        raise RuntimeError(
            f"No {args.metric!r} evaluations found under "
            f"{args.trl_eval}"
        )

    if not verl_binary_results:
        raise RuntimeError(
            f"No {args.metric!r} evaluations found under "
            f"{args.verl_binary_eval}"
        )

    if not verl_nods_results:
        raise RuntimeError(
            f"No {args.metric!r} evaluations found under "
            f"{args.verl_nods_eval}"
        )

    if not verl_withds_results:
        raise RuntimeError(
            f"No {args.metric!r} evaluations found under "
            f"{args.verl_withds_eval}"
        )

    base_result = None
    if args.base_eval is not None:
        base_result = load_single_evaluation(args.base_eval, args.metric)
        if base_result is None:
            raise RuntimeError(
                f"No {args.metric!r} evaluation found under {args.base_eval}"
            )

        base_score, base_path = base_result
        print(f"\n{args.base_label}")
        print("-" * len(args.base_label))
        print(f"{args.metric}: {base_score:.4f}  {base_path}")

    # print_evaluations(args.cascade_label, cascade_results)
    print_evaluations(args.trl_label, trl_results)

    # common_steps = sorted(
    #     set(cascade_results) & set(trl_results)
    # )
    #
    # print(f"\nCheckpoints found in both runs: {common_steps}")

    plt.figure(figsize=(10, 6))

    # plot_run(cascade_results, args.cascade_label)
    plot_run(trl_results, args.trl_label)
    plot_run(verl_binary_results, args.verl_binary_label)
    plot_run(verl_nods_results, args.verl_nods_label)
    plot_run(verl_withds_results, args.verl_withds_label)

    # if_rl_binary_dict = {
    #     step: (score, Path(""))
    #     for step, score in zip(IF_RL_BINARY_STEPS, IF_RL_BINARY_SCORES)
    # }
    # plot_run(if_rl_binary_dict, args.ifrl_binary_label)

    all_steps = sorted(
        set(trl_results) | set(verl_binary_results) | set(verl_nods_results)
    )

    if base_result is not None:
        base_score, _ = base_result
        plt.axhline(
            base_score,
            color="gray",
            linestyle="--",
            linewidth=1.5,
            label=args.base_label,
        )
        if all_steps:
            plt.text(
                max(all_steps),
                base_score,
                f" {base_score:.3f}",
                color="gray",
                fontsize=8,
                ha="left",
                va="bottom" if base_score < 0.5 else "top",
            )

    plt.xlabel("Checkpoint step")
    plt.ylabel(args.metric.replace("_", " ").title())
    plt.title("IFEval Checkpoint Comparison of IF-RL Binary Reward (temp=0)")
    plt.xticks(all_steps)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(args.output, dpi=200)
    print(f"\nGraph saved to: {args.output}")

    plt.show()


if __name__ == "__main__":
    main()