import argparse
from typing import Dict, Iterable, List

from dotenv import load_dotenv

from src.benchmark_config import (
    BENCHMARK_ORDER,
    dataset_names_for_benchmark,
    dataset_spec,
    ranking_feature_index_for_spec,
    spec_with_feature_index,
)
from src.feature_cards import feature_cards_for_task
from src.ranking import DEFAULT_RANKING_MODEL, RANKING_FEATURE_SPACES, RANKING_METHODS
from src.ranking_generation import generate_method
from src.ranking_prompts import (
    build_rank_prompt,
    build_score_all_prompt,
    build_score_prompt,
    build_seq_prompt,
)


MODEL_NAME = DEFAULT_RANKING_MODEL
DEFAULT_SCORE_SAMPLES = 5
RANKING_BENCHMARKS = tuple(
    benchmark for benchmark in BENCHMARK_ORDER
    if benchmark != "synthetic_ood"
)


def select_datasets(requested_dataset: str, benchmark: str) -> List[str]:
    valid_datasets = dataset_names_for_benchmark(benchmark)
    if requested_dataset == "all":
        return list(valid_datasets)
    if requested_dataset not in valid_datasets:
        valid = ", ".join(["all", *sorted(valid_datasets)])
        raise ValueError(f"Unsupported dataset: {requested_dataset}. Valid choices: {valid}")
    return [requested_dataset]


def select_methods(requested_method: str) -> List[str]:
    if requested_method == "all":
        return list(RANKING_METHODS)
    return [requested_method]


def load_task_metadata(
    benchmark: str,
    datasets: Iterable[str],
    ranking_feature_space: str,
) -> Dict[str, Dict[str, object]]:
    metadata = {}
    for dataset in datasets:
        spec = dataset_spec(benchmark, dataset)
        if not spec.feature_index:
            spec = spec_with_feature_index(spec)
        feature_index, resolved_feature_space = ranking_feature_index_for_spec(
            spec,
            ranking_feature_space,
        )
        metadata[dataset] = {
            "feature_index": feature_index,
            "resolved_feature_space": resolved_feature_space,
        }
    return metadata


def print_dry_run(
    benchmark: str,
    selected_datasets: Iterable[str],
    selected_methods: Iterable[str],
    metadata: Dict[str, Dict[str, object]],
) -> None:
    for dataset_name in selected_datasets:
        feature_index = metadata[dataset_name]["feature_index"]
        resolved_feature_space = metadata[dataset_name]["resolved_feature_space"]
        feature_cards = feature_cards_for_task(benchmark, dataset_name, feature_index)
        for method in selected_methods:
            print(f"### {dataset_name} / {method} / {resolved_feature_space}")
            if method == "rank":
                print(build_rank_prompt(benchmark, dataset_name, feature_cards))
            elif method == "score":
                print(build_score_prompt(benchmark, dataset_name, feature_cards[0]))
            elif method == "score_all":
                print(build_score_all_prompt(benchmark, dataset_name, feature_cards))
            elif method == "seq":
                print(build_seq_prompt(benchmark, dataset_name, feature_cards, []))
            print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LLM feature rankings for benchmark datasets.")
    parser.add_argument("--benchmark", choices=RANKING_BENCHMARKS, default="acs")
    parser.add_argument("--dataset", default="acsincome", help="Dataset key to rank, or `all`.")
    parser.add_argument("--method", default="all", choices=["all", *RANKING_METHODS])
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--ranking-feature-space", choices=RANKING_FEATURE_SPACES, default="semantic")
    parser.add_argument("--rank-temperature", type=float, default=0.0)
    parser.add_argument("--score-temperature", type=float, default=0.5)
    parser.add_argument("--seq-temperature", type=float, default=0.0)
    parser.add_argument("--score-samples", type=int, default=DEFAULT_SCORE_SAMPLES)
    parser.add_argument("--call-delay", type=float, default=1.0, help="Seconds to sleep between score calls.")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=60.0)
    parser.add_argument("--web-search", action="store_true", help="Enable Responses API web_search tool calls.")
    parser.add_argument("--skip-existing", action="store_true", help="Do not regenerate existing method JSON files.")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling the LLM.")
    args = parser.parse_args()

    load_dotenv()
    selected_datasets = select_datasets(args.dataset, args.benchmark)
    selected_methods = select_methods(args.method)
    metadata = load_task_metadata(
        args.benchmark,
        selected_datasets,
        args.ranking_feature_space,
    )

    if args.dry_run:
        print_dry_run(args.benchmark, selected_datasets, selected_methods, metadata)
        return

    for dataset_name in selected_datasets:
        feature_index = metadata[dataset_name]["feature_index"]
        resolved_feature_space = metadata[dataset_name]["resolved_feature_space"]
        feature_cards = feature_cards_for_task(args.benchmark, dataset_name, feature_index)
        for method in selected_methods:
            path = generate_method(
                method,
                args.benchmark,
                dataset_name,
                feature_index,
                feature_cards,
                args.model,
                args.rank_temperature,
                args.score_temperature,
                args.seq_temperature,
                args.score_samples,
                args.call_delay,
                args.max_retries,
                args.retry_sleep,
                args.web_search,
                args.skip_existing or (
                    args.ranking_feature_space == "model"
                    and resolved_feature_space == "semantic"
                ),
                resolved_feature_space,
            )
            print(
                f"Ready {path} "
                f"(requested_feature_space={args.ranking_feature_space}, "
                f"resolved_feature_space={resolved_feature_space})"
            )


if __name__ == "__main__":
    main()
