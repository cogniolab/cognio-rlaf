"""
Run all RLAF benchmarks and generate visualizations.

Usage:
    python benchmarks/run_all.py
"""

import os
import sys
from runner import BenchmarkRunner
from visualize import (
    plot_performance_comparison,
    plot_convergence_curves,
    plot_radar_chart,
    plot_sample_efficiency
)


def main():
    print("="*80)
    print(" RLAF BENCHMARK SUITE")
    print("="*80)
    print()

    # Create necessary directories
    os.makedirs("benchmarks/datasets", exist_ok=True)
    os.makedirs("benchmarks/results", exist_ok=True)
    os.makedirs("benchmarks/charts", exist_ok=True)

    # Create dummy dataset (replace with real dataset in production)
    import json
    dummy_dataset = [
        {
            "id": i,
            "category": ["hardware", "software", "network", "security", "access"][i % 5],
            "description": f"Sample ITSM ticket #{i}",
            "priority": ["low", "medium", "high", "critical"][i % 4],
        }
        for i in range(1000)
    ]

    dataset_path = "benchmarks/datasets/itsm_tickets.json"
    with open(dataset_path, "w") as f:
        json.dump(dummy_dataset, f, indent=2)

    print(f"✅ Created dataset: {dataset_path} (1000 samples)")
    print()

    # Run benchmarks
    print("Running benchmarks...")
    print()

    runner = BenchmarkRunner(
        methods=[
            "rlaf_arpo",
            "rlaf_grpo_tcr",
            "open_agentrl",
            "arpo_vanilla",
            "ppo",
            "dpo",
            "supervised_ft"
        ],
        dataset=dataset_path,
        num_samples=100  # Use subset for faster demo
    )

    results = runner.run()
    results_csv = "benchmarks/results/benchmark_results.csv"
    runner.save_results(results_csv)
    runner.print_summary()

    # Generate visualizations
    print("\nGenerating visualization charts...")
    print()

    try:
        plot_performance_comparison(results_csv)
        plot_convergence_curves()
        plot_radar_chart(results_csv)
        plot_sample_efficiency()

        print()
        print("="*80)
        print(" BENCHMARK COMPLETE!")
        print("="*80)
        print()
        print("📊 Results saved to:")
        print(f"  - {results_csv}")
        print()
        print("📈 Charts saved to:")
        print("  - benchmarks/charts/performance.png")
        print("  - benchmarks/charts/convergence.png")
        print("  - benchmarks/charts/radar.png")
        print("  - benchmarks/charts/sample_efficiency.png")
        print()
        print("📚 See benchmarks/README.md for detailed analysis")
        print()

    except Exception as e:
        print(f"\n⚠️  Visualization failed: {e}")
        print("Charts require matplotlib and seaborn:")
        print("  pip install matplotlib seaborn")
        print()
        print("Results are still available in:", results_csv)


if __name__ == "__main__":
    main()
