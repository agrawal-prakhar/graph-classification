import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.manifold import MDS

def stress_majorization_layout(graph: nx.Graph, seed: int = 42) -> dict:
    """Compute 2D coordinates with SMACOF stress majorization."""
    dist_matrix = nx.floyd_warshall_numpy(graph)
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        normalized_stress="auto",
        n_init=4,
        random_state=seed,
    )
    coords = mds.fit_transform(dist_matrix)
    return {node: coords[i] for i, node in enumerate(graph.nodes())}


def kamada_kawai_layout(graph: nx.Graph) -> dict:
    """Compute 2D coordinates with Kamada-Kawai."""
    return nx.kamada_kawai_layout(graph, weight=None)


def sample_graphs(count: int, seed: int) -> list[tuple[str, nx.Graph]]:
    """Generate a mixed set of random sample graphs."""
    rng = np.random.default_rng(seed)
    samples: list[tuple[str, nx.Graph]] = []

    for i in range(count):
        graph_type = rng.choice(["erdos_renyi", "barabasi_albert", "watts_strogatz"])
        n = int(rng.integers(12, 36))

        if graph_type == "erdos_renyi":
            p = float(rng.uniform(0.08, 0.24))
            graph = nx.erdos_renyi_graph(n, p, seed=int(rng.integers(1, 1_000_000)))
            name = f"graph_{i:03d}_erdos_renyi_n{n}"
        elif graph_type == "barabasi_albert":
            m = int(rng.integers(2, min(6, n - 1)))
            graph = nx.barabasi_albert_graph(n, m, seed=int(rng.integers(1, 1_000_000)))
            name = f"graph_{i:03d}_barabasi_albert_n{n}"
        else:
            k = int(rng.integers(2, min(8, n - 1)))
            if k % 2 != 0:
                k += 1
            p = float(rng.uniform(0.05, 0.25))
            graph = nx.watts_strogatz_graph(
                n, k, p, seed=int(rng.integers(1, 1_000_000))
            )
            name = f"graph_{i:03d}_watts_strogatz_n{n}"

        samples.append((name, graph))

    return samples


def draw_and_save(graph_name: str, graph: nx.Graph, output_dir: Path, seed: int) -> None:
    kk_pos = kamada_kawai_layout(graph)
    sm_pos = stress_majorization_layout(graph, seed=seed)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{graph_name} ({graph.number_of_nodes()} nodes)")

    nx.draw(
        graph,
        pos=kk_pos,
        ax=axes[0],
        with_labels=False,
        node_size=80,
        width=0.8,
    )
    axes[0].set_title("Kamada-Kawai")
    axes[0].axis("off")

    nx.draw(
        graph,
        pos=sm_pos,
        ax=axes[1],
        with_labels=False,
        node_size=80,
        width=0.8,
    )
    axes[1].set_title("Stress Majorization (SMACOF)")
    axes[1].axis("off")

    output_path = output_dir / f"{graph_name}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate sample graph visualizations for two layout algorithms."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/samples"),
        help="Directory where PNG files are written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stress majorization layout.",
    )
    parser.add_argument(
        "--num-graphs",
        type=int,
        default=100,
        help="How many connected graph images to generate.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped_disconnected = 0
    attempts = 0
    max_attempts = args.num_graphs * 10

    while generated < args.num_graphs and attempts < max_attempts:
        attempts += 1
        base_name, graph = sample_graphs(count=1, seed=args.seed + attempts)[0]
        graph = nx.convert_node_labels_to_integers(graph)

        if not nx.is_connected(graph):
            skipped_disconnected += 1
            continue

        graph_name = f"{generated:03d}_{base_name}"
        draw_and_save(graph_name, graph, args.out_dir, args.seed + generated)
        generated += 1

    print(f"Saved {generated} connected sample images to: {args.out_dir}")
    print(f"Skipped disconnected graphs: {skipped_disconnected}")

    if generated < args.num_graphs:
        print(
            "Warning: could not reach requested count. "
            f"Generated {generated}/{args.num_graphs} within {max_attempts} attempts."
        )


if __name__ == "__main__":
    main()
