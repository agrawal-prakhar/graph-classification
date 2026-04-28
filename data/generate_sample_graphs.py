import argparse
from pathlib import Path

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


def sample_graphs() -> list[tuple[str, nx.Graph]]:
    """Return a small library of graphs for quick experimentation."""
    return [
        ("path_10", nx.path_graph(10)),
        ("cycle_12", nx.cycle_graph(12)),
        ("complete_8", nx.complete_graph(8)),
        ("grid_4x4", nx.grid_2d_graph(4, 4)),
        ("erdos_renyi_20", nx.erdos_renyi_graph(20, 0.15, seed=42)),
        ("barabasi_albert_20", nx.barabasi_albert_graph(20, 2, seed=42)),
    ]


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
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for graph_name, graph in sample_graphs():
        graph = nx.convert_node_labels_to_integers(graph)
        draw_and_save(graph_name, graph, args.out_dir, args.seed)

    print(f"Saved {len(sample_graphs())} sample images to: {args.out_dir}")


if __name__ == "__main__":
    main()
