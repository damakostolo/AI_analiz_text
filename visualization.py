"""Visualization helpers for LiteraryAI outputs."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from models import ActionTriple


class Visualizer:
    """Encapsulates all plotting and graph-building utilities."""

    def __init__(self, layout_seed: int = 42):
        self.layout_seed = layout_seed

    # ---------- Graph build & save ----------
    def build_action_graph(self, triples: Iterable[ActionTriple], min_count: int = 2) -> nx.DiGraph:
        """Aggregate SVO triples into a weighted, labeled graph.

        Nodes are filtered by how often they appear as subject/object to
        reduce noise. Multiple identical subject→object pairs are merged and
        their verbs/counts are encoded into a single edge label.
        """
        subjects: dict[str, int] = defaultdict(int)
        objects: dict[str, int] = defaultdict(int)
        edge_bucket: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
        edge_first_sent: dict[tuple[str, str], int] = {}

        for t in triples:
            subjects[t.subject] += 1
            objects[t.object] += 1

        for t in triples:
            if subjects[t.subject] < min_count or objects[t.object] < min_count:
                continue
            key = (t.subject.strip(), t.object.strip())
            if not key[0] or not key[1]:
                continue
            edge_bucket[key][t.verb] += 1
            edge_first_sent[key] = min(edge_first_sent.get(key, t.sent_id), t.sent_id)

        G = nx.DiGraph()
        for (s, o), verb_counts in edge_bucket.items():
            verbs_sorted = sorted(verb_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            label_parts = [f"{verb}×{count}" if count > 1 else verb for verb, count in verbs_sorted]
            G.add_edge(
                s,
                o,
                label=" | ".join(label_parts),
                weight=sum(verb_counts.values()),
                sent_order=edge_first_sent[(s, o)],
            )
        return G

    # ---------- Plotting helpers ----------
    def plot_emotion_curve(self, curve_df: pd.DataFrame, out_path: str):
        """Plot the aggregated emotion tone curve."""
        if curve_df.empty:
            return
        plt.figure(figsize=(8, 4))
        plt.plot(
            curve_df["emotion"],
            curve_df["tone"],
            color="purple",
            marker="o",
            linewidth=2,
        )
        plt.ylim(-1.05, 1.05)
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        plt.title("Overall Emotional Tone")
        plt.ylabel("Tone (-1 negative, +1 positive)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()

    def plot_top_characters(self, char_df: pd.DataFrame, out_path: str, top_n: int = 20):
        if char_df.empty:
            return
        df = char_df.head(top_n)
        plt.figure(figsize=(10, 6))
        plt.barh(df["character"], df["count"], color="#7aa6f0")
        plt.gca().invert_yaxis()
        plt.xlabel("Mentions")
        plt.title("Top Characters")
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()

    def plot_action_graph(self, graph: nx.DiGraph, out_path: str):
        """Visualize the action graph with clearer layout and weighted edges."""
        if graph.number_of_nodes() == 0:
            return

        plt.figure(figsize=(11, 8))
        node_sizes = [1200 + 200 * (graph.degree(n)) for n in graph.nodes]
        node_color = [graph.degree(n) for n in graph.nodes]
        pos = nx.spring_layout(graph, seed=self.layout_seed, k=0.8)

        nx.draw_networkx_nodes(
            graph,
            pos,
            node_size=node_sizes,
            node_color=node_color,
            cmap=plt.cm.Blues,
            linewidths=1.0,
            edgecolors="black",
        )

        edge_weights = [1.5 + np.log1p(graph[u][v]["weight"]) for u, v in graph.edges]
        nx.draw_networkx_edges(
            graph,
            pos,
            arrowstyle="-|>",
            arrowsize=16,
            width=edge_weights,
            edge_color="#555555",
            alpha=0.75,
        )

        nx.draw_networkx_labels(graph, pos, font_size=10, font_weight="bold")

        labels = nx.get_edge_attributes(graph, "label")
        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels=labels,
            font_color="#8a2be2",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        )

        plt.title("Action Graph (aggregated verbs, thicker edges = more interactions)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()
