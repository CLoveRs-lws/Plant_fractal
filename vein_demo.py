from pathlib import Path
from collections import Counter, deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from skimage import io, color, filters, morphology, exposure, measure
from scipy import ndimage as ndi
import networkx as nx


# Config

IMG_PATH = Path("vein_demo/vein.png")
OUT_DIR = Path("vein_demo")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300

MIN_OBJECT_SIZE = 80
MIN_HOLE_SIZE = 80

BOX_DIM_MIN_POWER = 2
LACUNARITY_BOX_SIZES = [8, 16, 32, 64, 128]

sns.set_theme(style="whitegrid", context="talk")


# ============================================================
# Basic utilities
# ============================================================

def save_fig(fig, filename):
    fig.savefig(OUT_DIR / filename, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def save_image(img, filename, cmap="gray", title=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(img, cmap=cmap)
    if title:
        ax.set_title(title, fontsize=14)
    ax.axis("off")
    save_fig(fig, filename)


def crop_to_object(binary, padding=20):
    rows, cols = np.where(binary)
    if len(rows) == 0:
        return binary

    r0 = max(rows.min() - padding, 0)
    r1 = min(rows.max() + padding + 1, binary.shape[0])
    c0 = max(cols.min() - padding, 0)
    c1 = min(cols.max() + padding + 1, binary.shape[1])

    return binary[r0:r1, c0:c1]


# ============================================================
# 1. Binary segmentation and skeletonization
# ============================================================

def preprocess_image(img_path):
    img = io.imread(img_path)

    if img.ndim == 4:
        img = img[:, :, :3]

    if img.ndim == 3:
        gray = color.rgb2gray(img)
    else:
        gray = img.astype(float)
        gray = gray / gray.max()

    # Only use mild contrast enhancement + Otsu.
    gray_eq = exposure.equalize_adapthist(gray, clip_limit=0.02)
    thresh = filters.threshold_otsu(gray_eq)
    raw_binary = gray_eq > thresh

    # Vein is bright on dark background. If selected area is too large, invert.
    if raw_binary.mean() > 0.5:
        raw_binary = ~raw_binary

    raw_binary = morphology.remove_small_objects(raw_binary, min_size=MIN_OBJECT_SIZE)
    raw_binary = morphology.remove_small_holes(raw_binary, area_threshold=MIN_HOLE_SIZE)

    skeleton = morphology.skeletonize(raw_binary)

    return img, gray, raw_binary, skeleton


# ============================================================
# 2. Porosity
# ============================================================

def compute_porosity(binary):
    """
    Porosity is defined as void fraction inside the convex hull:

        porosity = 1 - vein_area / convex_hull_area

    For formal analysis, replace convex hull with a manually segmented leaf mask.
    """
    hull = morphology.convex_hull_image(binary)
    vein_area = binary.sum()
    hull_area = hull.sum()
    porosity = 1 - vein_area / hull_area if hull_area > 0 else np.nan
    return porosity, hull


def plot_porosity(binary, hull, porosity):
    display = np.zeros((*binary.shape, 3), dtype=float)

    # background black, hull gray, vein light
    display[hull] = [0.22, 0.22, 0.22]
    display[binary] = [0.95, 0.95, 0.88]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(display)
    ax.set_title(f"Porosity = {porosity:.3f}", fontsize=15)
    ax.axis("off")
    save_fig(fig, "03_porosity_hull.png")


# ============================================================
# 3. Box-counting dimension
# ============================================================

def box_count(binary, box_size):
    img = binary.astype(bool)
    h, w = img.shape

    h_trim = h - h % box_size
    w_trim = w - w % box_size

    img = img[:h_trim, :w_trim]

    if img.size == 0:
        return np.nan

    blocks = img.reshape(
        h_trim // box_size,
        box_size,
        w_trim // box_size,
        box_size
    )

    occupied = blocks.any(axis=(1, 3))
    return occupied.sum()


def box_counting_dimension(binary):
    binary = crop_to_object(binary, padding=5)

    min_side = min(binary.shape)
    max_power = int(np.floor(np.log2(min_side)))

    sizes = np.array([2 ** k for k in range(BOX_DIM_MIN_POWER, max_power)])
    counts = []

    for s in sizes:
        counts.append(box_count(binary, s))

    counts = np.array(counts)
    valid = np.isfinite(counts) & (counts > 0)

    sizes = sizes[valid]
    counts = counts[valid]

    x = np.log(1 / sizes)
    y = np.log(counts)

    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        D = slope
    else:
        D = np.nan
        intercept = np.nan

    return D, sizes, counts, x, y, intercept


def plot_box_dimension(binary):
    D, sizes, counts, x, y, intercept = box_counting_dimension(binary)

    palette = sns.color_palette("mako", 6)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.scatterplot(x=x, y=y, s=80, color=palette[4], ax=ax)

    if np.isfinite(D):
        xfit = np.linspace(x.min(), x.max(), 100)
        yfit = D * xfit + intercept
        ax.plot(xfit, yfit, lw=2.2, ls="--", color=palette[1])

    ax.set_xlabel(r"$\log(1/\varepsilon)$")
    ax.set_ylabel(r"$\log N(\varepsilon)$")
    ax.set_title(f"Box-counting dimension")
    sns.despine()
    save_fig(fig, "04_box_counting_dimension.png")

    pd.DataFrame({
        "box_size": sizes,
        "N_boxes": counts,
        "log_inverse_box_size": x,
        "log_N_boxes": y,
    }).to_csv(OUT_DIR / "box_counting_data.csv", index=False)

    return D


# ============================================================
# 4. Lacunarity
# ============================================================

def lacunarity_gliding_box(binary, box_sizes):
    img = binary.astype(float)
    values = []

    for r in box_sizes:
        kernel = np.ones((r, r), dtype=float)
        mass = ndi.convolve(img, kernel, mode="constant", cval=0)

        # only windows with foreground signal
        mass = mass[mass > 0]

        if len(mass) == 0:
            lac = np.nan
        else:
            lac = np.mean(mass ** 2) / (np.mean(mass) ** 2)

        values.append(lac)

    return np.array(values)


def plot_lacunarity(binary):
    lac = lacunarity_gliding_box(binary, LACUNARITY_BOX_SIZES)

    palette = sns.color_palette("mako", len(LACUNARITY_BOX_SIZES))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.lineplot(
        x=LACUNARITY_BOX_SIZES,
        y=lac,
        marker="o",
        markersize=8,
        lw=2.2,
        color=palette[-2],
        ax=ax
    )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Box size")
    ax.set_ylabel("Lacunarity")
    ax.set_title("Lacunarity across scales")
    sns.despine()
    save_fig(fig, "05_lacunarity_curve.png")

    pd.DataFrame({
        "box_size": LACUNARITY_BOX_SIZES,
        "lacunarity": lac
    }).to_csv(OUT_DIR / "lacunarity_data.csv", index=False)

    return lac


# ============================================================
# 5. Skeleton graph extraction
# ============================================================

def neighbor_count_8(skeleton):
    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=int)
    return ndi.convolve(skeleton.astype(int), kernel, mode="constant", cval=0)


def skeleton_to_graph(skeleton):
    """
    Convert skeleton to graph.

    Node pixels:
    - endpoint: neighbor count = 1
    - junction: neighbor count >= 3
    - isolated pixel: neighbor count = 0

    Degree-2 pixels are treated as edge chains.
    Junction clusters are merged into one node.
    """
    skel = skeleton.astype(bool)
    n_count = neighbor_count_8(skel)

    node_pixels = skel & (n_count != 2)
    labeled_nodes, n_nodes = measure.label(node_pixels, return_num=True, connectivity=2)

    G = nx.Graph()

    pixel_to_node = {}

    for node_id in range(1, n_nodes + 1):
        coords = np.argwhere(labeled_nodes == node_id)
        centroid = coords.mean(axis=0)

        local_neighbor_counts = n_count[labeled_nodes == node_id]
        if np.any(local_neighbor_counts >= 3):
            node_type = "junction"
        elif np.any(local_neighbor_counts == 1):
            node_type = "endpoint"
        else:
            node_type = "isolated"

        G.add_node(
            node_id,
            coord=tuple(centroid),
            pixel_count=len(coords),
            type=node_type
        )

        for r, c in coords:
            pixel_to_node[(r, c)] = node_id

    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    def neighbors(pixel):
        r, c = pixel
        out = []
        for dr, dc in directions:
            rr, cc = r + dr, c + dc
            if 0 <= rr < skel.shape[0] and 0 <= cc < skel.shape[1]:
                if skel[rr, cc]:
                    out.append((rr, cc))
        return out

    visited = set()

    for start_pixel, start_node in pixel_to_node.items():
        for nb in neighbors(start_pixel):
            edge_key = tuple(sorted([start_pixel, nb]))
            if edge_key in visited:
                continue

            visited.add(edge_key)

            path = [start_pixel, nb]
            prev = start_pixel
            curr = nb

            while True:
                # reached another graph node
                if curr in pixel_to_node and pixel_to_node[curr] != start_node:
                    end_node = pixel_to_node[curr]
                    length = len(path)

                    if start_node != end_node:
                        if G.has_edge(start_node, end_node):
                            G[start_node][end_node]["weight"] += length
                            G[start_node][end_node]["segments"] += 1
                        else:
                            G.add_edge(
                                start_node,
                                end_node,
                                weight=length,
                                segments=1
                            )
                    break

                nb_list = neighbors(curr)
                next_candidates = [p for p in nb_list if p != prev]

                if len(next_candidates) == 0:
                    break

                if len(next_candidates) > 1:
                    # Ambiguous tiny junction not captured by node cluster.
                    break

                nxt = next_candidates[0]
                edge_key = tuple(sorted([curr, nxt]))

                if edge_key in visited:
                    break

                visited.add(edge_key)
                path.append(nxt)

                prev, curr = curr, nxt

    return G, node_pixels, labeled_nodes


# ============================================================
# 6. Tree / hierarchy metrics
# ============================================================

def get_largest_component_subgraph(G):
    if G.number_of_nodes() == 0:
        return G.copy()

    components = list(nx.connected_components(G))
    largest = max(components, key=len)
    return G.subgraph(largest).copy()


def choose_root_node_by_position(G):
    """
    For this leaf vein image, choose the node closest to the bottom center
    as approximate petiole/root node.

    Image coordinate: row increases downward.
    """
    if G.number_of_nodes() == 0:
        return None

    coords = np.array([G.nodes[n]["coord"] for n in G.nodes])
    rows = coords[:, 0]
    cols = coords[:, 1]

    center_col = np.median(cols)
    score = rows - 0.25 * np.abs(cols - center_col)
    root_idx = np.argmax(score)

    return list(G.nodes)[root_idx]


def bfs_hierarchy_depth(G, root):
    """
    Compute BFS depth from root on unweighted topology.
    """
    if root is None or root not in G:
        return {}

    depths = nx.single_source_shortest_path_length(G, root)
    return depths


def compute_graph_metrics(G):
    degrees = dict(G.degree())
    degree_values = np.array(list(degrees.values())) if degrees else np.array([])

    node_types = nx.get_node_attributes(G, "type")
    endpoint_count = sum(1 for n, t in node_types.items() if t == "endpoint")
    junction_count = sum(1 for n, t in node_types.items() if t == "junction")
    isolated_count = sum(1 for n, t in node_types.items() if t == "isolated")

    n_components = nx.number_connected_components(G) if G.number_of_nodes() > 0 else 0
    cycle_rank = G.number_of_edges() - G.number_of_nodes() + n_components

    LCC = get_largest_component_subgraph(G)
    root = choose_root_node_by_position(LCC)
    depths = bfs_hierarchy_depth(LCC, root)

    if LCC.number_of_nodes() > 1:
        try:
            diameter = nx.diameter(LCC)
            avg_shortest_path = nx.average_shortest_path_length(LCC)
        except Exception:
            diameter = np.nan
            avg_shortest_path = np.nan
    else:
        diameter = np.nan
        avg_shortest_path = np.nan

    metrics = {
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "connected_components": n_components,
        "is_tree": nx.is_tree(G) if G.number_of_nodes() > 0 else False,
        "cycle_rank_loop_number": cycle_rank,
        "endpoint_count": endpoint_count,
        "junction_count": junction_count,
        "isolated_node_count": isolated_count,
        "mean_degree": float(degree_values.mean()) if len(degree_values) else np.nan,
        "max_degree": int(degree_values.max()) if len(degree_values) else np.nan,
        "largest_component_nodes": LCC.number_of_nodes(),
        "largest_component_edges": LCC.number_of_edges(),
        "largest_component_diameter": diameter,
        "largest_component_average_shortest_path": avg_shortest_path,
        "root_node": root,
        "max_bfs_depth_from_root": max(depths.values()) if depths else np.nan,
    }

    return metrics, degrees, LCC, root, depths


# ============================================================
# 7. Network visualization
# ============================================================

def plot_graph_nodes_by_degree(skeleton, G, degrees):
    palette = sns.color_palette("mako", as_cmap=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(skeleton, cmap="gray")

    if G.number_of_nodes() > 0:
        coords = np.array([G.nodes[n]["coord"] for n in G.nodes])
        node_ids = list(G.nodes)
        deg = np.array([degrees[n] for n in node_ids])

        sc = ax.scatter(
            coords[:, 1],
            coords[:, 0],
            c=deg,
            cmap=palette,
            s=18 + 12 * deg,
            edgecolors="white",
            linewidths=0.35,
            alpha=0.95
        )

        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Degree", fontsize=12)

    ax.set_title("Skeleton graph nodes colored by degree")
    ax.axis("off")
    save_fig(fig, "06_network_nodes_by_degree.png")


def plot_degree_distribution(degrees):
    degree_values = list(degrees.values())
    df = pd.DataFrame({"degree": degree_values})

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.countplot(
        data=df,
        x="degree",
        palette="mako",
        ax=ax
    )

    ax.set_xlabel("Node degree")
    ax.set_ylabel("Count")
    ax.set_title("Degree distribution")
    sns.despine()
    save_fig(fig, "07_degree_distribution.png")

    degree_table = (
        df.value_counts("degree")
        .reset_index(name="count")
        .sort_values("degree")
    )
    degree_table.to_csv(OUT_DIR / "degree_distribution.csv", index=False)


def plot_hierarchy_depth(skeleton, LCC, root, depths):
    palette = sns.color_palette("mako", as_cmap=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(skeleton, cmap="gray")

    if LCC.number_of_nodes() > 0 and depths:
        node_ids = list(depths.keys())
        coords = np.array([LCC.nodes[n]["coord"] for n in node_ids])
        depth_values = np.array([depths[n] for n in node_ids])

        sc = ax.scatter(
            coords[:, 1],
            coords[:, 0],
            c=depth_values,
            cmap=palette,
            s=24,
            edgecolors="white",
            linewidths=0.25
        )

        if root is not None:
            r, c = LCC.nodes[root]["coord"]
            ax.scatter(
                [c], [r],
                s=160,
                marker="*",
                color="white",
                edgecolors="black",
                linewidths=0.8
            )

        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("BFS depth", fontsize=12)

    ax.set_title("Hierarchy depth from approximate petiole node")
    ax.axis("off")
    save_fig(fig, "08_hierarchy_depth.png")


def plot_depth_distribution(depths):
    df = pd.DataFrame({"depth": list(depths.values())})

    if df.empty:
        return

    depth_counts = (
        df.value_counts("depth")
        .reset_index(name="count")
        .sort_values("depth")
    )

    n_colors = len(depth_counts)
    colors = sns.color_palette("mako", n_colors=n_colors + 2)[2:]

    fig, ax = plt.subplots(figsize=(6.2, 4.6))

    sns.barplot(
        data=depth_counts,
        x="depth",
        y="count",
        palette=colors,
        ax=ax
    )

    ax.set_xlabel("BFS depth from root")
    ax.set_ylabel("Node count")
    ax.set_title("Hierarchy depth distribution")

    # Reduce x-axis tick density
    max_depth = int(depth_counts["depth"].max())

    if max_depth <= 30:
        tick_step = 5
    elif max_depth <= 80:
        tick_step = 10
    else:
        tick_step = 20

    tick_depths = list(range(0, max_depth + 1, tick_step))

    # seaborn barplot uses categorical x positions: 0, 1, 2, ...
    all_depths = depth_counts["depth"].tolist()
    tick_positions = [
        all_depths.index(d) for d in tick_depths if d in all_depths
    ]
    tick_labels = [
        str(d) for d in tick_depths if d in all_depths
    ]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0)

    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)
    sns.despine()

    save_fig(fig, "09_hierarchy_depth_distribution.png")

    depth_counts.to_csv(
        OUT_DIR / "hierarchy_depth_distribution.csv",
        index=False
    )
    df.to_csv(
        OUT_DIR / "hierarchy_depths.csv",
        index=False
    )


# ============================================================
# 8. Main
# ============================================================

def main():
    img, gray, raw_binary, skeleton = preprocess_image(IMG_PATH)

    # Core intermediate outputs only
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(img)
    ax.set_title("Original image")
    ax.axis("off")
    save_fig(fig, "00_original.png")

    save_image(raw_binary, "01_raw_binary_mask.png", title="Raw binary vein mask")
    save_image(skeleton, "02_skeleton.png", title="Skeletonized vein network")

    # Porosity
    porosity, hull = compute_porosity(raw_binary)
    plot_porosity(raw_binary, hull, porosity)

    # Box-counting dimension
    box_D = plot_box_dimension(raw_binary)

    # Lacunarity
    lac = plot_lacunarity(raw_binary)

    # Network graph
    G, node_pixels, labeled_nodes = skeleton_to_graph(skeleton)
    metrics, degrees, LCC, root, depths = compute_graph_metrics(G)

    plot_graph_nodes_by_degree(skeleton, G, degrees)
    plot_degree_distribution(degrees)
    plot_hierarchy_depth(skeleton, LCC, root, depths)
    plot_depth_distribution(depths)

    # Summary table
    summary = {
        "image_path": str(IMG_PATH),
        "vein_area_px": int(raw_binary.sum()),
        "skeleton_length_px": int(skeleton.sum()),
        "convex_hull_area_px": int(hull.sum()),
        "porosity_convex_hull": porosity,
        "box_counting_dimension": box_D,
        "mean_lacunarity": float(np.nanmean(lac)),
        **metrics
    }

    pd.DataFrame([summary]).to_csv(OUT_DIR / "summary_metrics.csv", index=False)

    # Node table
    node_rows = []
    for n in G.nodes:
        r, c = G.nodes[n]["coord"]
        node_rows.append({
            "node_id": n,
            "row": r,
            "col": c,
            "degree": degrees.get(n, np.nan),
            "type": G.nodes[n].get("type", "NA"),
            "bfs_depth_from_root": depths.get(n, np.nan)
        })

    pd.DataFrame(node_rows).to_csv(OUT_DIR / "node_metrics.csv", index=False)

    # Edge table
    edge_rows = []
    for u, v, data in G.edges(data=True):
        edge_rows.append({
            "source": u,
            "target": v,
            "weight_pixel_length": data.get("weight", np.nan),
            "segments": data.get("segments", np.nan)
        })

    pd.DataFrame(edge_rows).to_csv(OUT_DIR / "edge_metrics.csv", index=False)

    print("\nDone. Results saved to:", OUT_DIR)
    print("\nKey summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()