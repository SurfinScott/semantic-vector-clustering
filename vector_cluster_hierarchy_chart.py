""" vector_cluster_hierarchy_chart.py

S. Kurowski, July 2024

Decode and chart an OPTICS clustered hierarchy tree circularly.
Parent clusters are added and "sum over" their children clusters.
Charted node size is proportional to the log scale of the
cluster's vector membership count.

No guarantees or warranties are made in this demo.
"""

# pylint: disable-msg=C0103,C0301,W0621,W0718

import math
import networkx as nx
import matplotlib.pyplot as plt


def decode_cluster_hierarchy(h: list[list]) -> list[dict]:
    """Decode OPTICS heirarchy h into labeled node-graph clusters list."""
    clusters = []
    for ix, c in enumerate(h):
        n = c[1] - c[0] + 1
        cluster = {
            "_id": ix,
            "L": c[0],
            "H": c[1],
            "members": n,
            "parent": [],
            "children": [],
        }
        for iy, cc in enumerate(reversed(h[:ix])):
            L, H = cc[0], cc[1]
            if L >= c[0] and L <= c[1] or H >= c[0] and H <= c[1]:
                L = len(h[:ix]) - iy - 1
                if len(clusters[L]["parent"]) == 0:
                    cluster["children"].append(L)
                    clusters[L]["parent"].append(ix)
                elif ix < min(clusters[L]["parent"]):
                    cluster["children"].append(L)
                    clusters[L]["parent"].append(ix)
        clusters.append(cluster)

    # dedup grandparents by selecting first/lowest parent branch
    # while counting parent nodes in graph.
    n_parents = 0
    for ix, c in enumerate(clusters):
        n_parents = n_parents + 1 if len(c["children"]) > 0 else n_parents
        clusters[ix]["parent"] = min(c["parent"]) if len(c["parent"]) > 0 else None

    # add node_labels and node_sizes
    n_parents_traversed = 0
    for ix, c in enumerate(clusters):
        if len(c["children"]) > 0:
            # label parent nodes A,B,C,...
            clusters[ix]["node_label"] = chr(
                ord("A") + n_parents - n_parents_traversed - 1
            )
            n_parents_traversed += 1
        else:
            # label child leaf node numbers 0,1,2,...
            clusters[ix]["node_label"] = f"{ix - n_parents_traversed}"
        clusters[ix]["node_size"] = int(100 * math.log(c["members"]))

    return clusters


def chart_cluster_hierarchy(
    h: list[list] | None = None,
    clusters: list[dict] | None = None,
    save_file_name: str | None = None,
) -> list:
    """Chart hierarchy tree h or its decoded clusters."""

    assert h is not None or clusters is not None
    # if we already decoded the clusters, use it, else get from h
    if h is not None:
        clusters = decode_cluster_hierarchy(h)

    G = nx.DiGraph()
    node_sizes = {}
    labels = {}
    for ix, c in enumerate(clusters):
        node_sizes[ix] = c["node_size"]
        labels[ix] = c["node_label"]
        for cc in c["children"]:
            w = clusters[cc]["members"]
            G.add_edge(ix, cc, weight=w)

    nodelist, node_sizes = zip(*node_sizes.items())

    nx.draw_circular(
        G,
        node_color="lightgreen",
        node_size=node_sizes,
        labels=labels,
        arrows=True,
        nodelist=nodelist,
    )

    # save to file, or show and pause until chart is closed by user
    plt.plot()
    if save_file_name is not None:
        plt.savefig(save_file_name)
    else:
        plt.show(block=True)
    plt.close()

    return clusters


if __name__ == "__main__":
    # test values
    h = [
        [530, 561],
        [530, 587],
        [1117, 1153],
        [1116, 1191],
        [1860, 1890],
        [2535, 2565],
        [2534, 2589],
        [3005, 3094],
        [3095, 3125],
        [2991, 3128],
        [3912, 3947],
        [6432, 6461],
        [6684, 6711],
        [7422, 7466],
        [7837, 7883],
        [10040, 10066],
        [10067, 10104],
        [10040, 10132],
        [10900, 10929],
        [11133, 11162],
        [13522, 13578],
        [13584, 13611],
        [14163, 14192],
        [18755, 18779],
        [20332, 20368],
        [21234, 21262],
        [33374, 33400],
        [36269, 36296],
        [43860, 44288],
        [0, 45016],
    ]

    h = [[1843, 1852], [1972, 1983], [4471, 4485], [5689, 5712], [0, 5716]]

    h = [
        [39, 43],
        [39, 52],
        [116, 121],
        [122, 127],
        [116, 128],
        [356, 359],
        [447, 451],
        [555, 558],
        [705, 714],
        [890, 905],
        [1034, 1038],
        [1050, 1053],
        [1125, 1130],
        [1165, 1169],
        [1240, 1245],
        [1400, 1411],
        [1579, 1584],
        [1918, 1925],
        [2083, 2086],
        [2216, 2221],
        [2309, 2312],
        [2313, 2326],
        [3523, 3528],
        [3662, 3671],
        [4008, 4012],
        [4110, 4114],
        [4437, 4440],
        [4697, 4711],
        [5687, 5710],
        [0, 5716],
    ]

    chart_cluster_hierarchy(h)
