import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


# ------------------------------
# NETWORK CENTRALITY CORRELATION PLOTS
# ------------------------------
def plot_centrality_correlation(G, path=""):
    # Create Boolean Variable
    directed = nx.is_directed(G)

    # For directed graphs
    if directed:
        df = pd.DataFrame.from_dict(
            {
                "In Degree Centrality": nx.in_degree_centrality(G),
                "Out Degree Centrality": nx.out_degree_centrality(G),
                # Defaults to inward closeness centrality
                "In Closeness Centrality": nx.closeness_centrality(G),
                # Reversing the Network will give us outward closeness centrality
                "Out Closeness Centrality": nx.closeness_centrality(G.reverse()),
                "Betweenness Centrality": nx.betweenness_centrality(G),
            }
        )

    # For undirected
    else:
        df = pd.DataFrame.from_dict(
            {
                "Degree": nx.degree_centrality(G),
                "Closeness": nx.closeness_centrality(G),
                "Betweenness": nx.betweenness_centrality(G),
            }
        )

    # Create plot
    sns.pairplot(df, diag_kind="hist")

    # Save plot if path exists
    if path != "":
        plt.savefig(path)

    # Display
    plt.show()


# ------------------------------
# AVERAGE DEGREE
# ------------------------------
def ave_degree(G):
    # Boolean
    directed = nx.is_directed(G)

    # Directed print statements
    if directed:
        print(
            "AVEREAGE IN DEGREE CONNECTIVITY: "
            + str(
                np.mean(list(nx.average_degree_connectivity(G, source="in").values()))
            )
        )
        print(
            "AVEREAGE OUT DEGREE CONNECTIVITY: "
            + str(
                np.mean(list(nx.average_degree_connectivity(G, source="out").values()))
            )
        )

    # Undirected
    else:
        print(
            "AVEREAGE DEGREE CONNECTIVITY: "
            + str(np.mean(list(nx.average_degree_connectivity(G).values())))
        )


# ------------------------------
# PLOT DEGREE DISTRIBUTION
# ------------------------------
def plot_degree_distribution(G, type="in", path="", fit=False):
    # From the example
    if not nx.is_directed(G):
        data = G.degree()
        type = ""

        # To label when we are using in/out degrees
        xlabel_start = ""
    else:
        if type == "in":
            data = G.in_degree()
            xlabel_start = "in "
        if type == "out":
            data = G.out_degree()
            xlabel_start = "out "

    # Create plot
    fig, axs = plt.subplots(1, 4)
    fig.set_size_inches(20, 5)

    # Help from https://stackoverflow.com/questions/65028854/plot-degree-distribution-in-log-log-scale
    # Also in my Lab 2.2
    import collections

    degree_sequence = sorted([d for _, d in data], reverse=True)
    degreeCount = collections.Counter(degree_sequence)

    aux_x, aux_y = zip(*degreeCount.items())
    aux_y = list(aux_y)

    axs[0].set_xlabel(xlabel_start + "Degree\n(log scale)")
    axs[0].set_ylabel("Probability\n(log scale)")

    # Adding trendline if fit
    if fit:
        # Log x and y
        log_aux_x = np.log(aux_x)
        log_aux_y = np.log(aux_y)

        # Find Trendline
        fit_coefficients = np.polyfit(log_aux_x, log_aux_y, deg=1)
        trendline = np.exp(np.poly1d(fit_coefficients)(log_aux_x))

        # Plot data
        axs[0].loglog(aux_x, aux_y, "-", color="orange")
        axs[0].loglog(aux_x, aux_y, "o", color="orange")

        # Plot trendline
        axs[0].loglog(aux_x, trendline, "-", color="blue", label="PDF Alpha")

        # Show the slope of the trendline for the Alpha value
        slope = fit_coefficients[0]

        # https://matplotlib.org/stable/tutorials/advanced/transforms_tutorial.html
        axs[0].text(
            0.65,
            0.95,
            f"Slope = {slope:.2f}",
            transform=axs[0].transAxes,
            ha="left",
            va="top",
            fontsize=12,
        )

        # Setting labels
        axs[0].set_xticklabels([-0.0001, 0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000])

    # Else is just the data, this time blue
    else:
        axs[0].loglog(aux_x, aux_y, "-", color="blue")
        axs[0].loglog(aux_x, aux_y, "o", color="blue")
        axs[0].set_xticklabels(np.arange(0, 500, 20))

    # Fixing labels
    y_ticks = np.arange(min(aux_y) - 1, max(aux_y) + 1, 0.5)
    axs[0].set_yticklabels(y_ticks)

    # Histogram
    axs[1].hist(degree_sequence, density=True, bins=30, color="blue")
    axs[1].set_ylabel("Probability")
    axs[1].set_xlabel(xlabel_start + "Degree")

    # Sort the degrees
    sorted_degree = np.sort(degree_sequence)

    # Calculate ccp values
    ccdf_degree = 1.0 - np.arange(len(sorted_degree)) / len(sorted_degree)

    # Plot
    axs[2].plot(sorted_degree, ccdf_degree, marker="o", color="blue")
    axs[2].plot(sorted_degree, ccdf_degree, color="orange")

    # Fix labels
    axs[2].set_xlabel(xlabel_start + "Degree")
    axs[2].set_ylabel("cCDF")
    axs[2].set_xticklabels(np.arange(-20, 200, 20))

    # If fit
    if fit:
        # log values
        log_sorted_degree = np.log(sorted_degree)
        log_ccdf_degree = np.log(ccdf_degree)

        # Find trendline
        fit_coefficients = np.polyfit(log_sorted_degree, log_ccdf_degree, deg=1)
        trendline = np.exp(np.poly1d(fit_coefficients)(log_sorted_degree))

        # Plot with trendline
        axs[3].loglog(sorted_degree, trendline, color="blue", label="cCDF Alpha")
        axs[3].loglog(sorted_degree, ccdf_degree, color="orange")

        # Show the slope of the trendline for the Alpha value
        slope = fit_coefficients[0]
        axs[3].text(
            0.65,
            0.95,
            f"Slope = {slope:.2f}",
            transform=axs[3].transAxes,
            ha="left",
            va="top",
            fontsize=12,
        )
        axs[3].set_xticklabels(np.arange(-20, 1000, 5))

    # Else plot just the data
    else:
        axs[3].loglog(sorted_degree, ccdf_degree, color="blue")
        axs[3].set_xticklabels(np.arange(-20, 500, 20))

    # Add labels
    axs[3].set_xlabel(xlabel_start + "Degree (log)")
    axs[3].set_ylabel("cCDF (log)")

    axs[3].set_yticklabels([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000])

    # Tidy up
    plt.tight_layout()

    # Save
    if path != "":
        plt.savefig(path)

    # Display
    plt.show()


# ------------------------------
# NETWORK PLOTTING FUNCTION
# ------------------------------
def plot_network(G, node_color="degree", layout="random"):
    # POSITIONS LAYOUT
    N = len(G.nodes)
    if layout == "spring":
        # pos=nx.spring_layout(G,k=50*1./np.sqrt(N),iterations=100)
        pos = nx.spring_layout(G)

    if layout == "random":
        pos = nx.random_layout(G)

    # INITALIZE PLOT
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 15)

    # NODE COLORS
    cmap = plt.cm.get_cmap("Greens")

    # DEGREE
    if node_color == "degree":
        centrality = list(dict(nx.degree(G)).values())

    # BETWENNESS
    if node_color == "betweeness":
        centrality = list(dict(nx.betweenness_centrality(G)).values())

    # CLOSENESS
    if node_color == "closeness":
        centrality = list(dict(nx.closeness_centrality(G)).values())

    # NODE SIZE CAN COLOR
    node_colors = [cmap(u / (0.01 + max(centrality))) for u in centrality]
    node_sizes = [4000 * u / (0.01 + max(centrality)) for u in centrality]

    # PLOT NETWORK
    nx.draw(
        G,
        with_labels=True,
        edgecolors="black",
        node_color=node_colors,
        node_size=node_sizes,
        font_color="white",
        font_size=18,
        pos=pos,
    )

    plt.show()


# ------------------------------
# NETWORK SUMMARY FUNCTION
# ------------------------------
def network_summary(G):
    def centrality_stats(x):
        x1 = dict(x)
        x2 = np.array(list(x1.values()))
        # print(x2)
        print("	min:", min(x2))
        print("	mean:", np.mean(x2))
        print("	median:", np.median(x2))
        # print("	mode:" ,stats.mode(x2)[0][0])
        print("	max:", max(x2))
        x = dict(x)
        sort_dict = dict(sorted(x1.items(), key=lambda item: item[1], reverse=True))
        print("	top nodes:", list(sort_dict)[0:6])
        print("	          ", list(sort_dict.values())[0:6])

    try:
        print("GENERAL")
        print("	number of nodes:", len(list(G.nodes)))
        print("	number of edges:", len(list(G.edges)))

        print("	is_directed:", nx.is_directed(G))
        print("	is_weighted:", nx.is_weighted(G))

        if nx.is_directed(G):
            print("IN-DEGREE (NORMALIZED)")
            centrality_stats(nx.in_degree_centrality(G))
            print("OUT-DEGREE (NORMALIZED)")
            centrality_stats(nx.out_degree_centrality(G))
        else:
            print("	number_connected_components", nx.number_connected_components(G))
            print("	number of triangle: ", len(nx.triangles(G).keys()))
            print("	density:", nx.density(G))
            print("	average_clustering coefficient: ", nx.average_clustering(G))
            print(
                "	degree_assortativity_coefficient: ",
                nx.degree_assortativity_coefficient(G),
            )
            print("	is_tree:", nx.is_tree(G))

            if nx.is_connected(G):
                print("	diameter:", nx.diameter(G))
                print("	radius:", nx.radius(G))
                print(
                    "	average_shortest_path_length: ",
                    nx.average_shortest_path_length(G),
                )

            # CENTRALITY
            print("DEGREE (NORMALIZED)")
            centrality_stats(nx.degree_centrality(G))

            print("CLOSENESS CENTRALITY")
            centrality_stats(nx.closeness_centrality(G))

            print("BETWEEN CENTRALITY")
            centrality_stats(nx.betweenness_centrality(G))
    except:
        print("unable to run")


# ------------------------------
# ISOLATE GCC
# ------------------------------
def isolate_GCC(G):
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    nodes_in_giant_comp = comps[0]
    return nx.subgraph(G, nodes_in_giant_comp)
