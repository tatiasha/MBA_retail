import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def exp1_hist():
    groc_full = pd.read_csv("tmp\\res\\paper\\groc_statistic_5000_v2.csv")
    groc_adapted = pd.read_csv("tmp\\res\\paper\\cleaned_groceries_statistics_5000_v2.csv")
    size1 = len(groc_full.values)
    size2 = len(groc_adapted.values)
    groc_full = groc_full.as_matrix().reshape(size1)
    groc_adapted = groc_adapted.as_matrix().reshape(size2)

    plt.figure(figsize=(6,5))
    plt.hist(groc_full, bins=20, alpha=0.5, density=1, align='mid', facecolor="#2244FF", edgecolor='black', label="G orig")
    plt.hist(groc_adapted, bins=20, alpha=0.5, density=1, align='mid', facecolor="#FF4422", edgecolor='black', label="G adapt")
    plt.ylabel("Density", fontsize=14)
    plt.xlabel("Recommendation conformity, RC", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("D:\Papers\ysc2018\market\\result\\groc_adapted_recomm_density.png")
    # plt.show()
    plt.close()

def exp2_hist():
    train = pd.read_csv("E:\Projects\MBA_retail\\tmp\\change_train_statistics_update_v2.csv")
    train_ext = pd.read_csv("E:\Projects\MBA_retail\\tmp\\train_statistics_update_v2.csv")
    size1 = len(train.values)
    size2 = len(train_ext.values)
    train = train.as_matrix().reshape(size1)
    train_ext = train_ext.as_matrix().reshape(size2)
    plt.figure(figsize=(6,5))
    plt.hist(train, bins=20, alpha=0.5, density=1, align='mid', facecolor="#2244FF", edgecolor='black', label="K train")
    plt.hist(train_ext, bins=20, alpha=0.5, density=1, align='mid', facecolor="#FF4422", edgecolor='black', label="K adapt")
    plt.ylabel("Density", fontsize=14)
    plt.xlabel("Recommendation conformity, RC", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("E:\Projects\MBA_retail\\tmp\\train_extend_recomm_density_v2.png")
    plt.close()

def exp1_m1():
    groc_full = pd.read_csv("tmp\\res\\paper\\groc_statistic_5000_v2.csv")
    groc_adapted = pd.read_csv("tmp\\res\\paper\\cleaned_groceries_statistics_5000_v2.csv")
    size1 = len(groc_full.values)
    size2 = len(groc_adapted.values)
    groc_full = groc_full.as_matrix().reshape(size1)
    groc_adapted = groc_adapted.as_matrix().reshape(size2)

    print("Groc vs cleaned groc m1:")
    groc_mean = np.mean(groc_full)
    cleaned_mean = np.mean(groc_adapted)
    print("groc mean = {}".format(groc_mean))
    print("cleaned mean = {}".format(cleaned_mean))

    profit = cleaned_mean / groc_mean - 1
    print("profit = {}".format(profit))
    print()

    plt.figure(figsize=(3,5))
    plt.boxplot([groc_full, groc_adapted], positions=[0.0, 0.2], showmeans=True, labels=['G orig', 'G adapt'])
    plt.ylabel("Recommendation conformity, RC", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.xlim(-0.1, 0.3)
    plt.tight_layout()
    plt.savefig("D:\Papers\ysc2018\market\\result\\groc_adapted_recomm_boxplot.png")
    plt.close()

def exp2_m1():
    groc_full = pd.read_csv("E:\Projects\MBA_retail\\tmp\\change_train_statistics_update_v2.csv")
    groc_adapted = pd.read_csv("E:\Projects\MBA_retail\\tmp\\train_statistics_update_v2.csv")
    size1 = len(groc_full.values)
    size2 = len(groc_adapted.values)
    groc_full = groc_full.as_matrix().reshape(size1)
    groc_adapted = groc_adapted.as_matrix().reshape(size2)

    print("train vs extended train m1:")
    groc_mean = np.mean(groc_full)
    cleaned_mean = np.mean(groc_adapted)
    print("groc mean = {}".format(groc_mean))
    print("cleaned mean = {}".format(cleaned_mean))

    profit = cleaned_mean / groc_mean - 1
    print("profit = {}".format(profit))
    print()

    plt.figure(figsize=(3,5))
    plt.boxplot([groc_full, groc_adapted], positions=[0.0, 0.2], showmeans=True, labels=['K train', 'K adapt'])
    plt.ylabel("Recommendation conformity, RC", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.xlim(-0.1, 0.3)
    plt.tight_layout()
    plt.savefig("E:\Projects\MBA_retail\\tmp\\train_extend_recomm_boxplot_v2.png")
    plt.close()

def exp1_m2():
    distances = np.load("tmp/res/paper/groc_distances.npy")
    distances2 = np.load("tmp/res/paper/groc_distances2.npy")

    print("groc vs cleaned m2:")
    d1 = np.mean(distances)
    d2 = np.mean(distances2)
    print("groc mean = {}".format(d1))
    print("cleaned mean = {}".format(d2))

    profit = 1 - d2 / d1
    print("profit = {}".format(profit))
    print()

    plt.figure(figsize=(3,5))
    plt.boxplot([distances, distances2], positions=[0.0, 0.2], showmeans=True, labels=['G orig', 'G adapt'])
    plt.ylabel("Confidence distance, CD", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.xlim(-0.1, 0.3)
    plt.tight_layout()
    plt.savefig("D:\Papers\ysc2018\market\\result\\groc_adapt_confidence_boxplot_v2.png")
    plt.close()

def exp2_m2():
    distances = np.load("E:\Projects\MBA_retail\\tmp\\distance_rules_cluster_integrated_train_network.npy")
    distances2 = np.load("E:\Projects\MBA_retail\\tmp\\distance_rules_change_train.npy")

    print("train vs extended m2:")
    d1 = np.mean(distances)
    d2 = np.mean(distances2)
    print("train mean = {}".format(d1))
    print("extended mean = {}".format(d2))

    profit = 1 - d2 / d1
    print("profit = {}".format(profit))
    print()

    plt.figure(figsize=(3,5))
    plt.boxplot([distances, distances2], showfliers=False, positions=[0.0, 0.2], showmeans=True, labels=['K train', 'K adapt'])
    plt.ylabel("Confidence distance, CD", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.xlim(-0.1, 0.3)
    plt.tight_layout()
    plt.savefig("E:\Projects\MBA_retail\\tmp\\train_extend_confidence_boxplot_v2.png")
    plt.close()


# метрика по дистанции конфиденца
def calculate(name):
    data = pd.read_csv("E:\Projects\MBA_retail\\tmp\\rules\\{0}.csv".format(name))
    data = data.drop(
        ['Unnamed: 0', 'antecedent support', 'consequent support', 'support', 'lift', 'leverage', 'conviction'], axis=1)
    data_prior = pd.read_csv("E:\Projects\MBA_retail\\tmp\\rules\\rules_prior.csv")
    data_prior = data_prior.drop(
        ['Unnamed: 0', 'antecedent support', 'consequent support', 'support', 'lift', 'leverage', 'conviction'], axis=1)
    distances = []
    c = 0
    for d in data.values:
        print(c)
        c += 1
        dx = d[0]
        dy = d[1]
        dc = d[2]
        for p in data_prior.values:
            px = p[0]
            py = p[1]
            pc = p[2]
            if dx == px and dy == py:
                distances.append(np.abs(dc - pc))
    distances = np.array(distances)
    np.save("E:\Projects\MBA_retail\\tmp\\distance_{0}".format(name), distances)

if __name__ == "__main__":
    #exp1_hist()
    #calculate("rules_cluster_integrated_train_network_v2")
    #calculate("rules_change_train_v2")
    exp2_hist()
    #exp1_m1()
    exp2_m1()
    #exp1_m2()
    exp2_m2()