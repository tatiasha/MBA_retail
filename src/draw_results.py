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
    train = pd.read_csv("../data/inter_statistics_train_all_rules.csv")
    train_ext = pd.read_csv("../data/inter_statistics_train_adapt_all_rules.csv")
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
    plt.savefig("../data/train_extend_recomm_density__all_rules.png")
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

    plt.figure(figsize=(3, 5))
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
    groc_full = pd.read_csv("../data/inter_statistics_train_all_rules.csv")
    groc_adapted = pd.read_csv("../data/inter_statistics_train_adapt_all_rules.csv")
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
    plt.savefig("../data//train_extend_recomm_boxplot_all_rules.png")
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
    distances = np.load("E:\Projects\MBA_retail\\tmp\\distance_rules_kaggle_adapted_v2.npy")
    distances2 = np.load("E:\Projects\MBA_retail\\tmp\\distance_rules_train_v2.npy")

    print("train vs extended m2:")
    d1 = np.mean(distances)
    d2 = np.mean(distances2)
    print("train mean = {}".format(d1))
    print("extended mean = {}".format(d2))

    profit = 1 - d2 / d1
    print("profit = {}".format(profit))
    print()

    plt.figure(figsize=(3, 5))
    plt.boxplot([distances, distances2], showfliers=False, positions=[0.0, 0.2], showmeans=True, labels=['K train', 'K adapt'])
    plt.ylabel("Confidence distance, CD", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.xlim(-0.1, 0.3)
    plt.tight_layout()
    plt.savefig("E:\Projects\MBA_retail\\tmp\\train_extend_confidence_boxplot_v2_1.png")
    plt.close()


# метрика по дистанции конфиденца
def calculate(name):
    data = pd.read_csv("E:\Projects\MBA_retail\\data\\{0}.csv".format(name))
    data = data.drop(
        ['Unnamed: 0', 'antecedent support', 'consequent support', 'support', 'lift', 'leverage', 'conviction'], axis=1)
    data_prior = pd.read_csv("E:\Projects\MBA_retail\\data\\rules_prior_5m.csv")
    data_prior = data_prior.drop(
        ['Unnamed: 0', 'antecedent support', 'consequent support', 'support', 'lift', 'leverage', 'conviction'], axis=1)

    distances = []
    c = 0
    for d in data_prior.values:
        print(c, len(data_prior))
        c += 1
        dx = d[0]
        dy = d[1]
        dc = d[2]
        for p in data.values:
            px = p[0]
            py = p[1]
            pc = p[2]
            if dx == px and dy == py:
                distances.append(np.abs(dc - pc))
    distances = np.array(distances)
    np.save("E:\Projects\MBA_retail\\data\\distance_{0}".format(name), distances)

def comparison():
    data_rec_ = pd.read_csv("../data/rec_all_rules_train.csv", delimiter=";", header=0)
    data_rec = [i[0] for i in data_rec_.values]
    data_adapt_rec_ = pd.read_csv("../data/rec_all_rules_train_adapt.csv", delimiter=";", header=0)
    data_adapt_rec = [i[0] for i in data_adapt_rec_.values]


    data_pr = pd.read_csv("../data/pr_all_rules_train.csv", header=0)
    data_adapt_pr = pd.read_csv("../data/pr_all_rules_train_adapt.csv", delimiter=";", header=0)

    data_f1_ = pd.read_csv("../data/f1_all_rules_train.csv", header=0)
    data_f1 = [i[0] for i in data_f1_.values]

    data_adapt_f1_ = pd.read_csv("../data/f1_all_rules_train_adapt.csv", delimiter=";", header=0)
    data_adapt_f1 = [i[0] for i in data_adapt_f1_.values]


    plt.figure(figsize=(3, 5))
    plt.boxplot([data_f1, data_adapt_f1], positions=[0.0, 0.2], showmeans=True, labels=['Init', 'Adapt'])
    plt.ylabel("F1", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.xlim(-0.1, 0.3)
    plt.tight_layout()
    plt.show()
    plt.close()
    # plt.bar([0], sum(data_rec.as_matrix()), label="Init recall", alpha=0.5)
    # plt.bar([1], sum(data_adapt_rec.as_matrix()), label="Adapt recall", alpha=0.5)
    #
    # plt.bar([3], sum(data_pr.as_matrix()), label="Init precision", alpha=0.5)
    # plt.bar([4], sum(data_adapt_pr.as_matrix()), label="Adapt precision", alpha=0.5)
    #
    # plt.bar([7], sum(data_f1.as_matrix()), label="Init f1", alpha=0.5)
    # plt.bar([8], sum(data_adapt_f1.as_matrix()), label="Adapt f1", alpha=0.5)
    #
    # plt.legend()
    # plt.savefig("../data/{0}_all_rules.png".format(name))
    # plt.close()


def all_metrics():
    data_recall_ = pd.read_csv("../data/rec_all_rules_train.csv", header=0)
    data_recall = [i[0] for i in data_recall_.values]

    data_recall_a_ = pd.read_csv("../data/rec_all_rules_train_adapt.csv", header=0)
    data_recall_a = [i[0] for i in data_recall_a_.values]

    data_precision_ = pd.read_csv("../data/pr_all_rules_train.csv", header=0)
    data_precision = [i[0] for i in data_precision_.values]

    data_precision_a_ = pd.read_csv("../data/pr_all_rules_train_adapt.csv", header=0)
    data_precision_a = [i[0] for i in data_precision_a_.values]

    data_f1_ = pd.read_csv("../data/f1_all_rules_train.csv", header=0)
    data_f1 = [i[0] for i in data_f1_.values]

    data_f1_a_ = pd.read_csv("../data/f1_all_rules_train_adapt.csv", header=0)
    data_f1_a = [i[0] for i in data_f1_a_.values]

    data_m2_ = pd.read_csv("../data/inter_all_rules_train.csv", header=0)
    data_m2 = [i[0] for i in data_m2_.values]

    data_m2_a_ = pd.read_csv("../data/inter_all_rules_train_adapt.csv", header=0)
    data_m2_a = [i[0] for i in data_m2_a_.values]

    # plt.figure(figsize=(3, 5))
    plt.boxplot([data_recall, data_recall_a], positions=[0.0, 0.2], showmeans=True, labels=['recall', 'Arecall'], meanline=True)
    plt.boxplot([data_precision, data_precision_a], positions=[0.5, 0.7], showmeans=True, labels=['precision', 'Aprecision'], meanline=True)
    plt.boxplot([data_f1, data_f1_a], positions=[1.0, 1.2], showmeans=True, labels=['F1', 'AF1'], meanline=True)
    plt.boxplot([data_m2, data_m2_a], positions=[1.5, 1.7], showmeans=True, labels=['M2', 'AM2'], meanline=True)

    plt.ylabel("Values", fontsize=14)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    plt.legend()
    plt.xlim(-0.1, 2)
    # plt.tight_layout()
    plt.show()


def count_distr():
    train_ = pd.read_csv("../data/number_recommendation_all_rules_train.csv", delimiter=";", header=0)
    train = [i[0] for i in train_.values]

    prior_ = pd.read_csv("../data/number_recommendation_all_rules_prior.csv", delimiter=";", header=0)
    prior = [i[0] for i in prior_.values]

    train_a_ = pd.read_csv("../data/number_recommendation_all_rules_train_adapt.csv", delimiter=";", header=0)
    train_a = [i[0] for i in train_a_.values]

    plt.hist(train, label="train", alpha=0.5, color='red', align='left')
    plt.hist(prior, label="prior", alpha=0.5, color='blue', align='left')
    plt.hist(train_a, label="Atrain", alpha=0.5, color='green', align='left')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # comparison()

    all_metrics()
    # count_distr()

    #exp1_hist()
    # calculate("rules_kaggle_adapted")
    # calculate("rules_train")
    # exp2_hist()
    #exp1_m1()
    # exp2_m1()
    # exp1_m2()
    # exp2_m2()