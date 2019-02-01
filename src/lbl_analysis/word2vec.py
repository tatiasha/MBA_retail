import pandas as pd
import re
from sklearn.decomposition import PCA
from matplotlib import pyplot
from nltk.stem import SnowballStemmer
from gensim.models import Word2Vec
from scipy import spatial


def get_words(external_data, original_data):
    words = []
    tmp = []
    sentences = []
    for product in external_data:
        q = re.split(r'(-|/|,|%|!| |"."|&|™|®)', product)
        for word in q:
            if (len(word) > 2 and word != 'no') and not word.isdigit():
                if word.find("("):
                    word = word.replace("(", "")
                if word.find(")"):
                    word = word.replace(")", "")
                if word.find("'s"):
                    word = word.replace("'s", "")
                if word.find("s'"):
                    word = word.replace("s'", "")
                if word.find("."):
                    word = word.replace(".", "")
                if word.find("d'"):
                    word = word.replace("d'", "")
                if word.find("\\"):
                    word = word.replace("\\", "")
                if len(word) > 2 and not word.isdigit():
                    words.append(word)
                    tmp.append(word)
        sentences.append(tmp)
        tmp = []
    tmp = []
    for product in original_data:
        q = re.split(r'(-|/|,|%|!| |"."|&|™|®)', product)
        for word in q:
            if (len(word) > 2 and word != 'no') and not word.isdigit():
                if word.find("("):
                    word = word.replace("(", "")
                if word.find(")"):
                    word = word.replace(")", "")
                if word.find("'s"):
                    word = word.replace("'s", "")
                if word.find("s'"):
                    word = word.replace("s'", "")
                if word.find("."):
                    word = word.replace(".", "")
                if word.find("d'"):
                    word = word.replace("d'", "")
                if word.find("\\"):
                    word = word.replace("\\", "")
                if len(word) > 2 and not word.isdigit():
                    words.append(word)
                    tmp.append(word)
        sentences.append(tmp)
        tmp = []

    return list(set(words)), sentences


def get_data(path, name_to_return, delimiter=',', names=None):
    data = pd.read_csv(path, sep=delimiter, names=names)
    return data[name_to_return]


def process_original_data(data):
    product_name = []
    for product in data:
        product_name.append(product.lower())
    return product_name


def transform_data(data):
    stoplist = ['8\"', "18.2z"]
    snowball_stemmer = SnowballStemmer("english")
    new_data = []
    for product in data:
        new_product = ""
        q = re.split(r'(-|/|,|%|!| |"."|&|™|®)', product)
        for word in q:
            if not word.isdigit() and word not in stoplist:
                if word.find("("):
                    word = word.replace("(", "")
                if word.find(")"):
                    word = word.replace(")", "")
                if word.find("'s"):
                    word = word.replace("'s", "")
                if word.find("s'"):
                    word = word.replace("s'", "")
                if word.find("."):
                    word = word.replace(".", "")
                if word.find("d'"):
                    word = word.replace("d'", "")
                word = snowball_stemmer.stem(word)
                if len(word) > 2 and not word.isdigit():
                    new_product += word + " "
        if len(new_product[:-1]) > 2:
            new_data.append(new_product[:-1])
        else:
            new_data.append("-1")
    return new_data


def process_external_data(data):
    data = data.tolist()
    product_name = []
    for order in data:
        o = order.split(",")
        for word in o:
            if word not in product_name:
                product_name.append(word)
    return product_name


def draw(model):
    X = model[model.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    x = result[:, 0]
    y = result[:, 1]
    words = list(model.wv.vocab)

    x_ = []
    y_ = []
    w_ = []
    for i in range(len(x)):
        # if x[i] > 3:
        y_.append(y[i])
        x_.append(x[i])
        w_.append(words[i])
    x = x_
    y = y_
    words = w_

    pyplot.scatter(x, y)

    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(x[i], y[i]), size=15)

    pyplot.show()


def get_matrix(data, model):
    matrix = []
    voc = list(model.wv.vocab)
    l = len(model[voc[0]])
    for product in data:
        p = product.split(" ")
        vec = [0 for _ in range(l)]
        product_u = []
        for elem in p:
            if elem in voc and elem not in product_u:
                product_u.append(elem)
                vec += model[elem]
            else:
                print(elem)
        matrix.append(vec)
        # print(vec)
    return matrix


def comparing(original_data, external_data, model, ori, ex):
    original_matrix = get_matrix(original_data, model)
    external_matrix = get_matrix(external_data, model)

    # kaggle = []
    # grocery = []
    step = 0
    # closes = []
    # res_orig = []
    # res_external = []

    kaggle_orig = []
    name_1 = []
    name_1_w = []
    name_2 = []
    name_2_w = []
    name_3 = []
    name_3_w = []

    for product_id in range(len(original_matrix)):
        ma = [-100, -100, -100]
        ma_idx = [-100, -100, -100]
        for query_id in range(len(external_matrix)):
            d = original_matrix[product_id]
            diff = 1 - spatial.distance.cosine(d, external_matrix[query_id])
            # diff = 1 - spatial.distance.euclidean(d, external_matrix[query_id])
            for m in range(len(ma)):
                if diff > ma[m]:
                    ma[m] = diff
                    ma_idx[m] = query_id
                    break
        step += 1
        print("{0}/{1}; {2}; {3} = {4}".format(step, len(transformed_original_data), ma[0], transformed_original_data[product_id],
                                               transformed_external_data[ma_idx[0]]))

        # kaggle.append(transformed_original_data[product_id])
        # grocery.append(transformed_external_data[ma_idx])
        # closes.append(ma)
        # res_orig.append(ex[ma_idx])
        # res_external.append(ori[product_id])
        kaggle_orig.append(ori[product_id])
        name_1.append(ex[ma_idx[0]])
        name_1_w.append(ma[0])

        name_2.append(ex[ma_idx[1]])
        name_2_w.append(ma[1])

        name_3.append(ex[ma_idx[2]])
        name_3_w.append(ma[2])

    p = pd.DataFrame()
    # p["kaggle_compressed"] = kaggle
    # p["grocery_compressed"] = grocery
    # p["close_W2V"] = closes
    # p["orig_grocery"] = res_orig
    # p["orig_kaggle"] = res_external
    p["Original"] = kaggle_orig
    p["name_1"] = name_1
    p["name_1_w"] = name_1_w

    p["name_2"] = name_2
    p["name_2_w"] = name_2_w

    p["name_3"] = name_3
    p["name_3_w"] = name_3_w

    # p = p.sort_values(by="grocery_compressed")
    p.to_csv("../../data/transformation_w2v_v13.csv")
    # print(matching)
    print("Transformations are ready.{0} products were removed. {1} - before; {2} - after.".format(
        len(transformed_original_data) - len(p), len(transformed_original_data), len(p)))
    return p


def comparing_v2(transformed_original_data, transformed_external_data, model, original_data, external_data):
    original_matrix = get_matrix(transformed_original_data, model)
    external_matrix = get_matrix(transformed_external_data, model)

    kaggle = []
    grocery = []
    step = 0
    closes = []
    res_orig = []
    res_external = []
    model_voc = list(model.wv.vocab)
    l = len(model[model_voc[0]])
    for product_id in range(len(original_matrix)):
        step += 1
        words = transformed_original_data[product_id].split(" ")
        words_matrix = []
        for w in words:
            if w in model_voc:
                words_matrix.append(model[w])
        diff = []
        for word_m in words_matrix:
            tmp_diff = -100
            for external in external_matrix:
                tmp = 1 - spatial.distance.cosine(word_m, external)
                # tmp = spatial.distance.euclidean(word_m, external)
                if tmp > tmp_diff:
                    tmp_diff = tmp
            diff.append(tmp_diff)

        res_vec = [0 for _ in range(l)]
        for i in range(len(diff)):
            res_vec += words_matrix[i]*diff[i]

        diff_res = -1000
        diff_res_id = -1000
        for external_ in range(len(external_matrix)):
            tmp = 1 - spatial.distance.cosine(res_vec, external_matrix[external_])
            # tmp = spatial.distance.euclidean(res_vec, external_matrix[external_])

            if tmp > diff_res:
                diff_res = tmp
                diff_res_id = external_

        if diff_res != -1000 and diff_res_id != -1000:
            print("{0}/{1}; {2}; {3} = {4}".format(step, len(transformed_original_data), diff_res,
                                                   transformed_original_data[product_id],
                                                       transformed_external_data[diff_res_id]))
            kaggle.append(transformed_original_data[product_id])
            grocery.append(transformed_external_data[diff_res_id])
            closes.append(diff_res)
            res_orig.append(external_data[diff_res_id])
            res_external.append(original_data[product_id])

    p = pd.DataFrame()
    p["kaggle_compressed"] = kaggle
    p["grocery_compressed"] = grocery
    p["close_W2V"] = closes
    p["orig_grocery"] = res_orig
    p["orig_kaggle"] = res_external

    p = p.sort_values(by="grocery_compressed")
    p.to_csv("../../data/transformation_w2v_v5.csv")




    # for product_id in range(len(original_matrix)):
    #     ma = -100
    #     ma_idx = -100
    #     for query_id in range(len(external_matrix)):
    #         d = original_matrix[product_id]
    #         diff = 1 - spatial.distance.cosine(d, external_matrix[query_id])
    #         if diff > ma:
    #             ma = diff
    #             ma_idx = query_id
    #     step += 1
    #     print("{0}/{1}; {2}; {3} = {4}".format(step, len(transformed_original_data), ma, transformed_original_data[product_id],
    #                                            transformed_external_data[ma_idx]))
    #     kaggle.append(transformed_original_data[product_id])
    #     grocery.append(transformed_external_data[ma_idx])
    #     closes.append(ma)
    #     res_orig.append(ex[ma_idx])
    #     res_external.append(ori[product_id])
    #
    #
    # p = pd.DataFrame()
    # p["kaggle_compressed"] = kaggle
    # p["grocery_compressed"] = grocery
    # p["close_W2V"] = closes
    # p["orig_grocery"] = res_orig
    # p["orig_kaggle"] = res_external
    #
    # p = p.sort_values(by="grocery_compressed")
    # p.to_csv("../../data/transformation_w2v_v2.csv")
    # # print(matching)
    # print("Transformations are ready.{0} products were removed. {1} - before; {2} - after.".format(
    #     len(transformed_original_data) - len(p), len(transformed_original_data), len(p)))

    return p

if __name__ == "__main__":
    external_path = "../../data/groceries/groceries.csv"
    original_path = "../../data/kaggle/products.csv"

    external_data = get_data(external_path, 'products', delimiter=';', names=['products'])
    external_data = process_external_data(external_data)

    original_data = get_data(original_path, 'product_name')
    original_data = process_original_data(original_data.tolist())
    print("All data are collected. {0} - original products; {1} - external products".format(len(original_data),
                                                                                            len(external_data)))

    transformed_external_data = transform_data(external_data)
    transformed_original_data = transform_data(original_data)
    print("All data are transformed. {0} - original products; {1} - external products".format(
        len(transformed_original_data), len(transformed_external_data)))

    all_unique_words, sentences = get_words(transformed_external_data, transformed_original_data)
    print("Unique words are collected - {0} words".format(len(all_unique_words)))
    terms = all_unique_words

    dt = pd.DataFrame()
    dt["transformed"] = transformed_external_data
    dt["product_name"] = external_data
    # dt.to_csv("../../data/groceries_transformation.csv")

    df = pd.DataFrame()
    df["transformed"] = transformed_original_data
    df["product_name"] = original_data
    # df.to_csv("../../data/kaggle_transformation.csv")

    model = Word2Vec.load("tmp_model")
    draw(model)
    # print(sentences[:2])
    # model = Word2Vec(sentences, min_count=20, compute_loss=True, iter=10000, size=200)
    # print(model)
    # print(model.get_latest_training_loss())
    # model.save("model")
    #
    # # print(model['salt'])
    # #
    # comparing(transformed_original_data, transformed_external_data, model, original_data, external_data)









