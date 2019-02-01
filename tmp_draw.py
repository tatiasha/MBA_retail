# import plotly.graph_objs as go
# import plotly
# import pandas as pd
#
# data = pd.read_csv("data/retail_exp_res_0612_1239.csv")
# print(data.head())
# t = data.describe()
# print(t['users']['min'])
# users = data['users'].unique()
# # user, res_average, res_min, res_max
# data_result_stat = pd.DataFrame(columns=["users", 'result_average', 'result_min', 'result_max'])
# data_comptime_stat = pd.DataFrame(columns=["users", 'comptime_average', 'comptime_min', 'comptime_max'])
#
# for u in users:
#     data_tmp = data.loc[data['users'] == u]
#     data_tmp_info = data_tmp.describe()
#
#     average_result = data_tmp_info['result']['mean']
#     data_result_stat = data_result_stat.append({'users': u, 'result_average': average_result,
#                                                 'result_min': average_result - data_tmp_info['result']['min'],
#                                                 'result_max': data_tmp_info['result']['max'] - average_result},
#                                                ignore_index=True)
#
#     average_comptime = data_tmp_info['comptime']['mean']
#     data_comptime_stat = data_comptime_stat.append({'users': u, 'comptime_average': average_comptime,
#                                                     'comptime_min': average_comptime - data_tmp_info['comptime']['min'],
#                                                     'comptime_max': data_tmp_info['comptime']['max'] - average_comptime},
#                                                    ignore_index=True)
#
# data_result = [
#     go.Scatter(
#         x=data_result_stat['users'].tolist(),
#         y=data_result_stat['result_average'].tolist(),
#         error_y=dict(
#             type='data',
#             symmetric=False,
#             array=data_result_stat['result_max'].tolist(),
#             arrayminus=data_result_stat['result_min'].tolist(),
#             visible=True
#         )
#     )
# ]
#
# data_comptime = [go.Scatter(
#         x=data_comptime_stat['users'].tolist(),
#         y=data_comptime_stat['comptime_average'].tolist(),
#         error_y=dict(
#             type='data',
#             symmetric=False,
#             array=data_comptime_stat['comptime_max'].tolist(),
#             arrayminus=data_comptime_stat['comptime_min'].tolist(),
#             visible=True,
#             color = 'green'
#         ),
#         line = dict(
#             color = 'green')
#     )
# ]
#
# layout = go.Layout(
#     # title='Comptime',
#     xaxis=dict(
#         title='Количество пользователей',
#         titlefont=dict(
#             family='Courier New, monospace',
#             size=18,
#             color='#7f7f7f'
#         )
#     ),
#     yaxis=dict(
#         title='Время планирования',
#         titlefont=dict(
#             family='Courier New, monospace',
#             size=18,
#             color='#7f7f7f'
#         )
#     )
# )
# # fig = go.Figure(data=data_comptime, layout=layout)
# # plotly.offline.plot(fig)
#
# layout = go.Layout(
#     # title='Результат',
#     xaxis=dict(
#         title='Количество пользователей',
#         titlefont=dict(
#             family='Courier New, monospace',
#             size=18,
#             color='#7f7f7f'
#         )
#     ),
#     yaxis=dict(
#         title='Оценка времени выполнения',
#         titlefont=dict(
#             family='Courier New, monospace',
#             size=18,
#             color='#7f7f7f'
#         )
#     ),
#
# )
# fig = go.Figure(data=data_result, layout=layout)
# plotly.offline.plot(fig)
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.decomposition import TruncatedSVD
import numpy as np
# x = ["W2V", "LSI", "WO"]
# y = [64, 25, 3]
# plt.bar(x, y)
# plt.show()

# def divide_matrix(matrix):
#     for row in range(len(matrix)):
#         for col in range(len(matrix[0])):
#             if matrix[row][col]:
#                 matrix[row][col] = 1/matrix[row][col]
#     return matrix
#
# def SVD(A, rank):
#     svd = TruncatedSVD(n_components=rank)
#     U = svd.fit_transform(A)
#     U = np.asmatrix(U)
#
#     S = svd.explained_variance_ratio_
#     S = np.diag(S)
#
#     VT = svd.components_
#     VT = np.asmatrix(VT)
#     V = VT.transpose()  # set of d
#     S_1 = divide_matrix(S)
#     return U, S_1, V
#
# model = Word2Vec.load("data/word2vec_original.model")
# voc = list(model.wv.vocab)
# matrix_voc = []
# for i in voc:
#     matrix_voc.append(model[i])
#
# u, s, v = SVD(matrix_voc, 3)
#
# x = u[:, 0]
# y = u[:, 1]
# z = u[:, 2]
# xs = [i[0] for i in x.tolist()]
# ys = [i[0] for i in y.tolist()]
# zs = [i[0] for i in z.tolist()]
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(xs, ys)
# for x, y, n in zip(xs, ys, voc):  # plot each point + it's index as text above
#     ax.text(x, y, '%s' % n, size=12, color='black')
# plt.show()
data = pd.read_csv("data//silhouette_clusters.csv", delimiter=';')
print(data.head())
x = data['n_clusters'].tolist()
min_vals = data['Min'].tolist()
mean_vals = data['Mean'].tolist()
t = x

fig, ax1 = plt.subplots()

ax1.set_xlabel('Number of clusters')
ax1.set_ylabel('Minimum values', color='blue')
ax1.plot(t, min_vals, color='blue')
ax1.tick_params(axis='y')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel('Mean values', color='orange')  # we already handled the x-label with ax1
ax2.plot(t, mean_vals, color='orange')
ax2.tick_params(axis='y')

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.grid()
plt.xlim((2, 30))
plt.show()