# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
#
# flag = 10
# if flag == 0:
#     conf = pd.read_csv('tmp/confidence_clusters_v1.csv')
#     plt.title('confidence_clusters')
# if flag == 1:
#     conf = pd.read_csv('tmp/confidence_train_v1.csv')
#     plt.title('confidence_train')
# if flag == 2:
#     conf = pd.read_csv('tmp/confidence_merge_average.csv')
#     plt.title('Altered train without clusters')
# if flag == 3:
#     conf = pd.read_csv('tmp/confidence_merge_clusters_average.csv')
#     plt.title('Altered train with clusters')
# if flag == 4:
#     conf = pd.read_csv('tmp/confidence_GK_clusters_average.csv')
#     plt.title('Altered train and grocery with clusters')
# if flag == 5:
#     conf = pd.read_csv('tmp/prior_train_statistic.csv')
#     plt.title('Train and Prior')
# if flag == 6:
#     conf = pd.read_csv('tmp/prior_train_statistic_extend.csv')
#     plt.title('Extended train and Prior')
#
# #conf = np.array(conf)
#
# # print("RESULT = ", sum / float(C))  # sum -> counter
# # print("Zeros", c_zeroz, '/', C)
# # counter = Counter(conf)
# # X = counter.values()
# # Y = counter.keys()
# data_path = "E:\Projects\MBA_retail\\tmp"
# conf1 = pd.read_csv('{0}/statistics result/train_statistic_5000_v1.csv'.format(data_path)).values
# conf2 = pd.read_csv('{0}/train_statistics_network.csv'.format(data_path)).values
# plt.hist(conf1, alpha = 0.5, label = 'train', normed=True)
# plt.hist(conf2, alpha = 0.5, label = 'extend', normed=True)
# plt.legend()
# # plt.boxplot([conf1, conf2], labels=['train', 'extend'])
# plt.xlabel('')
# plt.ylabel('frequency')
# plt.show()
# c2 = float(sum(conf2))/len(conf2)
# c1 = float(sum(conf1))/len(conf1)
# print(c2/c1)

import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
import plotly

# x = [1, 4, 8]
# y_5 = [2165, 544, 272]
# y_6 = [5198, 1306, 653]
# y_7 = [9730, 2445, 1222]
# y_8 = [15115, 3827, 1899]
#
import matplotlib.pyplot as plt
#
# plt.plot(x, y_5, label='m=5',  marker=".")
# plt.plot(x, y_6, label='m=6',  marker=".")
# plt.plot(x, y_7, label='m=7',  marker=".")
# plt.plot(x, y_8, label='m=8',  marker=".")
# plt.xlabel("Число ядер CPU")
# plt.ylabel("Время, с")
# plt.xlim((0, 8.5))
# plt.ylim((0, 16000))
# plt.grid()
# plt.legend(loc='best')
# plt.show()

import pandas as pd
data = pd.read_csv("C:\\Users\\Tatiana\\Downloads\\Telegram Desktop\\f.csv", delimiter=";", header=-1)
print(data.head())
plt.plot(data[0], data[1])
plt.title("Изменение фитнес-функции")
plt.xlabel("Количество итераций")
plt.ylabel("Время выполнения, с")
plt.xlim((0, 100))
plt.grid()
plt.ylim((1900, 2500))
plt.show()
#
#
# draw_5 = go.Scatter(
#         x=x,
#         y=y_5,
#         name='m = 5'
#     )
#
#
# draw_6 = go.Scatter(
#         x=x,
#         y=y_6,
#         name='m = 6'
#     )
# draw_7 = go.Scatter(
#         x=x,
#         y=y_7,
#         name='m = 7'
#     )
#
#
# draw_8 = go.Scatter(
#         x=x,
#         y=y_8,
#         name='m = 8',
#     )
#
# layout = go.Layout(
#     xaxis=dict(
#         title='Число ядер CPU',
#         titlefont=dict(
#             family='Times New Roman',
#             size=18,
#             color='black'
#         ),
#         showgrid=True,
#         tick0=0,
#         dtick=1,
#         ticklen=0,
#         tickcolor='#000',
#         position=0.035,
#         hoverformat=','
#
#     ),
#     yaxis=dict(
#         title='Время, c',
#         titlefont=dict(
#             family='Times New Roman',
#             size=18,
#             color='black'
#
#         ),
#         showgrid=True,
#
# ),
#     legend=dict(
#         x=0.88,
#         y=1,
#         traceorder='normal',
#         font=dict(
#             family='Times New Roman',
#             size=16,
#             color='black'
#         ),
#         bgcolor='white',
#         bordercolor='#FFFFFF',
#         borderwidth=2
#     )
# )
# fig = go.Figure(data=[draw_5, draw_6, draw_7, draw_8], layout=layout)
# plotly.offline.plot(fig)
