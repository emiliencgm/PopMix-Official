'''
1
Long-tail 10 groups
'''
import numpy as np
import matplotlib.pyplot as plt

'''
Yelp2018:
LightGCN-0.000005702,	0.00001772,	0.000064,	0.0001413,	0.0002997,	0.0006002,	0.001386,	0.003401,	0.007393,	0.04926
SGL-RW-0,	0.00001535,	0.0000411,	0.0001719,	0.0003512,	0.0005194,	0.001369,	0.003231,	0.007915,	0.05044
SimGCL-0.000005263,	0.00002747,	0.00007698,	0.0001853,	0.0004525,	0.0008153,	0.001723,	0.004312,	0.01006,	0.05004
LightGCL-0.0000656,	0.0001132,	0.00005836,	0.0002223,	0.000241,	0.0004735,	0.001254,	0.00297,	0.007927,	0.0551
BCloss-0.0006483,0.0003096,0.0002075,0.0002884,0.0004783,0.0007039,0.00146,0.003008,0.00687,0.05686
Ours-0.001545,0.0006367,0.000472,0.0006819,0.000797,0.001185,0.002056,0.003582,0.007786,0.05335

Gowalla:
LightGCN-0.001103,	0.001002,	0.001219,	0.001417,	0.001776,	0.002925,	0.005581,	0.008757,	0.019,	0.1326
SGL-RW-0.001617,	0.001596,	0.001725,	0.002022,	0.002393,	0.003554,	0.005804,	0.008838,	0.01796,	0.1305
SimGCL-0.00194,	0.001975,	0.001908,	0.002388,	0.002689,	0.003977,	0.006666,	0.009999,	0.01942,	0.1282
LightGCL-0.001586,0.001369,0.001259,0.001558,0.00201,0.002946,0.005221,0.00859,0.01718,0.1369
BCloss-0.003668,	0.002826,	0.002508,	0.002641,	0.003221,	0.004226,	0.006652,	0.009325,	0.01821,	0.1307
Ours-0.003541,	0.002814,	0.002368,	0.002909,	0.003294,	0.004368,	0.006879,	0.01008,	0.01917,	0.1295

Amazon-Book:
LightGCN-0.0002327,	0.000297,	0.000281,	0.0003135,	0.00049077,	0.0008372,	0.001218,	0.002546,	0.004283,	0.02969
SGL-RW-0.0004182,	0.0005524,	0.0004146,	0.0005087,	0.0006458,	0.001167,	0.001661,	0.003029,	0.005255,	0.03139
SimGCL-0.0005699,	0.0006588,	0.0005101,	0.0006956,	0.0009009,	0.001415,	0.002064,	0.003616,	0.005943,	0.03038
LightGCL-0.0008609,	0.0007907,	0.0005981,	0.0007576,	0.001075,	0.001825,	0.002435,	0.003773,	0.006283,	0.03174
BCloss-0.003455,	0.001799,	0.00107,	0.001476,	0.001554,	0.002306,	0.002809,	0.003912,	0.005492,	0.03119
Ours-0.004443,	0.002085,	0.001233,	0.001903,	0.001945,	0.002966,	0.003244,	0.004411,	0.006387,	0.02871

Last-FM:
LightGCN-0.0007955,	0.0007856,	0.0009892,	0.00188,	0.002166,	0.002793,	0.004088,	0.005902,	0.0104,	0.04098
SGL-RW-0.0008463,	0.0009,	0.001057,	0.002035,	0.002393,	0.003067,	0.00434,	0.006194,	0.01082,	0.04006
SimGCL-0.0007847,	0.0007755,	0.0009562,	0.001871,	0.002339,	0.002955,	0.00426,	0.006109,	0.01079,	0.04079
LightGCL-
BCloss-0.002091,	0.001876,	0.002134,	0.002935,	0.003232,	0.003928,	0.004791,	0.006272,	0.009971,	0.03814
Ours-0.002044,	0.002017,	0.0022,	0.003239,	0.00348,	0.004232,	0.005407,	0.007318,	0.01101,	0.0372

iFashion:
LightGCN-0,	0,	0,	0,	0,	0.00001116,	0.00002604,	0.00005765,	0.0004241,	0.07899
SGL-RW-0,	0,	0,	0,	0,	0,	0,	0.00002232,	0.0002523,	0.08843
SimGCL-0,	0,	0,	0,	0,	0,	0,	0.00001116,	0.0002579,	0.09419
LightGCL-
BCloss-0,	0,	0,	0,	0,	0,	0,	0,	0.0001655,	0.1134
Ours-0,	0,	0,	0,	0,	0,	0,	0.00001116,	0.0002679,	0.1167
'''

X_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
X1 = np.array([0.0007955,	0.0007856,	0.0009892,	0.00188,	0.002166,	0.002793,	0.004088,	0.005902,	0.0104,	0.04098])
X2 = np.array([0.0006211,	0.0006633,	0.0008189,	0.00173,	0.001778,	0.002484,	0.003673,	0.005282,	0.009737,	0.03884])
X3 = np.array([0.0004855,	0.0005537,	0.0007032,	0.001668,	0.00163,	0.00244,	0.003762,	0.005663,	0.01045,	0.04103])
X4 = np.array([])
X5 = np.array([0.00196,	0.001831,	0.001883,	0.002763,	0.002951,	0.003618,	0.004869,	0.006146,	0.009821,	0.03736])
X6 = np.array([0.001907,	0.001831,	0.001915,	0.00299,	0.003031,	0.004006,	0.005376,	0.006966,	0.01101,	0.03631])

def groups(X_values, X1, X2, X3, X4, X5, X6):

    # 设置图形大小和DPI以获得更高的分辨率
    figsize = (8, 6)
    dpi = 500

    # 绘制图形
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    plt.subplots_adjust(left=0.112, bottom=0.1, right=0.95, top=0.95)

    bar_width = 0.11
    bar_positions_X1 = X_values - 2.5*bar_width
    bar_positions_X2 = X_values -1.5*bar_width
    bar_positions_X3 = X_values - 0.5*bar_width
    bar_positions_X4 = X_values + 0.5*bar_width
    bar_positions_X5 = X_values + 1.5*bar_width
    bar_positions_X6 = X_values + 2.5*bar_width
    '''
    red:#DB7B6E
    orange:#FFB265
    yellow:#FFF2CC
    green:#D1ECB9
    blue:#6DC3F2
    purple:#A68BC2
    '''
    ax.bar(bar_positions_X1, X1, width=bar_width, color='#DB7B6E', label='LightGCN', edgecolor='black', linewidth=0.5)
    ax.bar(bar_positions_X2, X2, width=bar_width, color='#FFB265', label='SGL-RW', edgecolor='black', linewidth=0.5)
    ax.bar(bar_positions_X3, X3, width=bar_width, color='#FFF2CC', label='SimGCL', edgecolor='black', linewidth=0.5)
    ax.bar(bar_positions_X4, X4, width=bar_width, color='#D1ECB9', label='LightGCL', edgecolor='black', linewidth=0.5)
    ax.bar(bar_positions_X5, X5, width=bar_width, color='#6DC3F2', label='BC loss', edgecolor='black', linewidth=0.5)
    ax.bar(bar_positions_X6, X6, width=bar_width, color='#A68BC2', label='Ours', edgecolor='black', linewidth=0.5)


    # 添加标签和标题
    plt.xlabel('Group ID', fontsize=16, fontweight='bold')
    plt.ylabel('$Recall^{(g)}@20$', fontsize=16, fontweight='bold')
    plt.title('Gowalla', fontsize=16)#TODO
    plt.xticks(X_values)
    plt.legend(loc='upper left', fontsize=16)

    # 设置坐标轴刻度字体大小和加粗
    plt.tick_params(axis='both', labelsize=12, width=2, length=6, labelcolor='black')

    # 增加横向的网格参考线
    plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

    # 显示图形
    plt.savefig('group-gowalla_new.jpg')#TODO

groups(X_values, X1, X2, X3, X4, X5, X6)









'''
2
hyperparameters  折线图
双纵轴：Overall和Long-Tail
'''
# [0., 0.01, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.5, 1.]
# [0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]
X_values = [0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]
Y_values_1 = [0.05077,	0.05477,	0.05647,	0.05737,	0.05679,	0.05525,	0.05364,	0.05205,	0.04993,	0.04815]
Y_values_2 = [0.003105,	0.003723,	0.00441,	0.004503,	0.004367,	0.003754,	0.003434,	0.003003,	0.002643,	0.002334]

def double_y(X_values, Y_values_1,Y_values_2):
    # 设置图形大小和DPI以获得更高的分辨率
    figsize = (8, 6)
    dpi = 500

    fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
    plt.subplots_adjust(left=0.2, bottom=0.05, right=0.8, top=0.93)

    # title = ('Tune $\lambda\'$ on Amazon-Book')
    title = ('Tune $\\tau$ on Amazon-Book')
    plt.title(title,fontsize=20)
    plt.grid(axis='y',color='grey',linestyle='--',lw=0.5,alpha=0.5)
    plt.tick_params(axis='both',labelsize=16)
    plot1 = ax1.plot(X_values, Y_values_1, color='#DB7B6E', label='$Recall@20$', marker='o', linestyle='-')
    ax1.set_ylabel('Overall  $Recall@20$', fontsize = 18, color='black', labelpad=10)
    # ax1.set_ylim(0,240)
    for tl in ax1.get_yticklabels():
        tl.set_color('#DB7B6E')    
    ax2 = ax1.twinx()
    plot2 = ax2.plot(X_values, Y_values_2, color='#A68BC2', label='$Recall^{(1)}@20$', marker='s', linestyle='--')
    ax2.set_ylabel('Long-tail  $Recall^{(1)}@20$',fontsize=18, color='black', labelpad=10)
    # ax2.set_ylim(0,0.08)
    ax2.tick_params(axis='y',labelsize=16)
    for tl in ax2.get_yticklabels():
        tl.set_color('#A68BC2')                    
    # ax2.set_xlim(1966,2014.15)
    lines = plot1 + plot2           
    ax1.legend(lines,[l.get_label() for l in lines], loc='lower center', fontsize=16)    
    ax1.set_yticks(np.linspace(ax1.get_ybound()[0],ax1.get_ybound()[1],9)) 
    ax2.set_yticks(np.linspace(ax2.get_ybound()[0],ax2.get_ybound()[1],9)) 
    for ax in [ax1,ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)          

    # 显示图形
    plt.savefig('tau-amazon-book.jpg')


# double_y(X_values, Y_values_1,Y_values_2)



'''
3
hyperparameters  折线图
双纵轴：Overall和Long-Tail
'''
Tau_values = [0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]
Lambda_values = [0., 0.01, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.5]
Y_values_tau = [0.07347,	0.07822,	0.08378,	0.08915,	0.09433,	0.09885,	0.1029,	0.1063,	0.1092,	0.112]
Y_values_lambda = [0.09352,	0.09441,	0.09514,	0.09467,	0.0946,	0.09409,	0.09328,	0.09271,	0.09375]
BC_tau = [0.0992, 0.0992, 0.0992, 0.0992, 0.0992, 0.0992, 0.0992, 0.0992, 0.0992, 0.0992]
