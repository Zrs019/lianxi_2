import matplotlib.pyplot as plt

# 设置中文字体，防止乱码 (根据您的系统可能需要调整，如 'SimHei', 'Microsoft YaHei')
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# 电价与颜色设置 (参考原图风格)
colors = {'低谷': '#3E9697', '高峰': '#BA55D3', '尖峰': '#FF0000'}
y_vals = {'低谷': 0.2262, '高峰': 0.9821, '尖峰': 1.1786}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))

def plot_schedule(ax, schedule, title):
    # 设置X轴 (0-24小时)
    ax.set_xticks(range(25))
    ax.set_xlim(0, 24)
    # 设置Y轴 (根据电价)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0.2262, 0.9821, 1.1786])
    ax.set_yticklabels(['低谷\n0.2262', '高峰\n0.9821', '尖峰\n1.1786'])
    
    # 开启网格
    ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='#E0E0E0')
    ax.set_xlabel(title, fontsize=12)
    
    # 绘制阶梯图的线段
    for i, item in enumerate(schedule):
        start, end, phase = item
        y = y_vals[phase]
        
        # 画水平横线
        ax.plot([start, end], [y, y], color=colors[phase], linewidth=3)
        
        # 画垂直连接线
        if i > 0:
            prev_y = y_vals[schedule[i-1][2]]
            ax.plot([start, start], [prev_y, y], color='gray', linewidth=1.5)
            
        # 添加文字标签框
        mid_x = (start + end) / 2
        ax.text(mid_x, y - 0.1, phase, ha='center', va='top', 
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9, boxstyle='square,pad=0.3'))

# 常规月份数据 (2-6月、9-11月)
schedule_normal = [
    (0, 8, '低谷'), (8, 9, '高峰'), (9, 11, '尖峰'), 
    (11, 13, '低谷'), (13, 15, '高峰'), (15, 17, '尖峰'), 
    (17, 22, '高峰'), (22, 24, '低谷')
]

# 特殊月份数据 (1、7、8、12月) -> 13:00-15:00 转为尖峰，与 15:00-17:00 连成一段
schedule_special = [
    (0, 8, '低谷'), (8, 9, '高峰'), (9, 11, '尖峰'), 
    (11, 13, '低谷'), (13, 17, '尖峰'), 
    (17, 22, '高峰'), (22, 24, '低谷')
]

# 绘图
plot_schedule(ax1, schedule_normal, '常规月份（2-6月、9-11月）分时电价时段')
plot_schedule(ax2, schedule_special, '冬夏用电高峰月（1月、7月、8月和12月）分时电价时段')

plt.tight_layout()
plt.show()