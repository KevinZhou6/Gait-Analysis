import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.font_manager as font_manager
# 创建一个字体属性对象，用于图例
font_prop = font_manager.FontProperties(weight='bold',size=15)



# 数据
X = ['in Real World', 'in VR', 'Same-age', 'Old-age', 'Before Embodying', 'After Embodying']
y = [1.241, 1.089, 1.061, 1.067, 1.209, 1.171]
positions = range(len(X))  # 柱状图的位置
width = 0.4  # 柱宽

# 自定义颜色
custom_colors = [(0.3, 0.1, 0.4, 1), (0.3, 0.5, 0.4, 1), (0.3, 0.9, 0.4, 1)]

# 绘制柱状图
plt.figure(figsize=(9, 10))  # 调整图形的尺寸，减少重叠

for i, pos in enumerate(positions):
    plt.bar(pos, y[i], color=custom_colors[i//2], width=width, label=f'Part {i//2 + 1}' if i % 2 == 0 else "")
    # Display values without bold
    plt.text(pos, y[i]+0.005, f'{y[i]:.3f}', ha='center', va='bottom',fontweight='bold')  # Add a small offset (0.005) for better visibility

# Set x-axis and y-axis labels to bold
plt.xticks(positions, X, rotation=45, ha='right', fontsize=14, fontweight='bold')  # Bold font for x-axis labels
plt.ylabel('Stride Length  (in m)', fontsize=14, fontweight='bold')  # Bold font for y-axis label

# Manually set axis tick labels to bold
for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
    label.set_fontweight('bold')

# Adjust legend position without making text bold
plt.legend(loc='upper left', bbox_to_anchor=(1, 1),prop=font_prop)

# Adjust layout to make space for legend to ensure it does not overlay the chart
plt.tight_layout(rect=[0, 0, 0.85, 1])

# Save the figure with high resolution
plt.savefig('Stride Length.png', dpi=200)

plt.show()
