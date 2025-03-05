import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_list = []
# Load all files into a single DataFrame
for i in range(32):
    file_list.append(fr"../Output/report\heat_stress_detection_results_video_{i+1}.xlsx")  # Replace with actual file paths

# Use pd.read_excel() for Excel files
df = pd.concat([pd.read_excel(file) for file in file_list])

df['Video Number'] = df['Video ID'].str.extract(r'video \((\d+)\)')

# Loại bỏ cột 'Video ID'
df = df.drop(columns=['Video ID'])

# Lưu kết quả vào file Excel mới
df.to_excel('result.xlsx', index=False)

# # Calculate summary statistics
# summary_stats = df[['R1', 'R2', 'Average R']].describe()
# print(summary_stats)

# Histogram for R1, R2, and Average R
# Không cần plt.figure() vì hist() tự tạo figure
df[['R1', 'R2', 'R']].hist(bins=20, figsize=(10, 6))
plt.tight_layout()
plt.show()


# Scatter plot for R1 vs R2
plt.figure(figsize=(8, 6))
sns.scatterplot(x='R1', y='R2', data=df)
plt.title('R1 vs R2')
plt.show()