import matplotlib.pyplot as plt
import seaborn as sns

# Visualize feature importance
importances = stacked_model.named_estimators_['rf'].feature_importances_
sns.barplot(x=importances, y=data.columns[:-1])  # Adjust column indexing
plt.title("Feature Importances")
plt.show()
