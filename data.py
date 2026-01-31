import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Replace with your dataset path
df = pd.read_csv("sports_data.csv")

print(df.head())
print(df.info())

# Remove duplicates
df = df.drop_duplicates()

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

print(df.isnull().sum())

print(df.describe())

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Top 10 players by score
top_players = df.sort_values(by="score", ascending=False).head(10)

plt.figure(figsize=(10,5))
sns.barplot(x="player_name", y="score", data=top_players)
plt.xticks(rotation=45)
plt.title("Top 10 Player Performances")
plt.show()

team_avg = df.groupby("team")["score"].mean().sort_values(ascending=False)

team_avg.plot(kind="bar", figsize=(10,5), title="Average Team Score")
plt.show()


X = df[["matches_played", "minutes", "assists"]]  # example features
y = df["score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
