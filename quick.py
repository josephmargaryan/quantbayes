import pandas as pd

# Create the DataFrame
data = [
    ["Sunny", "Hot", "High", "Weak", "No"],
    ["Sunny", "Hot", "High", "Strong", "No"],
    ["Overcast", "Hot", "High", "Weak", "Yes"],
    ["Rainy", "Mild", "High", "Weak", "Yes"],
    ["Rainy", "Cool", "Normal", "Weak", "Yes"],
    ["Rainy", "Cool", "Normal", "Strong", "No"],
    ["Overcast", "Cool", "Normal", "Strong", "Yes"],
    ["Sunny", "Mild", "High", "Weak", "No"],
    ["Sunny", "Cool", "Normal", "Weak", "Yes"],
    ["Rainy", "Mild", "Normal", "Weak", "Yes"],
    ["Sunny", "Mild", "Normal", "Strong", "Yes"],
    ["Overcast", "Mild", "High", "Strong", "Yes"],
    ["Overcast", "Hot", "Normal", "Weak", "Yes"],
    ["Rainy", "Mild", "High", "Strong", "No"],
]

columns = ["Outlook", "Temperature", "Humidity", "Wind", "PlayTennis"]
df = pd.DataFrame(data, columns=columns)

print(df)

# Save as CSV (no index column)
df.to_csv("play_tennis.csv", index=False)
print("Saved to play_tennis.csv")
