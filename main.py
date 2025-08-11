import pandas as pd

data1 = [
    {'student_id': 1, 'subject': 'Math', 'exam_date': '2025-06-25', 'difficulty': 3, 'past_score': 75, 'syllabus_covered': 40, 'preferred_study_hours': 3, 'subject_priority': 3},
    {'student_id': 2, 'subject': 'Science', 'exam_date': '2025-07-05', 'difficulty': 2, 'past_score': 60, 'syllabus_covered': 50, 'preferred_study_hours': 2, 'subject_priority': 2},
    {'student_id': 3, 'subject': 'History', 'exam_date': '2025-06-20', 'difficulty': 1, 'past_score': 80, 'syllabus_covered': 60, 'preferred_study_hours': 1, 'subject_priority': 1},
    {'student_id': 4, 'subject': 'English', 'exam_date': '2025-07-10', 'difficulty': 1, 'past_score': 85, 'syllabus_covered': 70, 'preferred_study_hours': 2, 'subject_priority': 2},
    {'student_id': 5, 'subject': 'Computer', 'exam_date': '2025-06-28', 'difficulty': 2, 'past_score': 55, 'syllabus_covered': 35, 'preferred_study_hours': 3, 'subject_priority': 3},
    {'student_id': 6, 'subject': 'Physics', 'exam_date': '2025-07-01', 'difficulty': 2, 'past_score': 65, 'syllabus_covered': 55, 'preferred_study_hours': 3, 'subject_priority': 3},
    {'student_id': 7, 'subject': 'Chemistry', 'exam_date': '2025-06-30', 'difficulty': 3, 'past_score': 50, 'syllabus_covered': 30, 'preferred_study_hours': 4, 'subject_priority': 3},
    {'student_id': 8, 'subject': 'Biology', 'exam_date': '2025-07-03', 'difficulty': 1, 'past_score': 90, 'syllabus_covered': 80, 'preferred_study_hours': 2, 'subject_priority': 1},
    {'student_id': 9, 'subject': 'Geography', 'exam_date': '2025-07-06', 'difficulty': 2, 'past_score': 70, 'syllabus_covered': 60, 'preferred_study_hours': 2, 'subject_priority': 2},
    {'student_id': 10, 'subject': 'Economics', 'exam_date': '2025-06-29', 'difficulty': 3, 'past_score': 45, 'syllabus_covered': 40, 'preferred_study_hours': 3, 'subject_priority': 3}
]

df1 = pd.DataFrame(data1)
# Save to CSV
df1.to_csv("final_study_data.csv", index=False)
print("✅ Dataset saved as 'final_study_data.csv'")
print(df1.columns)
# Read back the saved dataset
loaded_df = pd.read_csv('final_study_data.csv')
print(loaded_df.head(10))
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

# 1️⃣ Load dataset
data = pd.read_csv("final_study_data.csv")

# 2️⃣ Convert date column
data['exam_date'] = pd.to_datetime(data['exam_date'])
data['exam_day'] = data['exam_date'].dt.day
data['exam_month'] = data['exam_date'].dt.month
data['exam_dayofweek'] = data['exam_date'].dt.dayofweek
data = data.drop(columns=['exam_date'])

# 3️⃣ OneHot encode categorical columns
data_encoded = pd.get_dummies(data, columns=['subject'], drop_first=True)

# 4️⃣ Features and target
X = data_encoded.drop(columns=['preferred_study_hours'])
y = data_encoded['preferred_study_hours']

# 5️⃣ Create Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# 🔵 6️⃣ ⚡️ ADD MAE CROSS-VALIDATION CODE HERE
cv_scores_mae = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=3)
# Scores are negative, so take absolute values
print("Cross-validated MAE Scores:", -cv_scores_mae)
print("Average MAE:", -cv_scores_mae.mean())

# 7️⃣ Train-test split for final testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 8️⃣ Train the model
model.fit(X_train, y_train)

# 9️⃣ Predict on test set
y_pred = model.predict(X_test)

# 🔴 1️⃣0️⃣ Final evaluation on test set
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error on final test set:", mae)
import pandas as pd
from datetime import datetime

# Create lists to hold data
user_data_list = []
subject_names = []
"""
today = pd.to_datetime(datetime.now().date())  # today's date

while True:
    subject = input("Enter subject name: ")
    exam_date_str = input("Enter exam date (YYYY-MM-DD): ")
    exam_date = pd.to_datetime(exam_date_str)

    days_left = (exam_date - today).days  # calculate days left

    difficulty = int(input("Difficulty (1-3) (1-Easy 2-Medium 3-Hard): "))
    past_score = int(input("Past score (%): "))
    syllabus_covered = int(input("Syllabus covered (%): "))
    subject_priority = int(input("Subject priority (1-3) (1-Low 2-Medium 3-High): "))

    user_data_list.append({
        'subject': subject,
        'days_left': days_left,    # calculated here!
        'difficulty': difficulty,
        'past_score': past_score,
        'syllabus_covered': syllabus_covered,
        'subject_priority': subject_priority,
        'exam_date': exam_date     # still needed to extract features
    })
    subject_names.append(subject)  # Save subject name separately

    cont = input("Do you want to enter another subject? (yes/no): ")
    if cont.lower() != 'yes':
        break

# Create DataFrame
user_df = pd.DataFrame(user_data_list)

user_df = pd.DataFrame(user_data_list)
print("\nHere is your data:")
print(user_df)
user_df_original=user_df.copy()
"""
# Use the sample data for now
today = pd.to_datetime(datetime.now().date())
data = [
    {
        'subject': 'Maths',
        'exam_date': pd.to_datetime('2025-07-01'),
        'difficulty': 2,
        'past_score': 75,
        'syllabus_covered': 60,
        'subject_priority': 3
    },
    {
        'subject': 'Physics',
        'exam_date': pd.to_datetime('2025-07-05'),
        'difficulty': 3,
        'past_score': 65,
        'syllabus_covered': 50,
        'subject_priority': 2
    },
    {
        'subject': 'Chemistry',
        'exam_date': pd.to_datetime('2025-07-10'),
        'difficulty': 1,
        'past_score': 85,
        'syllabus_covered': 70,
        'subject_priority': 1
    },
    {
        'subject': 'Biology',
        'exam_date': pd.to_datetime('2025-07-15'),
        'difficulty': 2,
        'past_score': 80,
        'syllabus_covered': 65,
        'subject_priority': 2
    },
    {
        'subject': 'English',
        'exam_date': pd.to_datetime('2025-07-20'),
        'difficulty': 1,
        'past_score': 90,
        'syllabus_covered': 80,
        'subject_priority': 1
    }
]

for entry in data:
    entry['days_left'] = (entry['exam_date'] - today).days

user_df = pd.DataFrame(data)
user_df_original = user_df.copy()

# ✅ Create subject_names list from the dataset
subject_names = [entry['subject'] for entry in data]

print("\n✅ Example dataset created:")
print(user_df)
# Convert exam_date to date features (commented out, so skipping)
#print(user_df_original)
# Drop original date column
user_df = user_df.drop(columns=['exam_date'], errors='ignore')

# Align columns with training data directly using reindex
user_df = user_df.reindex(columns=X.columns, fill_value=0)

# Predict study hours and boost by 3.0
predicted_hours = model.predict(user_df)
predicted_hours *= 8

# Print predictions
print("\n✅ Predicted study hours for each subject:")
for subject, hours in zip(subject_names, predicted_hours):
    print(f"{subject}: {hours:.2f} hours")

# Save predictions
user_df['predicted_study_hours'] = predicted_hours
user_df['subject_name'] = subject_names
user_df.to_csv("user_study_hours_predictions.csv", index=False)
print("\n✅ Predictions saved to 'user_study_hours_predictions.csv'")
# Recommended daily study hours
print("\n✅ Recommended daily study hours (with minutes):")
for subject, hours, days_left in zip(subject_names, predicted_hours, user_df_original['days_left']):
    if days_left > 0:
        daily_hours = hours / days_left
        daily_minutes = daily_hours * 60  # convert hours to minutes
        print(f"📌 {subject}: {daily_hours:.2f} hours/day (~{daily_minutes:.0f} minutes/day) for the next {days_left} days")
    else:
        print(f"⚠️ {subject}: Exam day has already passed or is today.")
print("\n✅ Study advice / warnings:")
for subject, hours, days_left in zip(subject_names, predicted_hours, user_df_original['days_left']):
    if days_left > 0:
        daily_hours = hours / days_left
        if daily_hours > 1:
            print(f"⚠️ {subject}: Daily study hours ({daily_hours:.2f}) are quite high! Consider revising your study plan.")
        else:
            print(f"✅ {subject}: Daily study hours ({daily_hours:.2f}) seem manageable.")
    else:
        print(f"⚠️ {subject}: Exam day has already passed or is today. Focus on revision and confidence boosting.")
print("\n✅ Priority-based study tips:")
for subject, priority in zip(subject_names, user_df_original['subject_priority']):
    if priority == 3:
        print(f"🔥 {subject}: High priority – make sure to cover all topics and revise well.")
    elif priority == 2:
        print(f"⚡️ {subject}: Medium priority – focus on important areas and practice questions.")
    elif priority == 1:
        print(f"🌱 {subject}: Low priority – revise basics and focus on weaker areas.")
import matplotlib.pyplot as plt

# Assuming you have:
# - `subject_names`: list of subject names
# - `predicted_hours`: array or list of predicted total hours

# Plotting
plt.figure(figsize=(8, 5))
bars = plt.bar(subject_names, predicted_hours, color='skyblue', edgecolor='black')
plt.xlabel('Subjects')
plt.ylabel('Predicted Study Hours')
plt.title('Predicted Total Study Hours per Subject')

# Add text on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f'{yval:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()
#plt.savefig('predicted_study_hours_graph.png')
#print("✅ Graph saved as 'predicted_study_hours_graph.png'")
import matplotlib.pyplot as plt

# Calculate daily study hours and minutes for each subject
daily_study_hours = []
daily_study_minutes = []

for subject, hours, days_left in zip(subject_names, predicted_hours, user_df_original['days_left']):
    if days_left > 0:
        daily_hours = hours / days_left
        daily_study_hours.append(daily_hours)
        daily_study_minutes.append(daily_hours * 60)
    else:
        daily_study_hours.append(0)
        daily_study_minutes.append(0)

# Plotting
plt.figure(figsize=(10, 6))
bars = plt.bar(subject_names, daily_study_hours, color='lightgreen', edgecolor='black')

# Label bars with both hours and minutes
for i, bar in enumerate(bars):
    hours = daily_study_hours[i]
    minutes = daily_study_minutes[i]
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
             f'{hours:.2f} hrs\n({minutes:.0f} mins)',
             ha='center', va='bottom')

plt.xlabel('Subjects')
plt.ylabel('Daily Study Hours')
plt.title('Recommended Daily Study Hours per Subject')
plt.ylim(0, max(daily_study_hours) * 1.2)  # Add some space above the bars

plt.tight_layout()
plt.show()
#plt.savefig('recommended_study_hours.png', dpi=300)  # dpi=300 for high-resolution
#print("\n✅ The bar chart has been saved as 'recommended_study_hours.png'!")
import matplotlib.pyplot as plt
import math

# Example data
# user_df_original = pd.DataFrame({
#     'subject': ['Math', 'Science', 'English', 'History', 'Geography', 'Art', 'PE'],
#     'syllabus_covered': [75, 50, 90, 60, 85, 30, 55]
# })

# Number of pies (donut charts)
num_pies = len(user_df_original)
pies_per_row = 3
num_rows = math.ceil(num_pies / pies_per_row)

fig, axs = plt.subplots(num_rows, pies_per_row, figsize=(4 * pies_per_row, 4 * num_rows))

# Flatten axs for easier iteration
axs = axs.flatten()

for ax, (index, row) in zip(axs, user_df_original.iterrows()):
    sizes = [row['syllabus_covered'], 100 - row['syllabus_covered']]
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=['Covered', 'Remaining'],
        autopct='%1.1f%%',
        startangle=90,
        colors=['green', 'lightgrey'],
        wedgeprops={'width': 0.3, 'edgecolor': 'white'}
    )

    # Draw the center circle to create the donut hole
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax.add_artist(centre_circle)

    ax.set_title(f'{row["subject"]} Syllabus Coverage')

    # Draw a border around each donut chart
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

# Remove unused axes if total pies < total subplots
for i in range(num_pies, len(axs)):
    fig.delaxes(axs[i])

# Adjust space between the donut charts
plt.subplots_adjust(wspace=0.5, hspace=0.5)

plt.show()
