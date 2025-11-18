# import pickle
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# # Load preprocessed landmark data
# data_dict = pickle.load(open('./data.pickle', 'rb'))
# data = np.asarray(data_dict['data'])
# labels = np.asarray(data_dict['labels'])

# # Split into train/test
# x_train, x_test, y_train, y_test = train_test_split(
#     data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
# )

# # Train RandomForest
# model = RandomForestClassifier(n_estimators=300, max_depth=25, random_state=42)
# model.fit(x_train, y_train)

# # Evaluate
# y_pred_train = model.predict(x_train)
# y_pred_test = model.predict(x_test)

# train_acc = accuracy_score(y_train, y_pred_train)
# test_acc = accuracy_score(y_test, y_pred_test)

# print(f"🎯 Training Accuracy: {train_acc*100:.2f}%")
# print(f"🎯 Testing Accuracy: {test_acc*100:.2f}%\n")

# print("📄 Classification Report:")
# print(classification_report(y_test, y_pred_test))

# # Save model + accuracy
# with open('model.p', 'wb') as f:
#     pickle.dump({'model': model, 'accuracy': test_acc}, f)

# print("💾 Model saved as 'model.p' with accuracy included")

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load preprocessed landmark data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

# Train RandomForest
model = RandomForestClassifier(n_estimators=300, max_depth=25, random_state=42)
model.fit(x_train, y_train)

# Evaluate
train_acc = accuracy_score(y_train, model.predict(x_train))
test_acc = accuracy_score(y_test, model.predict(x_test))

print(f"🎯 Training Accuracy: {train_acc*100:.2f}%")
print(f"🎯 Testing Accuracy: {test_acc*100:.2f}%\n")
print("📄 Classification Report:")
print(classification_report(y_test, model.predict(x_test)))

# Save model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model, 'accuracy': test_acc}, f)

print("💾 Model saved as 'model.p'")
