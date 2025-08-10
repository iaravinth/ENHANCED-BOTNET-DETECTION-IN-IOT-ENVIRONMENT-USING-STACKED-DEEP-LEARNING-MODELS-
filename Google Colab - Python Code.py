import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('data_1.csv')
print(df)

# Preprocessing
dataframes = []
samples_per_file = 1999
normal_samples = df[df['category'] == 'Normal']
non_normal_samples = df[df['category'] != 'Normal']
random_samples = non_normal_samples.sample(n=samples_per_file, random_state=42)
merged_df = pd.concat([normal_samples, random_samples])
dataframes.append(merged_df)
botiot = pd.concat(dataframes, ignore_index=True)

# Adjust category values
botiot.loc[botiot['category'] != 'Normal', 'category'] = 'Attack'
botiot.loc[botiot['category'] == 'Normal', 'category'] = '0_Normal'

# Drop irrelevant columns
botiot = botiot.drop(columns=['pkSeqID', 'stime', 'ltime', 'seq', 'smac', 'dmac', 'soui', 'doui', 'sco', 'dco', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'srate', 'drate', 'attack'])
botiot = botiot.dropna(subset=['sport'])
botiot.drop_duplicates(inplace=True)
if 'state' in botiot.columns:
    botiot = botiot.drop(columns=['saddr', 'daddr', 'state', 'sport', 'dport'])
else:
    botiot = botiot.drop(columns=['saddr', 'daddr', 'sport', 'dport'])

# Data Transformation using Label Encoding

label_encoder = LabelEncoder()

label_encoder_proto = LabelEncoder()
label_encoder_flgs = LabelEncoder()

botiot['proto'] = label_encoder_proto.fit_transform(botiot['proto'])
botiot['flgs'] = label_encoder_flgs.fit_transform(botiot['flgs'])

# Save encoders separately
joblib.dump(label_encoder_proto, 'label_encoder_proto.pkl')
joblib.dump(label_encoder_flgs, 'label_encoder_flgs.pkl')

# Check the mapping
proto_mapping = dict(zip(label_encoder_proto.classes_, label_encoder_proto.transform(label_encoder_proto.classes_)))
flgs_mapping = dict(zip(label_encoder_flgs.classes_, label_encoder_flgs.transform(label_encoder_flgs.classes_)))

print("Proto Encoding:", proto_mapping)
print("Flgs Encoding:", flgs_mapping)

# Separate features and target variable
botiot = botiot.drop(columns=['subcategory '])
X = botiot.drop(columns=['category'])
y = botiot['category']


# Encode target variable
y = y.map({'0_Normal': 0, 'Attack': 1}).astype(int)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, 'scaler.pkl')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshape for BiLSTM
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Hybrid DNN-BiLSTM Model
dnn_input = layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
dnn = layers.Dense(512, activation='relu')(dnn_input)
dnn = layers.Dropout(0.3)(dnn)
dnn = layers.Dense(256, activation='relu')(dnn)
dnn = layers.Dropout(0.3)(dnn)
dnn = layers.Dense(128, activation='relu')(dnn)
dnn = layers.Dropout(0.3)(dnn)
dnn = layers.Dense(64, activation='relu')(dnn)
dnn = layers.Dropout(0.3)(dnn)
dnn = layers.Dense(32, activation='relu')(dnn)

bilstm_input = layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
bilstm = layers.Bidirectional(layers.LSTM(50, return_sequences=False))(bilstm_input)
bilstm = layers.Dense(32, activation='relu')(bilstm)
bilstm = layers.Dense(32, activation='relu')(bilstm)
dnn = layers.Reshape((32,))(dnn)
combined = layers.concatenate([dnn, bilstm])
output = layers.Dense(1, activation='sigmoid')(combined)

model = models.Model(inputs=[dnn_input, bilstm_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit([X_train, X_train], y_train, epochs=25, batch_size=256)

# Save model
model.save('botnet_model.h5')

# Model Evaluation
y_pred = model.predict([X_test, X_test])
y_pred_binary = (y_pred > 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_binary)
plt.figure(figsize=(7, 4))
lang = ['Normal', 'Attack']
cm = pd.DataFrame(cm, columns=lang, index=lang)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix - Test Set')
plt.xlabel('Predicted')
plt.ylabel('Actual')


# Accuracy and Classification Report
print(f"Accuracy: {accuracy_score(y_test, y_pred_binary) * 100:.2f}%")
print(classification_report(y_test, y_pred_binary, digits=4))

plt.show()
