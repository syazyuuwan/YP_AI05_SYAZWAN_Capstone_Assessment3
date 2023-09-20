# %%
# import packages
import os, datetime
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,roc_auc_score
# %%
# define functions

# function for filling null values
def fill_na(df_fill):
    for column in df_fill.columns:
        if df_fill[column].dtypes == 'object':
            df_fill[column] = df_fill[column].fillna('unknown')
        else:
            df_fill[column] = df_fill[column].fillna(df_fill[column].mean())
    return df_fill
            
# function for one hot encoding
def one_hot_encode(df):
    df = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns)
    return df

# function to split features and labels
def feature_label_split(df):
    features = one_hot_encode(df.drop(columns='term_deposit_subscribed',axis=1))
    labels = df['term_deposit_subscribed']
    
    return features, labels

# function for splitting data into train, validation, and test set
def train_val_test_split(X,y,random_state):
    X_train, X_temp, y_train, y_temp = train_test_split(X,
                                                        y,
                                                        train_size=0.7,
                                                        random_state=random_state)

    X_val, X_test, y_val, y_test = train_test_split(X_temp,
                                                    y_temp,
                                                    train_size=0.7,
                                                    random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test

# function to compile and fit model, can also specify tensorboard log directory
def compile_and_fit(model, X_train, X_val, y_train, y_val, patience=3, epochs=10, tb_logdir='logs'):
    
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    
    early_stopping = keras.callbacks.EarlyStopping(patience=patience)
    
    base_log_path=f"tensorboard_logs\{tb_logdir}"
    log_path = os.path.join(base_log_path,datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tb = tf.keras.callbacks.TensorBoard(log_path)

    model.fit(X_train, y_train,
            validation_data=(X_val,y_val),
            epochs=epochs,
            callbacks=[early_stopping,tb])
    
# function to print metrics
def predict_and_metric(X_test,y_true):
    y_pred = model.predict(X_test).astype('int64')
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true,y_pred)
    recall = recall_score(y_true,y_pred)
    f1 = f1_score(y_true,y_pred)
    roc_auc = roc_auc_score(y_true,y_pred)
    confuse_matrix = confusion_matrix(y_true,y_pred)

    metrics = f"""
Accuracy Score:     {accuracy}
Precision Score:    {precision}
Recall Score:       {recall}
F1 Score:           {f1}
ROC AUC:            {roc_auc}
Confusion Matrix:
{confuse_matrix}
    """
    return metrics
    
# function for oversampling and undersampling
def over_under_sample(features, labels, resample_type):
    if resample_type == 'under':
        if (len(features[labels==0]) > len(features[labels==1])):
            features_major = features[labels==0]
            features_minor = features[labels==1]
            minor_label = 1
        else:
            features_major = features[labels==1]
            features_minor = features[labels==0]
            minor_label = 0

        from sklearn.utils import resample

        features_major_undersample = resample(features_major,
                                            replace=False,
                                            n_samples=len(features_minor),
                                            random_state=42)

        features_resampled = np.vstack([features_major_undersample, features_minor])
        if minor_label == 1:
            labels_resampled = np.concatenate([np.zeros(len(features_minor)), np.ones(len(features_minor))])
        else:
            labels_resampled = np.concatenate([np.ones(len(features_minor)), np.zeros(len(features_minor))])
    
    else:
        if (len(features[labels==0]) > len(features[labels==1])):
            features_major = features[labels==0]
            features_minor = features[labels==1]
            minor_label = 1
        else:
            features_major = features[labels==1]
            features_minor = features[labels==0]
            minor_label = 0

        from sklearn.utils import resample

        features_minor_oversample = resample(features_minor,
                                            replace=True,
                                            n_samples=len(features_major),
                                            random_state=42)

        features_resampled = np.vstack([features_major, features_minor_oversample])
        if minor_label == 1:
            labels_resampled = np.concatenate([np.zeros(len(features_major)), np.ones(len(features_major))])
        else:
            labels_resampled = np.concatenate([np.ones(len(features_major)), np.zeros(len(features_major))])
    
    return features_resampled, labels_resampled
# %%
# load data
FILEPATH = os.path.join(os.getcwd(),'train.csv')
df = pd.read_csv(FILEPATH)
# %%
df.head()
# %%
df.info()
# %%
df.isnull().sum()
# %%
df.describe().T
# %%
df['term_deposit_subscribed'].value_counts()
# %%
df_fill = fill_na(df.copy())
# %%
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
num_plots = len(numerical_columns)
num_rows = (num_plots - 1) // 2 + 1 
num_cols = 2  

fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 20))
fig.suptitle('Distribution of Numerical Columns', fontsize=16)
axes = axes.flatten()

for i, column in enumerate(numerical_columns):
    sns.histplot(df[column], kde=True, ax=axes[i])
    axes[i].set_title(f"Distribution of {column}")
    axes[i].set_xlabel(column)
    axes[i].set_ylabel("Frequency")    

if num_plots < num_rows * num_cols:
    for i in range(num_plots, num_rows * num_cols):
        fig.delaxes(axes[i])

plt.tight_layout()
plt.title("Distribution of numerical columns")
plt.show()
# %%
fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 20))
fig.suptitle('Distribution of Numerical Columns After Impute with Mean', fontsize=16)
axes = axes.flatten()

for i, column in enumerate(numerical_columns):
    sns.histplot(df_fill[column], kde=True, ax=axes[i])
    axes[i].set_title(f"Distribution of {column}")
    axes[i].set_xlabel(column)
    axes[i].set_ylabel("Frequency")

if num_plots < num_rows * num_cols:
    for i in range(num_plots, num_rows * num_cols):
        fig.delaxes(axes[i])

plt.tight_layout()

plt.show()
# %%
df_fill.drop(columns='id',axis=1,inplace=True)
# %%
# separate features and labels
features, labels = feature_label_split(df_fill)
# %%
RANDOM_STATE = 13
# %%
# build model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape = (features.shape[1],)),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1,activation='sigmoid')
])
# %%
# split data into train, validation, and test set
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(features.values, 
                                                                      labels.values,
                                                                      random_state=RANDOM_STATE)

# compile and fit model
compile_and_fit(model,X_train, X_val, y_train, y_val, patience=3,epochs=50)

# predict and show metric
base_result = predict_and_metric(X_test,y_test)
# %% ----------------------------------------------------
# undersampling
features_undersampled, labels_undersampled = over_under_sample(features,labels,'under')

# split data into train, validation, and test set
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(features_undersampled, 
                                                                      labels_undersampled,
                                                                      random_state=RANDOM_STATE)

# compile and fit model
compile_and_fit(model,X_train, X_val, y_train, y_val, patience=3,epochs=50, tb_logdir='undersample')

# predict and show metric
undersample_result = predict_and_metric(X_test,y_test)
# %% ----------------------------------------------------
# oversampling
features_oversampled, labels_oversampled = over_under_sample(features,labels,'over')

# split data into train, validation, and test set
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(features_oversampled, 
                                                                      labels_oversampled,
                                                                      random_state=RANDOM_STATE)

# compile and fit model
compile_and_fit(model,X_train, X_val, y_train, y_val, patience=3,epochs=50, tb_logdir='oversample')

# predict and show metric
oversample_result = predict_and_metric(X_test,y_test)
# %% ----------------------------------------------------

print(f"""
Base model results:
{base_result}

-------------------------------
""")
# %%
print(f"""
Undersampling result:
{undersample_result}

-------------------------------
""")
# %%
print(f"""
Oversampling result:
{oversample_result}
      """)

# %%
keras.utils.plot_model(model)
# %%
model.save(os.path.join('models', 'Capstone3_Syazwan.h5'))