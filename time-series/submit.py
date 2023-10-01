# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping


# %%
dataset = pd.read_csv('./jena_climate_2009_2016.csv')
total_null = dataset.isnull().sum().sort_values(ascending=False)
dataset = dataset[:20000]
newset = dataset[['Date Time','T (degC)']]
print("")
print("Total Record:")
print(dataset.shape[0])
print("\n")
print("is nan : \n",newset.isna().sum())
print("\n")
print("is null : \n",newset.isnull().sum())
newset

# %%
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    # series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[-1:]))
    return ds.batch(batch_size).prefetch(1)

# %%
scaler = MinMaxScaler()

dates = newset['Date Time'].values
temp  = newset['T (degC)'].values

# %%
early_stopping = EarlyStopping(
    monitor='mae',
    patience=10,
    mode='min',
    min_delta=0.1,
    verbose=1
)

# %%
temp_train, temp_test, dates_train, dates_test = train_test_split(
    temp, dates, test_size=0.2, shuffle=False)

# Apply Min-Max Scaling to 'T (degC)'
scaler = MinMaxScaler()
temp_scaled = scaler.fit_transform(temp_train.reshape(-1, 1))

train_set = windowed_dataset(
    temp_scaled, window_size=60, batch_size=100, shuffle_buffer=1000)
model = tf.keras.models.Sequential([
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64),
        merge_mode='concat',
    ),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(1),
])

optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.2, momentum=0.5)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(train_set, epochs=100,callbacks=[early_stopping])


