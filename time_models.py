import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers


def get_nn_time_model(df, epochs=5, training_split=0.8):
    xs = df[df.columns[:-1]]
    ys = df['move_time']
    train_xs = xs[:int(len(xs) * training_split)]
    train_ys = ys[:int(len(ys) * training_split)]
    test_xs = xs[int(len(xs) * training_split):]
    test_ys = ys[int(len(ys) * training_split):]

    nn_time_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)])

    adam = tf.keras.optimizers.Adam(learning_rate=0.0001)

    nn_time_model.compile(optimizer=adam, loss='mean_squared_error')

    nn_time_model.fit(train_xs,
                      train_ys,
                      validation_data=(test_xs, test_ys),
                      epochs=epochs)

    return nn_time_model

def get_bayesian_nn_time_model(df, epochs=5, training_split=0.8):
    xs = df[df.columns[:-1]]
    ys = df['move_time']
    train_xs = xs[:int(len(xs) * training_split)]
    train_ys = ys[:int(len(ys) * training_split)]
    test_xs = xs[int(len(xs) * training_split):]
    test_ys = ys[int(len(ys) * training_split):]

    input_length = len(df.columns[:-1])

    bayesian_nn_time_model = tfk.Sequential([
        tfkl.Input(shape=(input_length,)), 
        tfkl.Dense(64, activation='relu'),
        tfkl.Dense(64, activation='relu'),
        tfkl.Dense(64, activation='relu'),
        tfkl.Dense(1),
        tfpl.DistributionLambda(
            lambda t: tfd.LogNormal(loc=t, scale=1),
        )])

    adam = tf.keras.optimizers.Adam(learning_rate=0.0001)

    bayesian_nn_time_model.compile(optimizer=adam,
                                   loss=lambda y, rv_y: -rv_y.log_prob(y))
    
    bayesian_nn_time_model.fit(train_xs,
                               train_ys,
                               validation_data=(test_xs, test_ys),
                               epochs=epochs)
    
    return bayesian_nn_time_model


def plot_actual_vs_predicted(model, df):
    df_copy = df.copy()
    xs = df[df.columns[:-1]]
    ys = df['move_time']
    df_predicted = model.predict(xs)
    df_copy["predicted_time"] = df_predicted
    df_copy = df_copy.loc[(df_copy['move_time'] < 45) & (df_copy['predicted_time'] < 45)]
    num_columns = len(df_copy.columns) - 2

    # Calculate the number of rows and columns for the subplots
    subplot_rows = int(np.ceil(np.sqrt(num_columns)))
    subplot_cols = int(np.ceil(num_columns / subplot_rows))

    # Create a figure and array of subplots
    fig, axs = plt.subplots(subplot_rows, subplot_cols, figsize=(8, 8))
    axs = axs.ravel()

    # Iterate through each column (excluding the last two)
    for idx, column in enumerate(df_copy.columns[:-2]):
        # Plot the column against the last two columns as scatterplots
        axs[idx].scatter(df_copy[column], df_copy['move_time'], color='blue', alpha=0.003)
        axs[idx].scatter(df_copy[column], df_copy['predicted_time'], color='red', alpha=0.003)

        # Set the title, labels and legend
        axs[idx].set_title(f"{' '.join(column.split('_'))} vs. actual/predicted times")
        axs[idx].set_xlabel(' '.join(column.split('_')))
        axs[idx].set_ylabel("Time")

        # Create custom legend handles with alpha value set to 1
        blue_handle = mlines.Line2D([], [], color='blue', marker='o', linestyle='', alpha=1, label='move_time')
        red_handle = mlines.Line2D([], [], color='red', marker='o', linestyle='', alpha=1, label='predicted_time')
        
        # Add the custom legend handles to the subplot
        axs[idx].legend(handles=[blue_handle, red_handle])


    # Remove the unused subplots
    for idx in range(num_columns, len(axs)):
        fig.delaxes(axs[idx])

    # Adjust the layout and show the plot
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    csv_path = 'training_data.csv'
    df = pd.read_csv(csv_path)
    means = df[df.columns[:-1]].mean()
    stds = df[df.columns[:-1]].std()
    df[df.columns[:-1]] = (df[df.columns[:-1]] - means) / stds
    df['move_time'] += 0.05
    print(df['move_time'].min())

    # nn_time_model = get_nn_time_model(df, epochs=5, training_split=0.8)
    # plot_actual_vs_predicted(nn_time_model, df)

    bayesian_nn_time_model = get_bayesian_nn_time_model(df, epochs=2, training_split=0.8)
    plot_actual_vs_predicted(bayesian_nn_time_model, df)




