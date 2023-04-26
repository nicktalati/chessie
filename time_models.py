import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers


def log_normal_loss(y_true, y_pred):
    return -y_pred.log_prob(y_true)

def load_bayesian_nn_time_model(model_path):
    custom_objects = {
        'DistributionLambda': tfpl.DistributionLambda,
        'LogNormal': tfd.LogNormal,
        'log_normal_loss': log_normal_loss
    }
    
    loaded_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return loaded_model


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
    print(xs)
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
                                   loss=log_normal_loss)
    
    bayesian_nn_time_model.fit(train_xs,
                               train_ys,
                               validation_data=(test_xs, test_ys),
                               epochs=epochs)
    
    return bayesian_nn_time_model


def plot_actual_vs_predicted(model, df, means, stds):
    df_copy = df.copy()
    xs = (df[df.columns[:-1]] * stds) + means
    df_predicted = model.predict(df[df.columns[:-1]])
    df_copy["predicted_time"] = df_predicted
    df_copy[df_copy.columns[:-2]] = xs
    valid_times = (df_copy['move_time'] < 45) & (df_copy['predicted_time'] < 45)
    df_copy = df_copy.loc[valid_times]
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


def plot_side_by_side(model, df, means, stds):
    df_copy = df.copy()
    xs = (df[df.columns[:-1]] * stds) + means
    df_predicted = model.predict(df[df.columns[:-1]])
    df_copy["predicted_time"] = df_predicted
    df_copy[df_copy.columns[:-2]] = xs
    valid_times = (df_copy['move_time'] < 45) & (df_copy['predicted_time'] < 45)
    df_copy = df_copy.loc[valid_times]
    valid_times = (df_copy['move_time'] < 45) & (df_copy['predicted_time'] < 45)
    df_copy = df_copy.loc[valid_times]

    for idx, column in enumerate(df_copy.columns[:-2]):
        fig, axs = plt.subplots(1, 2, figsize=(16, 4))

        # Plot the actual time
        axs[0].scatter(df_copy[column], df_copy['move_time'], color='blue', alpha=0.003)
        axs[0].set_title(f"{' '.join(column.split('_'))} vs. actual move time")
        axs[0].set_xlabel(' '.join(column.split('_')))
        axs[0].set_ylabel("actual move time")

        # Plot the predicted time
        axs[1].scatter(df_copy[column], df_copy['predicted_time'], color='red', alpha=0.003)
        axs[1].set_title(f"{' '.join(column.split('_'))} vs. predicted move time")
        axs[1].set_xlabel(' '.join(column.split('_')))
        axs[1].set_ylabel("predicted move time")

        plt.tight_layout()
        plt.show()


def plot_with_color_scale(model, df, means, stds):
    df_copy = df.copy()
    xs = (df[df.columns[:-1]] * stds) + means
    df_predicted = model.predict(df[df.columns[:-1]])
    df_copy["predicted_time"] = df_predicted
    df_copy[df_copy.columns[:-2]] = xs
    valid_times = (df_copy['move_time'] < 45) & (df_copy['predicted_time'] < 45)
    df_copy = df_copy.loc[valid_times]
    df_copy[['white_material', 'black_material']] += np.random.normal(0, 0.5, size=(len(df_copy), 2))

    # Define the columns for the joint distributions
    joint_columns = [['white_elo', 'black_elo'],
                     ['white_time_left', 'black_time_left'],
                     ['white_material', 'black_material']]

    # Define the color maps
    cmap = cm.get_cmap('coolwarm')
    norm = mcolors.Normalize(vmin=0, vmax=5)

    # Iterate through the joint columns
    for columns in joint_columns:
        fig, axs = plt.subplots(1, 3, figsize=(16, 4), gridspec_kw={'width_ratios': [1, 1, 0.05]})

        # Plot the actual time
        colors = cmap(norm(df_copy['move_time']))
        colors[:, 3] = 0.005
        scatter = axs[0].scatter(df_copy[columns[0]], df_copy[columns[1]], c=colors)
        axs[0].set_title(f"{' '.join(columns[0].split('_'))} vs {' '.join(columns[1].split('_'))} (actual move time)")
        axs[0].set_xlabel(' '.join(columns[0].split('_')))
        axs[0].set_ylabel(' '.join(columns[1].split('_')))

        # Plot the predicted time
        colors = cmap(norm(df_copy['predicted_time']))
        colors[:, 3] = 0.005
        scatter = axs[1].scatter(df_copy[columns[0]], df_copy[columns[1]], c=colors)
        axs[1].set_title(f"{' '.join(columns[0].split('_'))} vs {' '.join(columns[1].split('_'))} (predicted move time)")
        axs[1].set_xlabel(' '.join(columns[0].split('_')))
        axs[1].set_ylabel(' '.join(columns[1].split('_')))

        # Remove the ticks and spines of the empty axis
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        for spine in axs[2].spines.values():
            spine.set_visible(False)

        # Add a colorbar to the figure
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=axs[2], label="move time")

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    csv_path = 'training_data2.csv'
    df = pd.read_csv(csv_path)
    valid_w_elos = (df['white_elo'] <= 1600) & (df['white_elo'] >= 1400)
    valid_b_elos = (df['black_elo'] <= 1600) & (df['black_elo'] >= 1400)
    df = df.loc[valid_w_elos & valid_b_elos]

    means = df[df.columns[:-1]].mean()
    stds = df[df.columns[:-1]].std()
    print(means, stds)
    df[df.columns[:-1]] = (df[df.columns[:-1]] - means) / stds
    df['move_time'] += 0.05

    # nn_time_model = get_nn_time_model(df, epochs=10, training_split=0.8)
    # plot_actual_vs_predicted(nn_time_model, df)

    bayesian_nn_time_model = get_bayesian_nn_time_model(df, epochs=10, training_split=0.8)
    tf.keras.models.save_model(bayesian_nn_time_model, 'bayesian_nn_time_model.h5')
    # bayesian_nn_time_model = load_bayesian_nn_time_model('bayesian_nn_time_model.h5')
    # tf.keras.models.save_model(bayesian_nn_time_model, 'bayesian_nn_time_model.h5')

    plot_side_by_side(bayesian_nn_time_model, df, means, stds)
    plot_with_color_scale(bayesian_nn_time_model, df, means, stds)

    # plot_actual_vs_predicted(bayesian_nn_time_model, df, means, stds)




