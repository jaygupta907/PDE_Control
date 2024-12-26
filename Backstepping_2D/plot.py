import pandas as pd
import matplotlib.pyplot as plt

def plot_moving_average(csv_file, column_name, window_size):
    try:
        data = pd.read_csv(csv_file)
        if column_name not in data.columns:
            raise ValueError(f"Column '{column_name}' not found in the CSV file.")
        
        column_data = data[column_name][:3000]
        moving_avg = column_data.rolling(window=window_size).mean()
        
        plt.figure(figsize=(12, 6))
        plt.plot(column_data,linewidth=2,alpha=0.5)
        plt.plot(moving_avg,linewidth=2,linestyle='--')
        plt.title(f'Moving Average of {column_name}')
        plt.xlabel('Timestep')
        plt.ylabel(column_name)
        plt.legend(["Original", "Moving Average"])
        plt.grid()
        plt.show()

    except FileNotFoundError:
        print(f"File '{csv_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

csv_file = 'data_5000.csv'
column_name = 'action'
window_size = 100
plot_moving_average(csv_file, column_name, window_size)
