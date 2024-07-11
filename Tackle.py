import numpy as np
from scipy import signal
from sklearn.neighbors import KernelDensity


def filter_wave(x, FS):
    fs = FS  # FS is the sampling frequency
    signalarray = np.array(x)

    nyquist = 0.5 * fs
    lowcut = 40
    lowcut_normalized = lowcut / nyquist

    # Design the Butterworth low-pass filter
    b, a = signal.butter(8, lowcut_normalized, btype='low')

    # Apply the filter to the signal
    filtered_signal = signal.lfilter(b, a, signalarray)

    return filtered_signal


def repointcloud(Nu, S, T, M):
    Time_window = Nu - M*T
    fnn_num = M
    Timelag = T
    acc = []
    # After confirming the embedding dimension and time delay, the final step is to perform a phase space
    # reconstruction to turn the original data into a point cloud: Reconstruct is the reconstructed point cloud array.
    for i in range(Nu):
        acc.append(S[i])
    Reconstruct = np.zeros([Time_window, fnn_num], dtype=np.float32)
    numb2 = 0
    for i in range(Time_window):
        for j in range(fnn_num):
            Reconstruct[i][j] = acc[j * Timelag + numb2]
        numb2 = numb2 + 1

    return Reconstruct


def KDEAN(data, band, value):

    # Estimating multi-dimensional Gaussian kernel densities
    kde = KernelDensity(bandwidth=band, kernel='gaussian')
    kde.fit(data)

    # Calculate density estimates for data points
    log_densities = kde.score_samples(data)

    KK = list(log_densities).copy()
    KK.sort()
    # Select a threshold to retain points with low density
    threshold = np.percentile(log_densities, value)

    index = []
    for t2 in range(len(data)):
        if log_densities[t2] <= threshold:
            index.append(t2)
    # Filtering data based on thresholds
    filtered_data = data[log_densities <= threshold]
    '''
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2, 10))
    #
    # ax1.scatter(data[:, 0], data[:, 1], alpha=0.5, label='Original Data')
    # ax1.set_title('Original Data')
    #
    # ax2.scatter(filtered_data[:, 0], filtered_data[:, 1], alpha=0.5, label='Filtered Data')
    # ax2.set_title('Filtered Data')
    # plt.show()
    # print(1)
    '''
    return [filtered_data, index]


if __name__ == '__main__':
    print(1)

