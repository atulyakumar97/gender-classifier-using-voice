import scipy.stats as stats
import numpy as np
import librosa

def get_frequencies(y,sr):
    '''Spectral centroid
    Each frame of a magnitude spectrogram is normalized and treated as a distribution over frequency bins,
    from which the mean (centroid) is extracted per frame.'''
    frequencies=librosa.feature.spectral_centroid(y=y, sr=sr)
    return frequencies
    
def get_features(frequencies):
    mean     =  np.mean(frequencies)
    median   = np.median(frequencies)
    mode     = stats.mode(frequencies).mode[0][0]
    std      = np.std(frequencies)
    sortedfreq = np.sort(frequencies)
    minfreq = sortedfreq[0][0]
    peak = sortedfreq[0][-1]
    q75,q25  = np.percentile(frequencies, [75 ,25])
    iqr      = q75 - q25
    return([mean, median, mode, std, minfreq, peak, q25, q75, iqr])

