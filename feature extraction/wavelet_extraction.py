##### WAV Wavelet extraction #####

import scipy.io.wavfile
import matplotlib.pyplot as plt
import pywt
import numpy as np

(rate, data) = scipy.io.wavfile.read("C:\\Users\\Atulya\\Documents\\GitHub\\gender-classifier-using-voice\\male\\00445.wav")
data=data[5000:5200] #taking shorter duration

# Using the Daubechies2 coefficients
(ca, cd) = pywt.dwt(data,'db2')

# Selecting the soft threshold -- Less computational time
cat = pywt.threshold(ca, np.std(ca)/2, 'soft')
cdt = pywt.threshold(cd, np.std(cd)/2, 'soft')


# Using the Daubechies2 coefficients
data_rec = pywt.idwt(cat, cdt, 'db2')

# For debugging uncomment
#print data_rec

plt.close('all')

plt.subplot(211)
# Original coefficients
plt.plot(ca, '--*b')
plt.plot(cd, '--*r')
# Thresholded coefficients
plt.plot(cat, '--*c')
plt.plot(cdt, '--*m')
plt.legend(['ca','cd','ca_thresh', 'cd_thresh'], loc=0)
plt.grid('on')

plt.subplot(212)
plt.plot(data)
plt.plot(data_rec, 'r')
plt.legend(['original signal', 'reconstructed signal'])
plt.grid('on')
plt.show()