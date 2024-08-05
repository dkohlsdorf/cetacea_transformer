import pandas as pd
import numpy as np 

from scipy.io.wavfile import read, write
from numpy.fft        import fft


def raw(path):
    try:
        x = read(path)[1]
        if len(x.shape) > 1:
            return x[:, 0]
        return x
    except:
        print("Could not read file: {}".format(path))
        return np.zeros(0)


def spectrogram(audio, lo = 20, hi = 200, win = 512, step=128, normalize=True):
    spectrogram = []
    hanning = np.hanning(win)
    for i in range(win, len(audio), step):
        dft = np.abs(fft(audio[i - win: i] * hanning))
        if normalize:
            mu  = np.mean(dft)
            std = np.std(dft) + 1.0
            spectrogram.append((dft - mu) / std)
        else:
            spectrogram.append(dft)
    spectrogram = np.array(spectrogram)
    spectrogram = spectrogram[:, win//2:][:, lo:hi]
    return spectrogram


def dataset_supervised_windows(label, wavfile, lo, hi, win, step, raw_size, label_dict = None, limit = None):
    df        = pd.read_csv(label)
    df        = df.rename(columns=lambda x: x.strip())
    audio     = raw(wavfile)
    labels    = []
    instances = []
    ra = []

    fill = False
    if label_dict is None:
        label_dict = {}
        fill = True
        
    cur_label  = 0
    for _, row in df.iterrows():
        if limit is not None and len(instances) > limit:
            instances, ra, labels, label_dict
        start = row['offset']
        stop  = start + raw_size
        label = row['annotation'].strip()
        if label not in label_dict and fill:
            label_dict[label] = cur_label
            cur_label += 1
        w = audio[start:stop]
        s = spectrogram(w, lo, hi, win, step)
        f, t = s.shape
        ra.append(w)
        instances.append(s)
        labels.append(label_dict[label])
    return instances, ra, labels, label_dict


def dataset_unsupervised_regions(regions, wavfile, lo, hi, win, step, limit = 100):
    df        = pd.read_csv(regions)
    N         = len(df)
    audio     = raw(wavfile)
    instances = []
    for i, row in df.iterrows():
        start = row['starts']
        stop  = row['stops']
        w     = audio[start:stop]
        if len(w) > 0:
            s = spectrogram(w, lo, hi, win, step)
            instances.append(s)
        if len(instances) >= limit:
          return instances
    return instances


def process_label(row):
  T = (row['sample'] - FFT_WIN) / FFT_STEP if row['sample'] > 0 else 0
  max_label = None
  max_value = 0
  for label in LABELS.keys():
    if row[label] > max_value:
      max_value = row[label]
      max_label = label
  return max_label, T, row['instance']


def unroll(x):
  labels = []
  i = 0
  for t in range(x[0][1], x[-1][1] + 36):
    labels.append(x[i][0])
    if i < len(x) - 1 and t > x[i+1][1]:
      i += 1
  return labels


def labels_l2(path):
  df = pd.read_csv(path)
  instances = {}
  max_label = []
  instance_label = None
  offset = 0
  for _, row in df.iterrows():
    l, t, i = process_label(row)
    if i != instance_label:
      if(instance_label is not None):
        instances[int(instance_label)] = unroll(max_label)
      max_label = []
      instance_label = i
      offset = t
    max_label.append((l, int(t - offset)))
  return instances                                                                                                                                                                        
                                                                                                                                                                             
  
