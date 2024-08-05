import sys

from lib_cetacea.audio import  *
from lib_cetacea.parameters import *
from lib_cetacea.patches import *


def header():
    return """
    ========================================
    Cetacea - A Audio Dolphin Transformer

    ./train.py train [supervised|unsupervised] [l1|l2|l2_label] WAV CSV 
    ./train.py plot [patches|reconstruct] wav output
    

    by Daniel Kohlsdorf
    ========================================
    """


def parameters(cmd, supervision, level, wav, csv):
    return f"""
    command: {cmd}
    supervision: {supervision}
    level: {level}
    wav: {wav}
    csv: {csv}
    ========================================
    """


def train(supervision, level, wav, csv):
    if supervision == 'supervised' and level == 'l1':
        print(f"... process labeled l1: {wav} {csv}")
        x, _, y, _ = dataset_supervised_windows(csv, wav, FFT_LO, FFT_HI, FFT_WIN, FFT_STEP, RAW_AUDIO, LABELS)


def plot(inp, outp, mode = 'patches'):
    if mode == 'patches':
        x = raw(inp)
        s = spectrogram(x, FFT_LO, FFT_HI, FFT_WIN, FFT_STEP)
        patch_extractor = Patches(patch_size=PATCHES)

        w, h = s.shape
        patch_tensor = patch_extractor(s.reshape(1, w, h, 1))
        plot_grid(outp, patch_tensor, w, h, PATCHES)
    if mode == 'reconstruct':
        x = raw(inp)
        s = spectrogram(x, FFT_LO, FFT_HI, FFT_WIN, FFT_STEP)
        w, h = s.shape
        
        patch_extractor = Patches(patch_size=PATCHES)
        patch_tensor = patch_extractor(s.reshape(1, w, h, 1))
        reconstruction = reconstruct_from_patch(patch_tensor, PATCHES, w)[:, :, 0]
        plt.imshow(reconstruction)
        plt.axis('off')
        plt.savefig(outp)
        
    
if __name__ == "__main__":
    print(header())
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'train' and len(sys.argv) == 5:
            supervision, level, wav_file, csv_file = sys.argv[2:6]
            print(parameters(cmd, supervision, level, wav_file, csv_file))
            train(supervision, level, wav_file, csv_file)
        if cmd == 'plot' and len(sys.argv) == 5:
            mode, inp, outp = sys.argv[2:5]
            plot(inp, outp, mode)
