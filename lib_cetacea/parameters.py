FFT_STEP     = 128
FFT_WIN      = 512
FFT_HI       = 230
FFT_LO       = 100
D            = FFT_WIN // 2 - FFT_LO - (FFT_WIN // 2 - FFT_HI)
RAW_AUDIO    = 5120
T            = int((RAW_AUDIO - FFT_WIN) / FFT_STEP)
PATCHES      = 10

LABELS = {
  "NOISE": 0,
  "ECHO": 1,
  "BURST": 2,
  "WSTL_DOWN": 3,
  "WSTL_UP": 4
}
