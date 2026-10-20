
from mmsdk import mmdatasdk

print("Starting CMU-MOSI download...")

# Download the pre-extracted text, audio and visual features
mosi = mmdatasdk.mmdataset(
    mmdatasdk.cmu_mosi.highlevel,
    "cmumosi/"
)

print("\nFeature streams downloaded:")
for stream in mosi.keys():
    print("-", stream)

# Download sentiment labels
mosi.add_computational_sequences(
    mmdatasdk.cmu_mosi.labels,
    "cmumosi/"
)

print("\nAll requested features and labels downloaded.")
print("Files are saved in the cmumosi/ directory.")
