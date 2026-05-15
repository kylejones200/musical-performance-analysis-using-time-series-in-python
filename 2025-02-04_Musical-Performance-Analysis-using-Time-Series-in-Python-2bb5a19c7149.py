# Description: Short example for Musical Performance Analysis using Time Series in Python.


import logging

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# Load audio
y, sr = librosa.load(librosa.example("trumpet"))


# Extract multiple musical features
def extract_musical_features(y, sr):
    # Timing
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Pitch
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=2000)

    # Energy/dynamics
    rms = librosa.feature.rms(y=y)[0]

    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    return beat_times, f0, rms, spectral_centroid, tempo



def main():
    # Extract features
    beat_times, f0, rms, spectral_centroid, tempo = extract_musical_features(y, sr)

    # Plot musical features with context
    plt.figure(figsize=(15, 10))

    # Plot 1: Pitch contour with context
    plt.subplot(3, 1, 1)
    times = librosa.times_like(f0)
    valid_f0 = f0[~np.isnan(f0)]  # Remove NaN values
    valid_times = times[~np.isnan(f0)]
    plt.plot(valid_times, librosa.hz_to_midi(valid_f0), label="Pitch (MIDI)")
    plt.title("Pitch Contour with Musical Context")
    plt.ylabel("MIDI Note")

    # Plot 2: Dynamic changes
    plt.subplot(3, 1, 2)
    times_rms = librosa.times_like(rms)
    plt.plot(times_rms, librosa.amplitude_to_db(rms), label="Dynamics")
    plt.title("Dynamic Changes")
    plt.ylabel("dB")

    # Plot 3: Spectral features
    plt.subplot(3, 1, 3)
    times_spec = librosa.times_like(spectral_centroid)
    plt.plot(times_spec, spectral_centroid, label="Spectral Centroid")
    plt.title("Timbral Changes")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")

    plt.tight_layout()
    plt.savefig("musical_features.png")
    plt.show()

    # Print detected musical events
    logger.info("\nDetected Musical Events:")
    logger.info(
        f"Average tempo: {np.asscalar(tempo):.1f} BPM"
        if np.isscalar(tempo)
        else f"Average tempo: {tempo[0]:.1f} BPM"
    )
    logger.info(f"Number of detected beats: {len(beat_times)}")
    logger.info(f"Duration: {len(y) / sr:.1f} seconds")

    # Additional musical statistics
    logger.info("\nMusical Statistics:")
    valid_f0 = f0[~np.isnan(f0)]
    if len(valid_f0) > 0:
        mean_midi = librosa.hz_to_midi(np.mean(valid_f0))
        pitch_range = librosa.hz_to_midi(np.max(valid_f0)) - librosa.hz_to_midi(
            np.min(valid_f0)
        )
        logger.info(f"Average pitch: {mean_midi:.1f} MIDI")
        logger.info(f"Pitch range: {pitch_range:.1f} semitones")

    mean_db = np.mean(librosa.amplitude_to_db(rms))
    logger.info(f"Average dynamics: {mean_db:.1f} dB")

    # Additional analysis for musical context
    if len(valid_f0) > 0:
        logger.info("\nMusical Context:")
        # Calculate note distribution
        midi_notes = librosa.hz_to_midi(valid_f0)
        unique_notes, note_counts = np.unique(np.round(midi_notes), return_counts=True)
        most_common_note = unique_notes[np.argmax(note_counts)]
        logger.info(
            f"Most common note: MIDI {most_common_note:.0f} ({librosa.midi_to_note(most_common_note)})"
        )

        # Calculate average note duration
        mean_beat_interval = np.mean(np.diff(beat_times))
        logger.info(f"Average beat duration: {mean_beat_interval:.3f} seconds")


if __name__ == "__main__":
    main()
