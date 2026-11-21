# Azaa Almousli

- Student ID **2309115421**

## MFCC + GMM-HMM for Icelandic ASR

I implemented and evaluated the project's classical Icelandic speech-recognition baseline. Automatic speech recognition (ASR) converts spoken audio into written text. This system uses the shared, cleaned Samrómur 21.05 recordings and the project's frozen train, development (dev), and test manifests so that its outputs can be compared with the other project models.

### Features and acoustic model

The feature extractor reads each cleaned mono recording at 16 kHz, divides it into 25 ms frames with a 10 ms step, and computes 13 mel-frequency cepstral coefficients (MFCCs) for each frame. It adds first- and second-order time differences, giving 39 values per frame, then normalizes the features within each recording.

The acoustic model fits diagonal-covariance Gaussian mixture models (GMMs) to the Icelandic character inventory, including word boundaries. A left-to-right hidden Markov model (HMM) represents how those characters can continue, pause at a boundary, or transition to the next character. The transition probabilities use character pairs counted from train transcripts only, with smoothing for unseen pairs. Viterbi decoding finds the most likely character sequence for each recording. The train vocabulary, MFCC settings, mixture count, and transition settings are recorded in the package configuration.

Because this baseline does not use a forced aligner, training assigns transcript characters to audio frames by a simple uniform alignment to initialize the acoustic models. This is an approximation; the system also does not use a curated Icelandic phonetic pronunciation lexicon. Those limitations are important when interpreting its accuracy.

### Training and evaluation

I first ran a one-train/one-dev utterance smoke fit to check that LFS audio loading, feature extraction, model fitting, decoding, scoring, and output writing worked. I then fitted the full model on all **10,488 frozen train utterances** (5,379,049 feature frames). Training and model serialization took **410.4 seconds on CPU**.

I selected the character language-model weight using the complete **1,314-utterance dev split**. The compared weights were 0.5, 1.0, and 1.5; 1.5 produced the lowest dev WER. Dev WER was **101.46%**, CER was **87.51%**, and decoding took **89.4 seconds** for about two hours of audio (RTF 0.0124).

After saving that configuration, I evaluated the held-out test split once: **1,884 utterances**, about three hours. Test WER was **101.47%**, CER was **87.74%**, and decoding took **135.9 seconds** (RTF 0.0126). The high error rates mean this implementation is a weak classical reference point, not a competitive recognizer. The selected configuration was not adjusted using test results.

### Reproducibility and handoff

The code, pinned dependencies, configuration, usage instructions, model card, and model access are in [`mfcc_gmm_hmm/`](../mfcc_gmm_hmm/). The fitted model is tracked with Git LFS at `models/mfcc_gmm_hmm/model.joblib`. The complete dev and test hypotheses are in `results/mfcc_gmm_hmm/predictions/`; the corresponding score, environment, model hash, and timing records are in `results/mfcc_gmm_hmm/model_info/`. These CSV predictions are ready for Saleh's scoring and statistical comparison against the other systems.
