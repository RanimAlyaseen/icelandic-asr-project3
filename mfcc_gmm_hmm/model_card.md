# MFCC + GMM-HMM model card

- **Task:** Icelandic automatic speech recognition (character-level)
- **Data:** frozen Samrómur 21.05 20-hour v1 train split only for fitting; dev only for selecting the character bigram language-model weight; test remains held out until configuration freeze.
- **Features:** 13 MFCCs, first and second temporal differences, 25 ms windows, 10 ms hop, per-utterance cepstral mean/variance normalization.
- **Acoustics:** diagonal-covariance Gaussian mixtures over each Icelandic character and whitespace, estimated from uniformly transcript-aligned training frames.
- **Sequence model:** character HMM with self-transition duration, an explicit boundary state for repeated letters, and add-smoothed character bigram transitions estimated only from train transcripts.
- **Decoder:** Viterbi search over character and boundary states.
- **Limitations:** transcript alignment is a simple uniform initialization; there is no expert phonetic lexicon or word pronunciation dictionary. This is a classical baseline and is expected to trail modern neural systems. It does not model cross-session speaker identity.
- **Evaluation:** use the repository's frozen references and `prepare.normalize_text`; report WER, CER, wall time and audio hours. Run test once only after freezing the chosen dev configuration.
- **Weights:** `models/mfcc_gmm_hmm/model.joblib` is the trained artifact; the SHA-256 and configuration are recorded in `results/mfcc_gmm_hmm/model_info/`.
