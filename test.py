from htkaligner import PhonemeForcedAligner

if __name__ == "__main__":
    aligner = PhonemeForcedAligner()
    with open('000.txt', encoding='utf-8') as f:
        text = f.read()
        print(text)
    phoneme_durations = aligner.align(text, 'vid_000.wav')
    for phoneme_duration in phoneme_durations:
        print(phoneme_duration.yinjie, phoneme_duration.begin, phoneme_duration.end)

