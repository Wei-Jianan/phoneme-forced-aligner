import shutil, subprocess, os, sys
import re
import wave
import hashlib
import jieba
import pypinyin
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from collections import namedtuple

IS_DELETE = True


def get_file_md5(file_path):
    with open(file_path, 'rb') as f:
        m = hashlib.sha1()
        while True:
            # open it with 'b' mode
            # data = f.read(1024).encode('utf-8')
            data = f.read(16 * 1024)
            # print(data)
            if not data:
                break
            m.update(data)
    return m.hexdigest()


class MissingInDictionaryException(Exception):
    def __init__(self, *args):
        pass


PhonemeDuration = namedtuple('Phoneme', ['yinjie', 'begin', 'end'])


class PhonemeForcedAligner():
    default_model_dir = Path(__file__).parent.joinpath('model').resolve(strict=False)
    default_dict = str(default_model_dir / 'dict')
    default_puncs = str(default_model_dir / 'puncs')
    default_mono = str(default_model_dir / 'monophones')
    pretrained_sample_rate = (8000, 16000)

    def __init__(self, auto_resample_rate=8000, dict_path=default_dict, puncs_path=default_puncs,
                 mono_path=default_mono):
        if auto_resample_rate not in self.pretrained_sample_rate:
            raise ValueError('auto resample rate must be equal to 8000 or 16000')
        self.auto_resample_rate = auto_resample_rate
        self.dict_set = self.load_dict(dict_path)
        self.pinyin_dict = {tuple(py[0] for py in pypinyin.pinyin(word, style=pypinyin.STYLE_NORMAL)): word
                            for word in self.dict_set}
        self.puncs_set = self.load_puncs(puncs_path)
        self.dict_file = self.open_temp_file(dict_path, suffix='.dict')
        self.puncs_file = self.open_temp_file(puncs_path, suffix='.puncs')
        self.mono_file = self.open_temp_file(mono_path)
        # print('dict hashing: ', get_file_md5(self.dict_file.name), self.dict_file.name)
        # print('puncs hashing: ', get_file_md5(self.puncs_file.name), self.puncs_file.name)


    def load_dict(self, dict_path):
        with open(dict_path, 'r', encoding='utf-8') as f:
            return list((line.split()[0] for line in f.readlines()))

    def load_puncs(self, puncs_path):
        with open(puncs_path, 'r', encoding='utf-8') as f:
            return set((line.strip() for line in f.readlines()))

    def set_dict(self, dict_set):
        self.dict_set = dict_set

    def extend_dict(self, dict_set):
        self.dict_set.extend(dict_set)

    def set_puncs(self, puncs_set):
        self.puncs_set = puncs_set

    def extend_puncs(self, puncs_set):
        self.puncs_set.extend(puncs_set)

    def align(self, text, wav_path):
        temp_wav_file = self._write_audio(wav_path)
        wav_path = temp_wav_file.name
        sample_rate = self.get_sample_rate(wav_path)
        if sample_rate not in self.pretrained_sample_rate:
            temp_wav_file = self._resample(wav_path, self.auto_resample_rate)
        else:
            # temp_wav_file = self.open_temp_file(wav_path, suffix='.wav')
            self.auto_resample_rate = sample_rate

        # print('wav file shashing: ', get_file_md5(temp_wav_file.name), temp_wav_file.name)
        with TemporaryDirectory() as temp_dir:
            plp_path = self._hcopy(temp_wav_file, temp_dir)
            # print('plp file shashing: ', get_file_md5(plp_path), plp_path)
            mlf_path = self.generate_mlf(text, temp_dir)
            # print('mlf file shashing: ', get_file_md5(mlf_path), mlf_path)

            aligned_path = self._hvite(temp_dir)
            # print('aligned file hashing: ', aligned_path)
            phoneme_durations = self._gen_res(aligned_path, mlf_path, sys.stdout)

        temp_wav_file.close()
        return phoneme_durations



    def _gen_res(self, aligned_path, mlf_path, result_path):
        def start(line, phoneme_durations):
            line = line.split()
            phoneme_durations.append(
                PhonemeDuration(yinjie=line[-1], begin=float(line[0]) / 1000_0000, end=float(line[1]) / 1000_0000))
            return ending

        def ending(line, phoneme_durations):
            line = line.split()
            if len(line) == 1:
                return end
            if len(line) == 4:
                phoneme_durations[-1] = phoneme_durations[-1]._replace(end=float(line[1]) / 1000_0000)
                return ending
            elif len(line) == 5:
                return start(' '.join(line), phoneme_durations)

        def end(line, phoneme_durations):
            return end

        with open(aligned_path, 'r', encoding='utf-8') as f:
            aligned_lines = f.readlines()[2:-1]

        with open(mlf_path, 'r', encoding='utf-8') as f:
            mlf_lines = f.readlines()[2:-1]

        phoneme_durations = []
        state_machine = start
        for line in aligned_lines:
            state_machine = state_machine(line, phoneme_durations)
        assert (len(mlf_lines) == len(phoneme_durations))
        for i, word in enumerate(mlf_lines):
            # todo multi yinjie character case
            phoneme_durations[i] = phoneme_durations[i]._replace(yinjie=word.split()[0])
        phoneme_durations = filter(lambda phoneme_duration: phoneme_duration.yinjie != 'sp', phoneme_durations)
        phoneme_durations = map(lambda phoneme_duration:
                                phoneme_duration._replace(yinjie=pypinyin.pinyin(phoneme_duration.yinjie,
                                                                                 style=pypinyin.STYLE_NORMAL)[0][0]),
                                phoneme_durations)
        return phoneme_durations

    def cut_text(self, text):
        # if text not in self.dict_set and len(text) == 1:
        #     raise MissingInDictionaryException('this word: {} does no find in dictionary'.format(text))
        # elif text in self.dict_set:
        #     return [text]

        splited_text = jieba.cut(text)
        final_words = []
        for word in splited_text:
            if word not in self.dict_set:
                final_words += list(word)
            else:
                final_words.append(word)
        final_words = filter(lambda word: word not in self.puncs_set, final_words)

        final_words = filter(lambda word: word not in ('\n', ' '), final_words)
        # for word in final_words:
        #     if word not in self.dict_set:
        #         raise MissingInDictionaryException('this word: {} does no find in dictionary'.format(word))
        return final_words

    def get_sample_rate(self, wav_path):
        with wave.open(wav_path) as wav:
            return wav.getframerate()

    def _resample(self, wav_path, sample_rate):
        # file_suffix = Path(wav_path).suffix
        resampled_wav_file = NamedTemporaryFile(suffix='.wav', mode='w+b', delete=IS_DELETE)
        subprocess.check_call(['sox', wav_path, '-r', str(sample_rate), resampled_wav_file.name])
        resampled_wav_file.seek(0)
        return resampled_wav_file

    def _hcopy(self, wav_file, dir):
        # plp_file = NamedTemporaryFile(suffix='.plp', delete=IS_DELETE)
        plp_path = Path(dir).joinpath('tmp.plp')
        config_path, _, _ = self.get_config_hmmdefs_macros()
        subprocess.check_call(['HCopy', '-C', config_path, wav_file.name, plp_path])
        # plp_file.seek(0)
        return plp_path

    def _hvite(self, dir):
        config_path, hmmdefs_path, macros_path = self.get_config_hmmdefs_macros()
        aligned_path = './tmp.aligned'

        subprocess.check_call(['HVite', '-T', str(1), '-a', '-m', '-t', str(10000.0), str(10000.0), str(10000.0),
                               '-I', './tmp.mlf', '-H', macros_path, '-H', hmmdefs_path,
                               '-i', aligned_path, self.dict_file.name, self.mono_file.name, './tmp.plp'],
                              cwd=Path(dir).resolve(),
                              stdout=open(os.devnull, 'w'))

        return Path(dir).joinpath(aligned_path)

    def generate_mlf(self, text, dir):
        mlf_path = Path(dir).joinpath('tmp.mlf')
        lab_path = '"./tmp.lab"\n'
        with open(mlf_path, 'w', encoding='utf-8') as mlf_file:
            mlf_file.write('#!MLF!#\n')
            mlf_file.write(lab_path)
            mlf_file.write('sp\n')
            words = self.cut_text(text)
            # TODO
            words = map(lambda word: '\n'.join(self._parse_word(word)), words)
            # mlf_file.write('\nsp\n'.join(words))
            mlf_file.write('\n'.join(words))
            mlf_file.write('\nsp\n.\n')
            mlf_file.seek(0)
        with open(mlf_path, 'r', encoding='utf-8') as f:
            print("mlf file: \n{}".format(f.read()))

        return mlf_path

    def _parse_word(self, word):
        word = list(word)
        pys = tuple(py[0] for py in pypinyin.pinyin(word, style=pypinyin.STYLE_NORMAL))
        for i, (py, char) in enumerate(zip(pys, word)):

            if char in self.dict_set:
                pass
            else:
                if (py, ) in self.pinyin_dict:
                    word[i] = self.pinyin_dict[(py, )]
                else:
                    raise MissingInDictionaryException('this word: {} does no find in dictionary'.format(char))
        return word


    def open_temp_file(self, file_path, suffix=None):
        if suffix == '.wav':
            return self._write_audio(file_path)
        file_suffix = Path(file_path).suffix if suffix is None else suffix
        temp_file = NamedTemporaryFile(suffix=file_suffix, delete=IS_DELETE)
        with open(file_path, mode='rb') as f:
            shutil.copyfileobj(f, temp_file)
        temp_file.seek(0)
        return temp_file

    def _write_audio(self, media_path):
        f = NamedTemporaryFile(mode='wb', suffix='.wav')
        subprocess.check_call(['ffmpeg', '-y', '-i', media_path, f.name],
                              stderr=open(os.devnull, 'w'))
        return f

    def write_temp_file(self, iterable_line, suffix):
        temp_file = NamedTemporaryFile(suffix=suffix, mode='w+', encoding='utf-8')
        for line in iterable_line:
            temp_file.write(line + '\n')
        temp_file.seek(0)
        return temp_file

    def get_config_hmmdefs_macros(self):
        return (self.default_model_dir / str(self.auto_resample_rate) / 'config',
                self.default_model_dir / str(self.auto_resample_rate) / 'hmmdefs',
                self.default_model_dir / str(self.auto_resample_rate) / 'macros')


if __name__ == '__main__':
    aligner = PhonemeForcedAligner()
    phoneme_durations = aligner.align('春走在路上，看看世界无限宽', './11.wav')
    print(phoneme_durations)
