from pydub import AudioSegment

in_add = ""
out_add = ""
sound = AudioSegment.from_mp3(in_add)
sound.export(out_add, format="wav")