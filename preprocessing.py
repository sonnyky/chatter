import MeCab
import re

def wakati(input_str):
        '''分かち書き用関数
        引数 input_str : 入力テキスト
        返値 m.parse(wakatext) : 分かち済みテキスト'''
        wakatext = input_str
        m = MeCab.Tagger('-Owakati')
        #print(m.parse(wakatext))
        return m.parse(wakatext)


with open("sequence.txt", encoding='utf-8') as seqtex, open("./input.txt",'w', encoding='utf-8') as infile,\
 open("./output.txt",'w', encoding='utf-8') as outfile:
    print("data writing...")
    for line in seqtex:
        if line.find("input:") != -1:   #input文の処理
            line = re.sub('input: ', '', line)
            wakati_in = wakati(line)
            infile.write(wakati_in)

        else:  #output文の処理
            line = re.sub('output: ', '', line)
            wakati_out = wakati(line)
            outfile.write(wakati_out)

print("finished.")