# 2019-Speech-Recognition
2019 语音识别

[论文]

一、  Listen, Attend and Spell
代码如下：

1、 https://github.com/thomasschmied/Speech_Recognition_with_Tensorflow
    

二、  TACOTRON: TOWARDS END-TO-END SPEECH SYNTHESIS
代码如下：

1、 https://github.com/audier/DeepSpeechRecognition/tree/master/tutorial/self-attention_tutorial.ipynb
    https://blog.csdn.net/chinatelecom08/article/details/85048019

[数据集]

1、 CSTRVCTK Corpus：   http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html

2、 LibriSpeech ASR corpus：  http://www.openslr.org/12/

3、 Switchboard-1 Telephone Speech Corpus：   https://catalog.ldc.upenn.edu/ldc97s62

4、 TED-LIUM Corpus： http://www-lium.univ-lemans.fr/en/content/ted-lium-corpus

5、 AIShell-1: An Open-Source Mandarin Speech Corpus and A Speech Recognition Baseline：   
    
    a、 进入到项目\datasets\AishellSpeech    【Terminal运行如下命令】  
        ```bash
        $ wget http://www.openslr.org/resources/33/data_aishell.tgz
        ```
    b、 运行aishell_speech.py Extract data_aishell.tgz:
        ```bash
        $ python aishell_speech.py
        ```
    c、 Extract wav files into train/dev/test folders:
        ```bash
        $ cd datasets/AishellSpeech/data_aishell/wav
        $ find . -name '*.tar.gz' -execdir tar -xzvf '{}' \;
        ```
    d、 Scan transcript data, generate features:
        ```bash
        $ cd demos/utils
        $ python pre_process.py
        ```
    
    Now the folder structure under data folder is sth. like:

    <pre>
    AishellSpeech/
        data_aishell.tgz
        data_aishell/
            transcript/
                aishell_transcript_v0.8.txt
            wav/
                train/
                dev/
                test/
        aishell.pickle
    </pre>
    
[参考]

1、 https://www.cnblogs.com/aibabel/p/11012794.html


