# coding=utf-8
import os
import re
import jieba
import time
import datetime

"""使用jieba分词进行中文分词"""
separated_word_file_dir = "word_separated"
# 清华新闻语料库
types = ["体育", "娱乐", "家居", "彩票", "房产", "教育", "时尚", "时政", "星座", "游戏", "社会", "科技", "股票", "财经"]


def ch_and_en_word_extraction(content_raw):
    """抽取中文和英文"""
    pattern = re.compile(u"([\u4e00-\u9fa5a-zA-Z0-9]+)")
    re_data = pattern.findall(content_raw)
    clean_content = ' '.join(re_data)
    return clean_content


def clean_str(s):
    # s = s.strip('\n')  # 换行符
    # s = re.sub("[\t\n\r]*", '', s)  # tab, newline, return
    s = re.sub('\|+',' ',s)
    s = re.sub('\s+',' ',s)
    s = s.strip()  # 前后的空格
    # s = re.sub("<(\S*?)[^>]*>.*?</\1>|<.*? />", '', s)  # html标签
    # s = re.sub("&nbsp+|&lt+|&gt+", '', s)  # html中的空格符号,大于，小于
    # s = re.sub("[a-zA-z]+://[^\s]*", '', s)  # URL
    # s = re.sub(r'([\w-]+(\.[\w-]+)*@[\w-]+(\.[\w-]+)+)', '', s)  # email
    # 标点符号,需要先转utf-8，否则符号匹配不成功
    # s = re.sub(ur"([%s])+" % zhon.hanzi.punctuation, " ", s.decode('utf-8'))
    # 抽取中文和英文
    # s = ch_and_en_word_extraction(s)
    return s


def separate_words(infile, outfile):
    try:
        outf = open(outfile, 'w')
        inf = open(infile, 'r')

        space = ' '
        # print 'separate '+infile
        isFirstLine = True
        for line in inf.readlines():
            line = clean_str(line)
            # 除空行
            if not len(line):
                continue
            seg_list = jieba.cut(line)
            """此处需要循环每个单词编码为utf-8，jieba.cut将结果转为了unicode编码，
            直接write(space.join(seg_list))会报编码错误"""
            for word in seg_list:
                if not len(word.strip()):
                    continue
                try:
                    word = word.strip().encode('UTF-8')
                except:
                    continue
                outf.write(word)
                outf.write(space)
            if isFirstLine:
                outf.write("。")
                isFirstLine = False
        outf.write('\n')
        # close file stream
        outf.close()
        inf.close()
    except:
        print "error occured when write to " + outfile


def is_target_dir(path):
    if os.path.dirname(path).split("/")[-1] in types and not re.match(".DS_Store", os.path.basename(path)):
        return True
    else:
        return False


def explore(dir):
    for root, dirs, files in os.walk(dir):
        for file in files:
            path = os.path.join(root, file)
            if is_target_dir(path):
                child_dir = os.path.join(root, separated_word_file_dir)
                if not os.path.exists(child_dir):
                    os.mkdir(child_dir)
                    print "make dir: " + child_dir
                separate_words(path, os.path.join(child_dir, file))


def do_batch_separate(path):
    if os.path.isfile(path) and is_target_dir(path):
        separate_words(path, os.path.join(root, separated_word_file_dir, path))
    if os.path.isdir(path):
        explore(path)


original_dir = "THUCNews_deal_title/"
now = datetime.datetime.now()
print "separate word begin time:", now
begin_time = time.time()
do_batch_separate(original_dir)
end_time = time.time()
now = datetime.datetime.now()
print "separate word,end time:", now
print "separate word,time used:" + str(end_time - begin_time) + "秒"


"""将所有语料，整合成csv类型文件，文件格式：type|content"""
split_mark = '|'

def combine_file(file, outfile):
    # the type of file ，file示例：xxx/互联网／xxx/xxx.txt
    label = os.path.dirname(file).split('/')[-2]
    content = open(file).read()
    #     print "content:"+content
    #     print "len:",len(content)
    if len(content) > 1:  # 排除前面步骤中写文件时，内容为只写入一个空格的情况
        new_content = label + split_mark + content
        #         print "new_content:\n " + new_content
        open(outfile, "a").write(new_content)


def do_combine(dir, outfile):
    print "deal with dir: " + dir
    for root, dirs, files in os.walk(dir):
        for file in files:
            match = re.match(r'\d+\.txt', file)
            if match:
                path = os.path.join(root, file)
#                print "combine " + path
                combine_file(path, outfile)


def create_csv_file(dir, filename):
    csv_title = "type"+ split_mark + "content\n"
    filepath = os.path.join(dir, filename + '.csv')
    open(filepath, 'w').write(csv_title)
    return filepath


base_dir = "THUCNews_deal_title/"
"""创建处理后的数据集的目录"""
dataset_dir = os.path.join(base_dir, "dataset")
if not os.path.exists(dataset_dir):
    os.mkdir(dataset_dir)

"""创建每个type目录对应的csv文件，并将一个type目录下的文件写到同一个对应的csv文件"""
# 清华新闻语料库
type_name_list = ["体育", "娱乐", "家居", "彩票", "房产", "教育", "时尚", "时政", "星座", "游戏", "社会", "科技", "股票", "财经"]

combine_begin_time = datetime.datetime.now()
print "combine begin time:",combine_begin_time
for name in type_name_list:
    path = create_csv_file(dataset_dir, name)
    print "going to combine file to  " + path
    do_combine(os.path.join(base_dir, name, "word_separated"), path)

combine_end_time = datetime.datetime.now()
print "combine end time:",combine_end_time

"""随机采样每个类别的约20%作为测试集,80%作为训练集"""
import random

def extract_test_and_train_set(filepath, train_file, test_file):
    try:
        test_f = open(test_file, 'a')
        train_f = open(train_file, 'a')
        try:
            with open(filepath) as f:
                is_title_line = True
                for line in f.readlines():
                    if is_title_line:
                        is_title_line = False
                        continue
                    if not len(line):
                        continue
                    if random.random() <= 0.2:
                        test_f.write(line)
                    else:
                        train_f.write(line)
        except:
            print "IO ERROR"
        finally:
            test_f.close()
            train_f.close()
    except:
        print "can not open file"


def do_extract(source_dir, train_f, test_f):
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if re.match("test|train\.csv", file) or not re.match(".*\.csv", file):
                continue
            path = os.path.join(root, file)
            print "extract file: " + path
            extract_test_and_train_set(path, train_f, test_f)


# do extract
dataset_dir = "THUCNews_deal_title/dataset/"
train_dataset = os.path.join(dataset_dir, "train.csv")
test_dataset = os.path.join(dataset_dir, "test.csv")
if not os.path.exists(train_dataset):
    print "create file: " + train_dataset
    open(train_dataset, 'w').write("type"+ split_mark+"content\n")
if not os.path.exists(test_dataset):
    print "create file:" + test_dataset
    open(test_dataset, 'w').write("type"+split_mark+"content\n")

do_extract(dataset_dir, train_dataset, test_dataset)


"""清洗数据，除掉停用词，剔除坏样本"""

# 
# def clean_stopwords(content_raw, stopwords_set):
#     content_list = [x for x in re.split(' +|\t+',content_raw) if x != '']
#     common_set = set(content_list) & stopwords_set
#     new_content = filter(lambda x: x not in common_set, content_list)
#     return new_content
# 
# 
# def do_clean_stopwords(content_file, stopwords_file, newfile):
#     print "clean stopwords in " + content_file
#     stopwords = []
#     # 获取停用词
#     with open(stopwords_file) as fi:
#         for line in fi.readlines():
#             stopwords.append(line.strip())
#     newf = open(newfile, 'w')
#     with open(content_file) as f:
#         for line in f.readlines():
#             type_content = line.split(split_mark)
#             content_raw = type_content[1]
#             new_cont = clean_stopwords(content_raw, set(stopwords))
#             new_line = type_content[0] + split_mark + ' '.join(new_cont).strip()
#             newf.write(new_line)
#             newf.write('\n')
#     newf.close()
# 
# test_file = "THUCNews_deal_title/dataset/test.csv"
# train_file = "THUCNews_deal_title/dataset/train.csv"
# new_test_file = "THUCNews_deal_title/dataset/cleaned_test.csv"
# new_train_file = "THUCNews_deal_title/dataset/cleaned_train.csv"
# stop_words_file = "THUCNews_deal_title/dataset/news.stopwords.txt"
# do_clean_stopwords(test_file,stop_words_file,new_test_file)
# do_clean_stopwords(train_file,stop_words_file,new_train_file)
