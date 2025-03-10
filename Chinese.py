import os
import re
import math
import jieba
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ==================== 配置参数 ====================
BASE_DIR = "wiki_zh"  # 根目录
SUB_DIRS = [f"{chr(65)}{chr(65+i)}" for i in range(12)]  # AA-AM（按实际调整）
FILE_PREFIX = "wiki_"

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 分布式文件读取 ====================
def iter_wiki_files():
    """生成器：逐文件读取内容"""
    for subdir in SUB_DIRS:
        dir_path = os.path.join(BASE_DIR, subdir)
        if not os.path.isdir(dir_path):
            continue
            
        for i in range(100):  # 00-99
            filename = f"{FILE_PREFIX}{i:02d}"
            file_path = os.path.join(dir_path, filename)
            if not os.path.exists(file_path):
                continue
                
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                yield f.read()

# ==================== 流式处理管道 ====================
class ProcessingPipeline:
    def __init__(self):
        self.char_counter = Counter()
        self.word_counter = Counter()
        self.sentence_counter = Counter()
        self.total_chars = 0
        self.total_words = 0
        self.total_sentences = 0

    def process_text(self, text):
        # 清洗文本
        text = re.sub(r'[^\u4e00-\u9fa5。]', '', text)
        
        # 字符统计
        self._count_chars(text)
        
        # 分句处理
        sentences = [s for s in text.split('。') if s]
        self._count_sentences(sentences)
        
        # 分词处理
        words = jieba.lcut(text.replace('。', ''))  # 去句号后分词
        self._count_words(words)

    def _count_chars(self, text):
        chars = list(text.replace('。', ''))  # 排除句号
        self.char_counter.update(chars)
        self.total_chars += len(chars)

    def _count_words(self, words):
        self.word_counter.update(words)
        self.total_words += len(words)

    def _count_sentences(self, sentences):
        self.sentence_counter.update(sentences)
        self.total_sentences += len(sentences)

# ==================== 熵计算 ====================
def calculate_entropy(counter, total, alpha=1e-5):
    """带平滑的熵计算"""
    entropy = 0.0
    for count in counter.values():
        prob = (count + alpha) / (total + alpha * len(counter))
        entropy -= prob * math.log2(prob)
    return entropy

# ==================== 可视化 ====================
def plot_combined(results, titles):
    """组合图表"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # 字符分布
    items = results['char_top50']
    axes[0,0].bar(range(50), [v for _,v in items], tick_label=[k for k,_ in items])
    axes[0,0].set_title(titles['char_top'])
    plt.sca(axes[0,0])
    plt.xticks(rotation=90)

    # 词语分布
    items = results['word_top50']
    axes[0,1].bar(range(50), [v for _,v in items], tick_label=[k for k,_ in items])
    axes[0,1].set_title(titles['word_top'])
    plt.sca(axes[0,1])
    plt.xticks(rotation=90)

    # Zipf定律
    axes[1,0].loglog(results['char_zipf'][0], results['char_zipf'][1])
    axes[1,0].set_title(titles['char_zipf'])
    
    axes[1,1].loglog(results['word_zipf'][0], results['word_zipf'][1])
    axes[1,1].set_title(titles['word_zipf'])
    
    plt.tight_layout()
    plt.savefig('result-wiki.png', dpi=300, bbox_inches='tight')
    plt.show()
    

# ==================== 主流程 ====================
if __name__ == "__main__":
    # 初始化处理管道
    pipeline = ProcessingPipeline()
    
    # 流式处理数据
    for i, text in enumerate(iter_wiki_files()):
        pipeline.process_text(text)
        if (i+1) % 10 == 0:
            print(f"已处理 {i+1} 个文件...")
    
    # 计算结果
    results = {
        'char_top50': pipeline.char_counter.most_common(50),
        'word_top50': pipeline.word_counter.most_common(50),
        'char_zipf': (np.arange(1, len(pipeline.char_counter)+1), 
                      sorted(pipeline.char_counter.values(), reverse=True)),
        'word_zipf': (np.arange(1, len(pipeline.word_counter)+1),
                      sorted(pipeline.word_counter.values(), reverse=True))
    }
    
    # 计算熵值
    char_entropy = calculate_entropy(pipeline.char_counter, pipeline.total_chars)
    word_entropy = calculate_entropy(pipeline.word_counter, pipeline.total_words)
    sentence_entropy = calculate_entropy(pipeline.sentence_counter, pipeline.total_sentences)
    
    print("\n" + "="*40)
    print(f"字符熵: {char_entropy:.4f} bits/char")
    print(f"词语熵: {word_entropy:.4f} bits/word")
    print(f"句子熵: {sentence_entropy:.4f} bits/sentence")
    print("="*40 + "\n")
    
    # 生成图表
    plot_combined(results, {
        'char_top': '单字频率TOP50',
        'word_top': '词语频率TOP50',
        'char_zipf': '单字Zipf分布',
        'word_zipf': '词语Zipf分布'
    })

    