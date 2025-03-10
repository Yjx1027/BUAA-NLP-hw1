import nltk
import re
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import gutenberg
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ==================== 初始化配置 ====================
nltk.download('gutenberg', quiet=True)
nltk.download('punkt', quiet=True)

SHAKESPEARE_WORKS = [
    'shakespeare-caesar.txt',
    'shakespeare-hamlet.txt',
    'shakespeare-macbeth.txt'
]

# ==================== 数据加载管道 ====================
class ShakespeareLoader:
    def __init__(self):
        self.fileids = SHAKESPEARE_WORKS
        
    def stream_text(self):
        """文本流生成器"""
        for fileid in self.fileids:
            text = gutenberg.raw(fileid)
            yield re.sub(r'\n{2,}', '\n', text)  # 合并多个空行

# ==================== 分析引擎 ====================
class LinguisticAnalyzer:
    def __init__(self):
        self.char_stats = Counter()
        self.word_stats = Counter()
        self.counts = {
            'chars': 0,
            'words': 0,
            'files': 0
        }
    
    def process(self, text):
        """处理单个文本"""
        clean_text = self._clean_text(text)
        self._analyze_chars(clean_text)
        self._analyze_words(clean_text)
        self.counts['files'] += 1
        
    def _clean_text(self, text):
        """文本清洗"""
        return text.lower().strip()
    
    def _analyze_chars(self, text):
        """字符级分析"""
        chars = re.findall(r'[a-z]', text)
        self.char_stats.update(chars)
        self.counts['chars'] += len(chars)
    
    def _analyze_words(self, text):
        """词汇级分析"""
        words = re.findall(r"\b[a-z']{2,}\b", text)
        self.word_stats.update(words)
        self.counts['words'] += len(words)

# ==================== 信息熵计算 ====================
def calculate_entropy(counter, total_count):
    """计算标准化信息熵"""
    entropy = 0.0
    smoothing = 1e-10  # 平滑系数
    
    for count in counter.values():
        prob = (count + smoothing) / (total_count + smoothing * len(counter))
        entropy -= prob * math.log2(prob)
        
    return entropy

# ==================== 可视化模块 ====================
def create_plots(analyzer):
    """创建可视化图表"""
    fig = plt.figure(figsize=(14, 10))
    
    # 字符分布
    ax1 = plt.subplot(221)
    chars, counts = zip(*analyzer.char_stats.most_common(20))
    ax1.bar(chars, counts)
    ax1.set_title(f"Character Distribution (Entropy: {calculate_entropy(analyzer.char_stats, analyzer.counts['chars']):.2f} bits)")
    
    # 单词分布
    ax2 = plt.subplot(222)
    words, w_counts = zip(*analyzer.word_stats.most_common(20))
    ax2.barh(words[::-1], w_counts[::-1])
    ax2.set_title(f"Word Distribution (Entropy: {calculate_entropy(analyzer.word_stats, analyzer.counts['words']):.2f} bits)")
    
    # 字母Zipf定律
    ax3 = plt.subplot(223)
    char_freq = sorted(analyzer.char_stats.values(), reverse=True)
    ax3.loglog(np.arange(1, len(char_freq)+1), char_freq, 'o')
    ax3.set_title("Character Zipf's Law")
    
    # 单词Zipf定律
    ax4 = plt.subplot(224)
    word_freq = sorted(analyzer.word_stats.values(), reverse=True)
    ax4.loglog(np.arange(1, len(word_freq)+1), word_freq, 'o')
    ax4.set_title("Word Zipf's Law")
    
    plt.tight_layout()
    plt.savefig('analysis_result.png', dpi=150)
    plt.show()

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 初始化组件
    loader = ShakespeareLoader()
    analyzer = LinguisticAnalyzer()
    
    # 处理数据
    print("Processing texts...")
    for text in tqdm(loader.stream_text(), total=len(SHAKESPEARE_WORKS)):
        analyzer.process(text)
    
    # 输出统计结果
    print("\n=== Analysis Results ===")
    print(f"Processed {analyzer.counts['files']} files")
    print(f"Total characters: {analyzer.counts['chars']}")
    print(f"Total words: {analyzer.counts['words']}")
    print(f"Character entropy: {calculate_entropy(analyzer.char_stats, analyzer.counts['chars']):.2f} bits/char")
    print(f"Word entropy: {calculate_entropy(analyzer.word_stats, analyzer.counts['words']):.2f} bits/word")
    
    # 生成图表
    create_plots(analyzer)