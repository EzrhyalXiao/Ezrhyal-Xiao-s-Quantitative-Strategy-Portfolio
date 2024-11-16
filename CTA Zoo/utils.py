import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class PnLAnalysis:
    def __init__(self, df: pd.DataFrame, drawdown_threshold: float = 0):
        """
        初始化分析类

        :param df: 包含策略净值的DataFrame，列是策略名称，行是日期或时间戳
        :param drawdown_threshold: 回撤阈值，默认0.05表示5%的回撤
        """
        self.df = df
        self.drawdown_threshold = drawdown_threshold
        self.drawdowns = self._calculate_drawdowns(df)

    def _calculate_drawdowns(self, df):
        """
        计算每个策略的回撤
        :param df: 策略净值数据框
        :return: 每个策略的回撤数据框
        """
        drawdowns = pd.DataFrame(index=df.index, columns=df.columns)

        for col in df.columns:
            peak = df[col].cummax()  # 累计最大值
            drawdown = (df[col] - peak) / peak  # 计算回撤
            drawdowns[col] = drawdown

        return drawdowns

    def _generate_heatmap(self, data, title, filename):
        """
        生成热力图
        :param data: 数据框
        :param title: 热力图的标题
        :param filename: 保存热力图的文件名
        """
        plt.figure(figsize=(10, 8), dpi=200)  # 设置显示时的dpi
        sns.heatmap(data, annot=True, cmap='PuBuGn', fmt='.2f', cbar=True)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(filename,dpi=200)
        plt.show()

    def strategy_returns_correlation(self):
        """
        计算并展示每个策略收益的相关性热力图
        :return: None
        """
        # 计算收益
        returns = self.df.pct_change().dropna()
        correlation = returns.corr()
        self._generate_heatmap(correlation, "Strategy Returns Correlation", "strategy_returns_correlation.png")

    def strategy_drawdowns_correlation(self):
        """
        计算并展示每个策略回撤的相关性热力图
        :return: None
        """
        correlation = self.drawdowns.corr()
        self._generate_heatmap(correlation, "Strategy Drawdowns Correlation", "strategy_drawdowns_correlation.png")

