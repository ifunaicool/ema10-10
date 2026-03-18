#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全市场MA10平滑策略监控系统 - 优化版
整合get-data-v3.py的高质量数据获取模块，提供更稳定的全市场监控
"""

import akshare as ak
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
import requests
import threading

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('market_scanner_optimized.log', encoding='utf-8', mode='a')
    ]
)
logger = logging.getLogger(__name__)


class StockDataConfig:
    """股票数据配置类"""
    def __init__(self):
        self.request_interval = 0.8      # 请求间隔
        self.max_retries = 3               # 最大重试次数
        self.batch_interval = 3.0        # 批次间隔
        self.batch_size = 15               # 批次大小
        self.timeout = 30                  # 请求超时时间
        self.max_workers = 5               # 最大线程数
        self.retry_delay_factor = 1.5    # 重试延迟因子


class StockDataFetcher:
    """股票数据获取器 - 优化版本"""

    def __init__(self, config: StockDataConfig = None):
        """初始化数据获取器"""
        self.config = config or StockDataConfig()
        self.request_count = 0
        self.request_times = []
        self.failed_requests = 0
        self.successful_requests = 0

        # 创建会话
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Connection': 'keep-alive'
        })
        self.session.timeout = self.config.timeout
        self.request_lock = threading.RLock()

        logger.info("StockDataFetcher初始化完成")

    def get_stock_data(self, stock_code: str, years: int = 1) -> Optional[pd.DataFrame]:
        """
        获取股票历史数据
        :param stock_code: 股票代码
        :param years: 历史数据年数
        :return: DataFrame or None
        """
        max_retries = self.config.max_retries
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                with self.request_lock:
                    # 控制请求频率
                    if len(self.request_times) > 0:
                        last_request_time = self.request_times[-1]
                        time_since_last = time.time() - last_request_time
                        if time_since_last < self.config.request_interval:
                            time.sleep(self.config.request_interval - time_since_last)

                    end_date = datetime.now().strftime('%Y%m%d')
                    start_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y%m%d')

                    df = ak.stock_zh_a_hist(
                        symbol=stock_code,
                        period='daily',
                        start_date=start_date,
                        end_date=end_date,
                        adjust='qfq'
                    )

                    self.request_count += 1
                    self.request_times.append(time.time())
                    self.successful_requests += 1

                    if df is None or len(df) == 0:
                        return None

                    # 统一列名格式
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.rename(columns={'date': '日期', 'open': '开盘', 'close': '收盘',
                                               'high': '最高', 'low': '最低', 'volume': '成交量',
                                               'amount': '成交额'})

                    df = df.sort_values('日期').reset_index(drop=True)
                    return df

            except Exception as e:
                self.failed_requests += 1
                if attempt < max_retries - 1:
                    logger.warning(f"获取股票 {stock_code} 数据失败 (第{attempt+1}次尝试): {e}")
                    time.sleep(retry_delay)
                    retry_delay *= self.config.retry_delay_factor
                else:
                    logger.error(f"获取股票 {stock_code} 数据最终失败: {e}")
                    return None

        return None


class MarketScannerOptimized:
    """优化版全市场MA10平滑策略扫描器"""

    def __init__(self, ma_period=10, smooth_window=10, years=1):
        """初始化扫描器"""
        self.ma_period = ma_period
        self.smooth_window = smooth_window
        self.years = years
        self.stock_list = []
        self.results = []

        # 策略参数
        self.stop_loss_pct = 10.0
        self.take_profit_pct = 20.0

        # 数据获取器
        self.data_fetcher = StockDataFetcher()

        # 统计信息
        self.total_stocks = 0
        self.success_count = 0
        self.failed_count = 0
        self.st_excluded_count = 0

        # 数据缓存
        self.stock_data_cache = {}

    def get_all_stocks(self):
        """获取沪深所有A股股票列表，剔除ST和*ST股票"""
        logger.info("正在获取沪深A股股票列表...")

        try:
            # 获取A股实时行情数据
            df = ak.stock_zh_a_spot_em()

            logger.info(f"获取到 {len(df)} 只股票")

            # 筛选条件
            conditions = [
                ~df['名称'].str.contains(r'ST|\*ST|退', na=False, regex=True),  # 剔除ST、*ST、退市股票
                df['最新价'] > 0,  # 价格大于0
                df['最新价'] < 1000,  # 价格合理范围
            ]

            # 应用筛选条件
            df_filtered = df[conditions[0] & conditions[1] & conditions[2]]

            self.stock_list = df_filtered['代码'].tolist()
            self.st_excluded_count = len(df) - len(df_filtered)
            self.total_stocks = len(self.stock_list)

            logger.info(f"✓ 筛选后有效股票: {self.total_stocks} 只")
            logger.info(f"✓ 剔除ST/*ST股票: {self.st_excluded_count} 只")

            return self.stock_list

        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return []

    def calculate_indicators(self, df):
        """计算技术指标 - 10日窗口平滑MA10"""
        # 计算基础MA10
        df[f'MA{self.ma_period}'] = df['收盘'].rolling(window=self.ma_period).mean()

        # 对MA10进行二次平滑（10日移动平均）
        df[f'SmoothMA{self.ma_period}'] = df[f'MA{self.ma_period}'].rolling(
            window=self.smooth_window, min_periods=1
        ).mean()

        # 计算斜率
        df[f'SmoothMA{self.ma_period}_slope'] = df[f'SmoothMA{self.ma_period}'].diff()

        # 计算斜率变化幅度
        df[f'SmoothMA{self.ma_period}_slope_change'] = df[f'SmoothMA{self.ma_period}_slope'].abs()

        # 计算原始MA10的斜率
        df[f'MA{self.ma_period}_slope'] = df[f'MA{self.ma_period}'].diff()

        return df

    def generate_signals(self, df):
        """生成交易信号"""
        df = self.calculate_indicators(df)

        df['signal'] = None
        df['signal_type'] = None
        df['signal_strength'] = None

        smooth_ma_col = f'SmoothMA{self.ma_period}'

        for i in range(self.ma_period + self.smooth_window, len(df)):
            current_slope = df.iloc[i][f'{smooth_ma_col}_slope']
            prev_slope = df.iloc[i-1][f'{smooth_ma_col}_slope']
            slope_change = df.iloc[i][f'{smooth_ma_col}_slope_change']

            # 斜率由负转正且信号强度达标，生成买入信号
            if prev_slope <= 0 and current_slope > 0 and slope_change > 0.001:
                df.loc[df.index[i], 'signal'] = 'BUY'
                df.loc[df.index[i], 'signal_type'] = 1
                df.loc[df.index[i], 'signal_strength'] = slope_change

            # 斜率由正转负，生成卖出信号
            elif prev_slope >= 0 and current_slope < 0:
                df.loc[df.index[i], 'signal'] = 'SELL'
                df.loc[df.index[i], 'signal_type'] = -1
                df.loc[df.index[i], 'signal_strength'] = slope_change

        return df

    def analyze_stock(self, stock_code, cache_data=False):
        """分析单只股票"""
        try:
            # 使用优化的数据获取器
            df = self.data_fetcher.get_stock_data(stock_code, self.years)
            if df is None:
                return None, None

            # 确保有足够的数据
            min_periods = self.ma_period + self.smooth_window + 10
            if len(df) < min_periods:
                return None, None

            df = self.calculate_indicators(df)
            df_signals = self.generate_signals(df.copy())

            # 缓存数据用于图表生成
            if cache_data:
                self.stock_data_cache[stock_code] = df_signals

            # 提取最新信号
            latest_data = df_signals.iloc[-1]
            latest_signal = latest_data['signal']
            latest_slope = latest_data[f'SmoothMA{self.ma_period}_slope']

            # 统计历史信号
            buy_signals = df_signals[df_signals['signal'] == 'BUY']
            sell_signals = df_signals[df_signals['signal'] == 'SELL']

            # 获取股票名称
            try:
                stock_name = ak.stock_individual_info_em(symbol=stock_code).loc[
                    '股票简称', 'value'
                ]
            except:
                stock_name = stock_code

            result = {
                '股票代码': stock_code,
                '股票名称': stock_name,
                '当前价格': latest_data['收盘'],
                '最新信号': latest_signal if pd.notna(latest_signal) else '无',
                '最新斜率': latest_slope,
                'SmoothMA10': latest_data[f'SmoothMA{self.ma_period}'],
                'MA10': latest_data[f'MA{self.ma_period}'],
                '历史买入次数': len(buy_signals),
                '历史卖出次数': len(sell_signals),
                '数据长度': len(df),
                '最新日期': latest_data['日期']
            }

            # 如果有最新买入信号，记录详细信息
            if latest_signal == 'BUY':
                result['信号强度'] = latest_data['signal_strength']
                result['信号日期'] = latest_data['日期']

            return result, df_signals

        except Exception as e:
            return None, None

    def create_trading_chart(self, stock_code, df_signals):
        """为单只股票创建买卖信息图表"""
        if df_signals is None or len(df_signals) == 0:
            return None

        try:
            # 获取股票名称
            try:
                stock_name = ak.stock_individual_info_em(symbol=stock_code).loc[
                    '股票简称', 'value'
                ]
            except:
                stock_name = stock_code

            smooth_ma_col = f'SmoothMA{self.ma_period}'

            # 创建子图
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                subplot_titles=(f'{stock_name} ({stock_code}) - MA{self.ma_period}平滑策略', '斜率变化'),
                vertical_spacing=0.05
            )

            # K线图
            fig.add_trace(
                go.Candlestick(
                    x=df_signals['日期'],
                    open=df_signals['开盘'],
                    high=df_signals['最高'],
                    low=df_signals['最低'],
                    close=df_signals['收盘'],
                    name='K线',
                    increasing_line_color='#ff5252',
                    decreasing_line_color='#00e676'
                ),
                row=1, col=1
            )

            # 原始MA10
            fig.add_trace(
                go.Scatter(
                    x=df_signals['日期'],
                    y=df_signals[f'MA{self.ma_period}'],
                    mode='lines',
                    name=f'原始MA{self.ma_period}',
                    line=dict(color='#ffa726', width=1, dash='dot'),
                    opacity=0.6
                ),
                row=1, col=1
            )

            # 平滑MA10
            fig.add_trace(
                go.Scatter(
                    x=df_signals['日期'],
                    y=df_signals[smooth_ma_col],
                    mode='lines',
                    name=f'平滑MA{self.ma_period}',
                    line=dict(color='#2979ff', width=2)
                ),
                row=1, col=1
            )

            # 买卖信号
            buy_signals = df_signals[df_signals['signal'] == 'BUY']
            sell_signals = df_signals[df_signals['signal'] == 'SELL']

            if len(buy_signals) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals['日期'],
                        y=buy_signals['收盘'],
                        mode='markers',
                        name='买入信号',
                        marker=dict(
                            symbol='triangle-up',
                            size=15,
                            color='#00e676',
                            line=dict(color='white', width=2)
                        ),
                        text=[f"买入<br>价格: {price:.2f}<br>强度: {strength:.4f}"
                              for price, strength in zip(buy_signals['收盘'], buy_signals['signal_strength'])],
                        hovertemplate='%{text}<extra></extra>'
                    ),
                    row=1, col=1
                )

            if len(sell_signals) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=sell_signals['日期'],
                        y=sell_signals['收盘'],
                        mode='markers',
                        name='卖出信号',
                        marker=dict(
                            symbol='triangle-down',
                            size=15,
                            color='#ff5252',
                            line=dict(color='white', width=2)
                        ),
                        text=[f"卖出<br>价格: {price:.2f}"
                              for price in sell_signals['收盘']],
                        hovertemplate='%{text}<extra></extra>'
                    ),
                    row=1, col=1
                )

            # 斜率图
            fig.add_trace(
                go.Scatter(
                    x=df_signals['日期'],
                    y=df_signals[f'{smooth_ma_col}_slope'],
                    mode='lines',
                    name='斜率',
                    line=dict(color='#7c4dff', width=1.5)
                ),
                row=2, col=1
            )

            # 零轴
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

            # 更新布局
            fig.update_layout(
                template='plotly_dark',
                height=800,
                showlegend=True,
                hovermode='x unified',
                title_font=dict(size=16, color='white'),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            fig.update_xaxes(title_text="日期", gridcolor='rgba(128,128,128,0.2)', row=2, col=1)
            fig.update_yaxes(title_text="价格", gridcolor='rgba(128,128,128,0.2)', row=1, col=1)
            fig.update_yaxes(title_text="斜率", gridcolor='rgba(128,128,128,0.2)', row=2, col=1)

            return fig

        except Exception as e:
            logger.error(f"创建图表失败 {stock_code}: {e}")
            return None

    def scan_market(self, max_workers=5, sample_size=None, create_charts_for_signals=True):
        """全市场扫描"""
        logger.info("开始全市场扫描...")
        logger.info(f"策略参数: MA{self.ma_period} + {self.smooth_window}日平滑窗口")

        # 获取股票列表
        if not self.get_all_stocks():
            logger.error("无法获取股票列表")
            return

        # 如果指定采样数量
        if sample_size and sample_size < len(self.stock_list):
            self.stock_list = self.stock_list[:sample_size]
            logger.info(f"采样模式: 扫描前 {sample_size} 只股票")

        # 并发扫描
        start_time = time.time()
        self.results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_stock = {
                executor.submit(self.analyze_stock, stock, cache_data=create_charts_for_signals): stock
                for stock in self.stock_list
            }

            for i, future in enumerate(as_completed(future_to_stock)):
                stock_code = future_to_stock[future]

                try:
                    result, df_signals = future.result()
                    if result:
                        self.results.append(result)
                        self.success_count += 1
                    else:
                        self.failed_count += 1
                except Exception as e:
                    self.failed_count += 1

                # 进度显示
                if (i + 1) % 50 == 0:
                    elapsed = time.time() - start_time
                    success_rate = (self.success_count / (i + 1)) * 100 if (i + 1) > 0 else 0
                    logger.info(f"进度: {i+1}/{len(self.stock_list)} | "
                              f"成功: {self.success_count} | "
                              f"失败: {self.failed_count} | "
                              f"成功率: {success_rate:.1f}% | "
                              f"耗时: {elapsed:.1f}秒")

                # 请求间隔控制
                time.sleep(0.05)

        elapsed = time.time() - start_time
        logger.info(f"✓ 扫描完成! 总耗时: {elapsed:.1f}秒")
        logger.info(f"✓ 成功分析: {self.success_count} 只")
        logger.info(f"✓ 失败: {self.failed_count} 只")
        logger.info(f"✓ 数据获取成功率: {self.data_fetcher.successful_requests}/{self.data_fetcher.request_count}")

        # 为有信号的股票创建图表
        if create_charts_for_signals:
            self.create_signal_charts()

        # 保存结果
        self.save_results()

        return self.results

    def create_signal_charts(self):
        """为所有有信号的股票创建买卖信息图表"""
        logger.info("开始创建买卖信号图表...")

        df = pd.DataFrame(self.results)

        # 筛选有信号的股票
        df_with_signals = df[df['最新信号'] != '无']

        if len(df_with_signals) == 0:
            logger.info("没有发现买卖信号，跳过图表生成")
            return

        # 为每只有信号的股票创建图表
        chart_count = 0
        for stock_code in df_with_signals['股票代码']:
            if stock_code in self.stock_data_cache:
                fig = self.create_trading_chart(stock_code, self.stock_data_cache[stock_code])

                if fig:
                    filename = f"chart_{stock_code}_signals.html"
                    fig.write_html(filename)
                    chart_count += 1
                    logger.info(f"✓ 已创建图表: {filename}")

        logger.info(f"✓ 共创建 {chart_count} 个买卖信号图表")

    def save_results(self):
        """保存扫描结果"""
        if not self.results:
            logger.warning("没有结果需要保存")
            return

        # 转换为DataFrame
        df_results = pd.DataFrame(self.results)

        # 按最新信号分类保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存完整结果
        df_results.to_csv(f'market_scan_results_{timestamp}.csv',
                         index=False, encoding='utf-8-sig')
        logger.info(f"✓ 完整结果已保存: market_scan_results_{timestamp}.csv")

        # 保存有信号的股票
        df_with_signals = df_results[df_results['最新信号'] != '无']
        if len(df_with_signals) > 0:
            df_with_signals.to_csv(f'market_scan_signals_{timestamp}.csv',
                                  index=False, encoding='utf-8-sig')
            logger.info(f"✓ 有信号股票已保存: market_scan_signals_{timestamp}.csv "
                       f"(共{len(df_with_signals)}只)")

        # 保存买入信号股票
        df_buy_signals = df_results[df_results['最新信号'] == 'BUY']
        if len(df_buy_signals) > 0:
            df_buy_signals = df_buy_signals.sort_values('信号强度', ascending=False)
            df_buy_signals.to_csv(f'market_scan_buy_{timestamp}.csv',
                                 index=False, encoding='utf-8-sig')
            logger.info(f"✓ 买入信号股票已保存: market_scan_buy_{timestamp}.csv "
                       f"(共{len(df_buy_signals)}只)")

    def generate_summary_report(self):
        """生成扫描摘要报告"""
        if not self.results:
            logger.warning("没有结果可生成报告")
            return

        df = pd.DataFrame(self.results)

        print(f"\n{'='*80}")
        print(f"{' '*25}全市场扫描摘要报告")
        print(f"{'='*80}")

        # 基本统计
        print(f"\n【扫描概况】")
        print(f"• 扫描时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"• 策略参数: MA{self.ma_period} + {self.smooth_window}日平滑窗口")
        print(f"• 历史数据: 最近{self.years}年")
        print(f"• 扫描股票: {self.total_stocks}只")
        print(f"• 成功分析: {self.success_count}只")
        print(f"• 失败数量: {self.failed_count}只")
        print(f"• 剔除ST: {self.st_excluded_count}只")

        # 信号统计
        print(f"\n【信号分布】")
        signal_counts = df['最新信号'].value_counts()
        print(f"• 无信号: {signal_counts.get('无', 0)}只")
        print(f"• 买入信号: {signal_counts.get('BUY', 0)}只")
        print(f"• 卖出信号: {signal_counts.get('SELL', 0)}只")

        # 买入信号详情
        buy_stocks = df[df['最新信号'] == 'BUY']
        if len(buy_stocks) > 0:
            print(f"\n【买入信号详情】(共{len(buy_stocks)}只)")
            print(f"{'股票代码':<8} {'股票名称':<12} {'当前价格':<10} {'信号强度':<12} {'信号日期':<12}")
            print("-" * 70)

            buy_stocks_sorted = buy_stocks.sort_values('信号强度', ascending=False).head(20)
            for _, row in buy_stocks_sorted.iterrows():
                print(f"{row['股票代码']:<8} {row['股票名称']:<12} "
                      f"{row['当前价格']:<10.2f} {row['信号强度']:<12.6f} "
                      f"{str(row['信号日期'])[:10]:<12}")

        # 卖出信号详情
        sell_stocks = df[df['最新信号'] == 'SELL']
        if len(sell_stocks) > 0:
            print(f"\n【卖出信号详情】(共{len(sell_stocks)}只)")
            print(f"{'股票代码':<8} {'股票名称':<12} {'当前价格':<10} {'MA10':<12} {'当前斜率':<12}")
            print("-" * 70)

            sell_stocks_sorted = sell_stocks.sort_values('当前价格', ascending=False).head(20)
            for _, row in sell_stocks_sorted.iterrows():
                print(f"{row['股票代码']:<8} {row['股票名称']:<12} "
                      f"{row['当前价格']:<10.2f} {row['MA10']:<12.2f} "
                      f"{row['最新斜率']:<12.6f}")

        print(f"\n{'='*80}\n")


def main():
    """主函数"""
    print(f"\n{'='*80}")
    print(f"{' '*30}全市场MA10平滑策略监控 - 优化版")
    print(f"{'='*80}\n")

    # 创建扫描器
    scanner = MarketScannerOptimized(
        ma_period=10,
        smooth_window=10,
        years=1
    )

    # 全市场扫描
    # sample_size=None 表示全量扫描
    # create_charts_for_signals=True 表示为有信号的股票创建图表
    scanner.scan_market(max_workers=8, sample_size=None, create_charts_for_signals=True)

    # 生成报告
    scanner.generate_summary_report()


if __name__ == "__main__":
    main()
