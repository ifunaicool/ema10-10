#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模拟强信号股票分析演示系统
基于真实策略逻辑，生成模拟数据展示完整的强信号股票筛选和图表生成流程
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging
import warnings

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('demo_signal_analysis.log', encoding='utf-8', mode='a')
    ]
)
logger = logging.getLogger(__name__)


class DemoSignalAnalyzer:
    """演示用强信号分析器"""

    def __init__(self, ma_period=10, smooth_window=10):
        """初始化分析器"""
        self.ma_period = ma_period
        self.smooth_window = smooth_window
        self.analysis_results = []

        # 策略参数
        self.stop_loss_pct = 10.0
        self.take_profit_pct = 20.0

        # 模拟股票样本
        self.demo_stocks = [
            {'code': '600895', 'name': '张江高科', 'base_price': 15.50, 'trend': 'strong_up'},
            {'code': '600519', 'name': '贵州茅台', 'base_price': 1850.00, 'trend': 'moderate_up'},
            {'code': '000858', 'name': '五粮液', 'base_price': 160.00, 'trend': 'strong_up'},
            {'code': '601318', 'name': '中国平安', 'base_price': 45.00, 'trend': 'turning_up'},
            {'code': '601398', 'name': '工商银行', 'base_price': 5.20, 'trend': 'steady_up'},
            {'code': '002594', 'name': '比亚迪', 'base_price': 220.00, 'trend': 'strong_up'},
            {'code': '600276', 'name': '恒瑞医药', 'base_price': 48.00, 'trend': 'turning_up'},
            {'code': '000333', 'name': '美的集团', 'base_price': 65.00, 'trend': 'moderate_up'},
            {'code': '601888', 'name': '中国中免', 'base_price': 95.00, 'trend': 'strong_signal'},
            {'code': '300750', 'name': '宁德时代', 'base_price': 180.00, 'trend': 'strong_signal'},
            {'code': '600036', 'name': '招商银行', 'base_price': 32.00, 'trend': 'turning_up'},
            {'code': '002415', 'name': '海康威视', 'base_price': 38.00, 'trend': 'strong_up'},
            {'code': '601012', 'name': '隆基绿能', 'base_price': 28.00, 'trend': 'strong_signal'},
            {'code': '000651', 'name': '格力电器', 'base_price': 42.00, 'trend': 'turning_up'},
            {'code': '600887', 'name': '伊利股份', 'base_price': 32.50, 'trend': 'moderate_up'},
            {'code': '601899', 'name': '紫金矿业', 'base_price': 16.50, 'trend': 'strong_up'},
            {'code': '600030', 'name': '中信证券', 'base_price': 22.00, 'trend': 'turning_up'},
            {'code': '002142', 'name': '宁波银行', 'base_price': 25.00, 'trend': 'strong_signal'},
            {'code': '601919', 'name': '中远海控', 'base_price': 14.50, 'trend': 'strong_up'},
            {'code': '000725', 'name': '京东方A', 'base_price': 4.20, 'trend': 'strong_signal'}
        ]

    def generate_simulated_data(self, stock_info, days=252):
        """生成模拟股票数据"""
        base_price = stock_info['base_price']
        trend = stock_info['trend']

        np.random.seed(hash(stock_info['code']) % 1000)

        # 生成日期序列
        dates = pd.date_range(end=datetime.now(), periods=days, freq='B')

        # 根据趋势生成价格变化
        if trend == 'strong_up':
            # 强势上涨趋势
            daily_returns = np.random.normal(0.0015, 0.025, days)
            daily_returns[-20:] = np.random.normal(0.0025, 0.018, 20)  # 最后20天加速上涨
        elif trend == 'moderate_up':
            # 温和上涨趋势
            daily_returns = np.random.normal(0.0008, 0.020, days)
        elif trend == 'turning_up':
            # 转折向上（先跌后涨）
            daily_returns = np.concatenate([
                np.random.normal(-0.001, 0.025, days//2),
                np.random.normal(0.002, 0.022, days - days//2)
            ])
        elif trend == 'strong_signal':
            # 强信号特征（近期明显拐点）
            daily_returns = np.random.normal(0.0003, 0.028, days)
            daily_returns[-15:] = np.random.normal(0.0035, 0.015, 15)  # 最近15天强势
        else:
            # 稳定上涨
            daily_returns = np.random.normal(0.0005, 0.018, days)

        # 计算价格
        prices = base_price * (1 + daily_returns).cumprod()

        # 确保价格为正
        prices = np.maximum(prices, base_price * 0.7)

        # 生成OHLC数据
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            high = close * (1 + abs(np.random.normal(0, 0.015)))
            low = close * (1 - abs(np.random.normal(0, 0.015)))
            open_price = prices[i-1] if i > 0 else close
            high = max(high, open_price, close)
            low = min(low, open_price, close)

            data.append({
                '日期': date,
                '开盘': open_price,
                '最高': high,
                '最低': low,
                '收盘': close,
                '成交量': np.random.randint(100000, 5000000)
            })

        df = pd.DataFrame(data)
        return df

    def calculate_indicators(self, df):
        """计算技术指标"""
        # 计算基础MA10
        df[f'MA{self.ma_period}'] = df['收盘'].rolling(window=self.ma_period).mean()

        # 对MA10进行二次平滑
        df[f'SmoothMA{self.ma_period}'] = df[f'MA{self.ma_period}'].rolling(
            window=self.smooth_window, min_periods=1
        ).mean()

        # 计算斜率
        df[f'SmoothMA{self.ma_period}_slope'] = df[f'SmoothMA{self.ma_period}'].diff()

        # 计算斜率变化幅度
        df[f'SmoothMA{self.ma_period}_slope_change'] = df[f'SmoothMA{self.ma_period}_slope'].abs()

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

    def analyze_stock(self, stock_info):
        """深度分析单只股票"""
        try:
            # 生成模拟数据
            df = self.generate_simulated_data(stock_info, days=252)
            df_signals = self.generate_signals(df.copy())

            # 提取最新信号
            latest_data = df_signals.iloc[-1]
            latest_signal = latest_data['signal']

            # 统计历史信号
            buy_signals = df_signals[df_signals['signal'] == 'BUY']
            sell_signals = df_signals[df_signals['signal'] == 'SELL']

            # 计算信号质量指标
            signal_quality = 0.0
            if latest_signal == 'BUY' and len(buy_signals) > 0:
                latest_buy_signal = buy_signals.iloc[-1]
                signal_quality = latest_buy_signal['signal_strength'] * 1000
            elif len(buy_signals) > 0:
                # 使用最近的历史买入信号
                recent_buy_signal = buy_signals.iloc[-1]
                signal_quality = recent_buy_signal['signal_strength'] * 1000 * 0.8

            # 计算趋势强度
            trend_strength = abs(latest_data[f'SmoothMA{self.ma_period}_slope']) * 1000

            result = {
                '股票代码': stock_info['code'],
                '股票名称': stock_info['name'],
                '当前价格': latest_data['收盘'],
                '最新信号': latest_signal if pd.notna(latest_signal) else '无',
                '最新斜率': latest_data[f'SmoothMA{self.ma_period}_slope'],
                'SmoothMA10': latest_data[f'SmoothMA{self.ma_period}'],
                'MA10': latest_data[f'MA{self.ma_period}'],
                '历史买入次数': len(buy_signals),
                '历史卖出次数': len(sell_signals),
                '数据长度': len(df),
                '最新日期': latest_data['日期'],
                '信号质量': signal_quality,
                '趋势强度': trend_strength
            }

            # 如果有最新买入信号，记录详细信息
            if latest_signal == 'BUY':
                result['信号强度'] = latest_data['signal_strength']
                result['信号日期'] = latest_data['日期']

            return result, df_signals

        except Exception as e:
            logger.error(f"分析股票 {stock_info['code']} 失败: {e}")
            return None, None

    def create_comprehensive_chart(self, stock_info, df_signals, rank=None):
        """创建综合性图表"""
        if df_signals is None or len(df_signals) == 0:
            return None

        try:
            # 从不同结构中提取股票信息
            if isinstance(stock_info, dict):
                if 'code' in stock_info:
                    stock_code = stock_info['code']
                    stock_name = stock_info['name']
                else:
                    stock_code = stock_info['股票代码']
                    stock_name = stock_info['股票名称']
            else:
                logger.error(f"stock_info 类型错误: {type(stock_info)}")
                return None
            smooth_ma_col = f'SmoothMA{self.ma_period}'

            # 创建子图
            fig = make_subplots(
                rows=4, cols=1,
                row_heights=[0.4, 0.2, 0.2, 0.2],
                subplot_titles=(
                    f'#{stock_name} ({stock_code}) - 价格与MA趋势',
                    '成交量变化',
                    '斜率趋势分析',
                    '信号强度评估'
                ),
                vertical_spacing=0.08
            )

            # 主图：K线图
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
                            size=20,
                            color='#00e676',
                            line=dict(color='white', width=2)
                        ),
                        text=[f"买入<br>价格: {price:.2f}<br>强度: {strength:.4f}<br>日期: {date.strftime('%Y-%m-%d')}"
                              for price, strength, date in zip(buy_signals['收盘'], buy_signals['signal_strength'], buy_signals['日期'])],
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
                            size=20,
                            color='#ff5252',
                            line=dict(color='white', width=2)
                        ),
                        text=[f"卖出<br>价格: {price:.2f}<br>日期: {date.strftime('%Y-%m-%d')}"
                              for price, date in zip(sell_signals['收盘'], sell_signals['日期'])],
                        hovertemplate='%{text}<extra></extra>'
                    ),
                    row=1, col=1
                )

            # 成交量图
            colors = ['#ff5252' if close >= open_ else '#00e676'
                     for close, open_ in zip(df_signals['收盘'], df_signals['开盘'])]
            fig.add_trace(
                go.Bar(
                    x=df_signals['日期'],
                    y=df_signals['成交量'],
                    name='成交量',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )

            # 斜率图
            fig.add_trace(
                go.Scatter(
                    x=df_signals['日期'],
                    y=df_signals[f'{smooth_ma_col}_slope'],
                    mode='lines',
                    name='斜率趋势',
                    line=dict(color='#7c4dff', width=2)
                ),
                row=3, col=1
            )

            # 斜率零轴
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

            # 斜率信号点
            if len(buy_signals) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals['日期'],
                        y=buy_signals[f'{smooth_ma_col}_slope'],
                        mode='markers',
                        name='买入拐点',
                        marker=dict(
                            symbol='star',
                            size=12,
                            color='#00e676'
                        ),
                        showlegend=False
                    ),
                    row=3, col=1
                )

            # 信号强度图
            df_signals_filtered = df_signals[df_signals['signal_strength'].notna()]
            if len(df_signals_filtered) > 0:
                fig.add_trace(
                    go.Bar(
                        x=df_signals_filtered['日期'],
                        y=df_signals_filtered['signal_strength'] * 100,
                        name='信号强度',
                        marker_color=['#00e676' if sig == 'BUY' else '#ff5252'
                                   for sig in df_signals_filtered['signal']],
                        opacity=0.8
                    ),
                    row=4, col=1
                )

            # 更新布局
            title_prefix = f"TOP{rank} - " if rank else ""
            fig.update_layout(
                template='plotly_dark',
                height=1200,
                showlegend=True,
                hovermode='x unified',
                title=dict(
                    text=f"{title_prefix}{stock_name} ({stock_code}) - MA{self.ma_period}平滑策略深度分析",
                    x=0.5,
                    xanchor='center',
                    font=dict(size=18, color='white')
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            # 更新坐标轴
            fig.update_xaxes(title_text="日期", gridcolor='rgba(128,128,128,0.2)', row=4, col=1)
            fig.update_yaxes(title_text="价格", gridcolor='rgba(128,128,128,0.2)', row=1, col=1)
            fig.update_yaxes(title_text="成交量", gridcolor='rgba(128,128,128,0.2)', row=2, col=1)
            fig.update_yaxes(title_text="斜率", gridcolor='rgba(128,128,128,0.2)', row=3, col=1)
            fig.update_yaxes(title_text="信号强度(x100)", gridcolor='rgba(128,128,128,0.2)', row=4, col=1)

            return fig

        except Exception as e:
            logger.error(f"创建图表失败 {stock_info['code']}: {e}")
            return None

    def analyze_market_sample(self):
        """分析市场样本股票"""
        logger.info("开始分析市场样本股票...")

        analysis_results = []
        stock_data_cache = {}

        for i, stock_info in enumerate(self.demo_stocks):
            logger.info(f"正在分析股票 {i+1}/{len(self.demo_stocks)}: {stock_info['code']}")

            result, df_signals = self.analyze_stock(stock_info)
            if result and df_signals is not None:
                analysis_results.append(result)
                stock_data_cache[stock_info['code']] = df_signals
                logger.info(f"✓ {stock_info['code']} ({stock_info['name']}) 分析完成")
            else:
                logger.warning(f"✗ {stock_info['code']} 分析失败")

        # 按信号质量排序
        if analysis_results:
            df_results = pd.DataFrame(analysis_results)

            # 计算综合得分
            df_results['综合得分'] = (
                df_results['信号质量'] * 0.4 +
                df_results['趋势强度'] * 0.3 +
                df_results['历史买入次数'] * 10 +
                (df_results['最新信号'] == 'BUY').astype(int) * 50
            )

            # 按综合得分排序
            df_results = df_results.sort_values('综合得分', ascending=False)
            self.analysis_results = df_results.head(20).to_dict('records')

            logger.info(f"✓ 分析完成，推荐前20只股票")

        return self.analysis_results, stock_data_cache

    def create_top_charts(self, stock_data_cache):
        """为前20只股票创建详细图表"""
        logger.info("开始创建详细分析图表...")

        if not self.analysis_results:
            logger.warning("没有结果可创建图表")
            return []

        created_charts = []
        for rank, stock_info in enumerate(self.analysis_results):
            stock_code = stock_info['股票代码']

            if stock_code in stock_data_cache:
                fig = self.create_comprehensive_chart(
                    stock_info,
                    stock_data_cache[stock_code],
                    rank=rank+1
                )

                if fig:
                    rank_str = f"TOP{rank+1:02d}"
                    filename = f"{rank_str}_{stock_code}_强信号分析.html"
                    fig.write_html(filename)
                    created_charts.append(filename)
                    logger.info(f"✓ 已创建图表: {filename} (综合得分: {stock_info.get('综合得分', 0):.2f})")

        return created_charts

    def generate_final_report(self):
        """生成最终分析报告"""
        if not self.analysis_results:
            print("没有结果可生成报告")
            return

        print(f"\n{'='*100}")
        print(f"{' '*30}🔥 强信号股票分析报告 🔥")
        print(f"{' '*20}基于MA{self.ma_period}平滑策略的智能筛选")
        print(f"{'='*100}")

        # 策略概况
        print(f"\n【策略概况】")
        print(f"• 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"• 策略参数: MA{self.ma_period} + {self.smooth_window}日平滑窗口")
        print(f"• 历史数据: 最近1年（252个交易日）")
        print(f"• 样本股票: {len(self.demo_stocks)}只")
        print(f"• 推荐股票: {len(self.analysis_results)}只")

        # 详细分析结果
        print(f"\n【📊 TOP 20 强信号股票排行榜 📊】")
        print(f"{'排名':<6} {'股票代码':<10} {'股票名称':<12} {'当前价格':<10} {'最新信号':<10} {'综合得分':<12} {'信号强度':<15}")
        print("-" * 100)

        for i, stock_info in enumerate(self.analysis_results):
            rank = i + 1
            signal_strength = stock_info.get('信号强度', 0)
            signal_strength_str = f"{signal_strength:.4f}" if signal_strength > 0 else "历史信号"
            print(f"{rank:<6} {stock_info['股票代码']:<10} {stock_info['股票名称']:<12} "
                  f"{stock_info['当前价格']:<10.2f} {stock_info['最新信号']:<10} "
                  f"{stock_info['综合得分']:<12.2f} {signal_strength_str:<15}")

        # 深度分析
        print(f"\n【🔍 深度分析】")

        # 信号分布
        signal_counts = {}
        for result in self.analysis_results:
            signal = result['最新信号']
            signal_counts[signal] = signal_counts.get(signal, 0) + 1

        print(f"• 强力买入信号: {signal_counts.get('BUY', 0)}只")
        print(f"• 潜力关注股票: {signal_counts.get('无', 0)}只")

        # 价格分析
        prices = [r['当前价格'] for r in self.analysis_results]
        print(f"• 价格区间: {min(prices):.2f}元 - {max(prices):.2f}元")
        print(f"• 平均价格: {np.mean(prices):.2f}元")

        # 综合得分分析
        scores = [r['综合得分'] for r in self.analysis_results]
        print(f"• 最高得分: {max(scores):.2f}")
        print(f"• 最低得分: {min(scores):.2f}")
        print(f"• 平均得分: {np.mean(scores):.2f}")

        # 投资建议
        print(f"\n【💡 投资建议】")

        if signal_counts.get('BUY', 0) > 0:
            buy_stocks = [r for r in self.analysis_results if r['最新信号'] == 'BUY']
            print(f"• 当前有 {len(buy_stocks)} 只股票出现强力买入信号")
            print(f"• 建议重点关注排名前3的股票：")
            for i, stock in enumerate(buy_stocks[:3], 1):
                print(f"  {i}. {stock['股票名称']} ({stock['股票代码']}) - 综合得分: {stock['综合得分']:.2f}")
            print(f"• 设置止损位：当前价格下浮{self.stop_loss_pct}%")
            print(f"• 设置止盈位：当前价格上浮{self.take_profit_pct}%")
        else:
            print(f"• 建议关注综合得分排名前5的潜力股票")
            print(f"• 等待MA10斜率转向，捕捉更好的入场时机")

        print(f"• 密切关注MA10斜率变化和信号强度")
        print(f"• 结合成交量分析确认信号有效性")
        print(f"• 建议分批建仓，控制单股仓位不超过总资金10%")

        print(f"\n【📈 图表说明】")
        print(f"• 已为TOP 20股票生成详细的HTML图表")
        print(f"• 图表包含：K线走势、成交量、斜率趋势、信号强度")
        print(f"• 图表文件命名：TOP01_股票代码_强信号分析.html")
        print(f"• 使用浏览器打开对应的HTML文件查看详细分析")

        print(f"\n【⚠️ 风险提示】")
        print(f"• 本分析基于模拟数据，仅供策略演示使用")
        print(f"• 实际投资需结合真实市场数据和基本面分析")
        print(f"• 历史表现不代表未来收益，投资有风险")
        print(f"• 建议在专业投资顾问指导下进行投资决策")

        print(f"\n{'='*100}\n")

    def save_results(self):
        """保存分析结果"""
        if not self.analysis_results:
            return

        # 转换为DataFrame
        df_results = pd.DataFrame(self.analysis_results)

        # 保存完整结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        df_results.to_csv(f'TOP20_强信号分析_{timestamp}.csv',
                         index=False, encoding='utf-8-sig')
        logger.info(f"✓ TOP20分析结果已保存: TOP20_强信号分析_{timestamp}.csv")


def main():
    """主函数"""
    print(f"\n{'='*100}")
    print(f"{' '*35}🚀 强信号股票分析演示 🚀")
    print(f"{' '*30}基于MA10平滑策略的智能筛选")
    print(f"{' '*25}（模拟数据演示版本）")
    print(f"{'='*100}\n")

    # 创建分析器
    analyzer = DemoSignalAnalyzer(
        ma_period=10,
        smooth_window=10
    )

    # 分析市场样本
    results, stock_data_cache = analyzer.analyze_market_sample()

    if results:
        # 创建详细图表
        created_charts = analyzer.create_top_charts(stock_data_cache)
        logger.info(f"✓ 共创建 {len(created_charts)} 个详细分析图表")

        # 保存结果
        analyzer.save_results()

        # 生成最终报告
        analyzer.generate_final_report()
    else:
        print("没有找到有效的分析结果")


if __name__ == "__main__":
    main()
