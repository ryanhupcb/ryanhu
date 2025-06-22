"""
高级分析和可视化系统
提供深度数据分析、智能洞察生成和交互式可视化功能
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import json
import logging
from pathlib import Path

# 可视化库
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.collections import LineCollection
import seaborn as sns

# 科学计算
from scipy import stats, signal, optimize
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, t_SNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score

# 时间序列分析
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 网络分析
import networkx as nx
from networkx.algorithms import community
from pyvis.network import Network

# 自然语言处理
from textblob import TextBlob
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ==================== 数据分析器 ====================

@dataclass
class AnalysisResult:
    """分析结果"""
    analysis_type: str
    timestamp: datetime
    data: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class AdvancedDataAnalyzer:
    """高级数据分析器"""
    
    def __init__(self):
        self.analyzers = {
            'descriptive': DescriptiveAnalyzer(),
            'diagnostic': DiagnosticAnalyzer(),
            'predictive': PredictiveAnalyzer(),
            'prescriptive': PrescriptiveAnalyzer(),
            'exploratory': ExploratoryAnalyzer()
        }
        self.insight_generator = InsightGenerator()
        self.anomaly_detector = AdvancedAnomalyDetector()
        
    async def analyze(self, data: Union[pd.DataFrame, Dict, List], 
                     analysis_types: List[str] = None) -> Dict[str, AnalysisResult]:
        """执行综合分析"""
        if analysis_types is None:
            analysis_types = list(self.analyzers.keys())
            
        # 数据预处理
        processed_data = await self.preprocess_data(data)
        
        # 并行执行多种分析
        results = {}
        tasks = []
        
        for analysis_type in analysis_types:
            if analysis_type in self.analyzers:
                analyzer = self.analyzers[analysis_type]
                task = asyncio.create_task(
                    self.run_analysis(analyzer, processed_data, analysis_type)
                )
                tasks.append((analysis_type, task))
                
        # 等待所有分析完成
        for analysis_type, task in tasks:
            try:
                result = await task
                results[analysis_type] = result
            except Exception as e:
                logging.error(f"Analysis failed for {analysis_type}: {e}")
                
        # 生成综合洞察
        insights = await self.insight_generator.generate_insights(results)
        
        # 检测异常
        anomalies = await self.anomaly_detector.detect(processed_data)
        
        return {
            'analyses': results,
            'insights': insights,
            'anomalies': anomalies,
            'summary': await self.generate_summary(results, insights, anomalies)
        }
    
    async def preprocess_data(self, data: Any) -> pd.DataFrame:
        """预处理数据"""
        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
            
        # 处理缺失值
        df = df.fillna(df.mean(numeric_only=True))
        
        # 处理异常值
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower_bound, upper_bound)
            
        return df
    
    async def run_analysis(self, analyzer: Any, data: pd.DataFrame, 
                          analysis_type: str) -> AnalysisResult:
        """运行单个分析"""
        start_time = datetime.now()
        
        # 执行分析
        analysis_data = await analyzer.analyze(data)
        
        # 生成洞察
        insights = await analyzer.generate_insights(analysis_data)
        
        # 生成建议
        recommendations = await analyzer.generate_recommendations(analysis_data)
        
        # 计算置信度
        confidence = await analyzer.calculate_confidence(analysis_data)
        
        return AnalysisResult(
            analysis_type=analysis_type,
            timestamp=start_time,
            data=analysis_data,
            insights=insights,
            recommendations=recommendations,
            confidence=confidence,
            metadata={
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'data_shape': data.shape,
                'analyzer_version': getattr(analyzer, 'version', '1.0')
            }
        )

class DescriptiveAnalyzer:
    """描述性分析器"""
    
    async def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """执行描述性分析"""
        results = {}
        
        # 基本统计
        results['basic_stats'] = {
            'summary': data.describe().to_dict(),
            'shape': data.shape,
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict()
        }
        
        # 分布分析
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        results['distributions'] = {}
        
        for col in numeric_cols:
            results['distributions'][col] = {
                'mean': data[col].mean(),
                'median': data[col].median(),
                'mode': data[col].mode().tolist(),
                'std': data[col].std(),
                'skewness': data[col].skew(),
                'kurtosis': data[col].kurtosis(),
                'percentiles': {
                    f'p{p}': data[col].quantile(p/100) 
                    for p in [5, 25, 50, 75, 95]
                }
            }
            
        # 相关性分析
        if len(numeric_cols) > 1:
            results['correlations'] = data[numeric_cols].corr().to_dict()
            
        # 分类变量分析
        categorical_cols = data.select_dtypes(include=['object']).columns
        results['categorical'] = {}
        
        for col in categorical_cols:
            value_counts = data[col].value_counts()
            results['categorical'][col] = {
                'unique_values': data[col].nunique(),
                'top_values': value_counts.head(10).to_dict(),
                'frequency_distribution': (value_counts / len(data)).to_dict()
            }
            
        return results
    
    async def generate_insights(self, analysis_data: Dict[str, Any]) -> List[str]:
        """生成洞察"""
        insights = []
        
        # 分布洞察
        for col, dist in analysis_data.get('distributions', {}).items():
            if abs(dist['skewness']) > 1:
                direction = "right" if dist['skewness'] > 0 else "left"
                insights.append(f"{col} shows significant {direction} skewness ({dist['skewness']:.2f})")
                
            if dist['kurtosis'] > 3:
                insights.append(f"{col} has heavy tails (kurtosis: {dist['kurtosis']:.2f})")
                
        # 相关性洞察
        correlations = analysis_data.get('correlations', {})
        for col1, corr_dict in correlations.items():
            for col2, corr_value in corr_dict.items():
                if col1 < col2 and abs(corr_value) > 0.7:
                    strength = "strong" if abs(corr_value) > 0.8 else "moderate"
                    direction = "positive" if corr_value > 0 else "negative"
                    insights.append(f"{strength} {direction} correlation between {col1} and {col2} ({corr_value:.2f})")
                    
        return insights
    
    async def generate_recommendations(self, analysis_data: Dict[str, Any]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于分布的建议
        for col, dist in analysis_data.get('distributions', {}).items():
            if abs(dist['skewness']) > 1:
                recommendations.append(f"Consider transforming {col} to reduce skewness")
                
        # 基于缺失值的建议
        missing = analysis_data.get('basic_stats', {}).get('missing_values', {})
        for col, count in missing.items():
            if count > 0:
                recommendations.append(f"Address missing values in {col} ({count} missing)")
                
        return recommendations
    
    async def calculate_confidence(self, analysis_data: Dict[str, Any]) -> float:
        """计算置信度"""
        # 基于数据完整性和样本量计算置信度
        shape = analysis_data.get('basic_stats', {}).get('shape', (0, 0))
        n_samples = shape[0]
        
        if n_samples < 30:
            return 0.5
        elif n_samples < 100:
            return 0.7
        elif n_samples < 1000:
            return 0.85
        else:
            return 0.95

class DiagnosticAnalyzer:
    """诊断性分析器"""
    
    async def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """执行诊断性分析"""
        results = {}
        
        # 根因分析
        results['root_cause'] = await self.root_cause_analysis(data)
        
        # 影响因素分析
        results['impact_factors'] = await self.impact_factor_analysis(data)
        
        # 趋势分解
        if 'timestamp' in data.columns or data.index.name == 'timestamp':
            results['trend_decomposition'] = await self.trend_decomposition(data)
            
        # 异常诊断
        results['anomaly_diagnosis'] = await self.diagnose_anomalies(data)
        
        return results
    
    async def root_cause_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """根因分析"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {}
            
        # 使用互信息识别关键影响因素
        from sklearn.feature_selection import mutual_info_regression
        
        results = {}
        for target_col in numeric_cols:
            feature_cols = [col for col in numeric_cols if col != target_col]
            if not feature_cols:
                continue
                
            X = data[feature_cols].fillna(0)
            y = data[target_col].fillna(0)
            
            mi_scores = mutual_info_regression(X, y)
            
            results[target_col] = {
                'impact_scores': dict(zip(feature_cols, mi_scores)),
                'top_factors': sorted(
                    zip(feature_cols, mi_scores), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
            }
            
        return results
    
    async def impact_factor_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """影响因素分析"""
        # 使用PCA识别主要因素
        numeric_data = data.select_dtypes(include=[np.number]).fillna(0)
        
        if numeric_data.shape[1] < 2:
            return {}
            
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        pca = PCA()
        pca.fit(scaled_data)
        
        # 累积解释方差
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumsum_var >= 0.95) + 1
        
        return {
            'n_significant_components': n_components,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': cumsum_var.tolist(),
            'component_loadings': {
                f'PC{i+1}': dict(zip(numeric_data.columns, pca.components_[i]))
                for i in range(min(5, len(pca.components_)))
            }
        }
    
    async def trend_decomposition(self, data: pd.DataFrame) -> Dict[str, Any]:
        """趋势分解"""
        results = {}
        
        # 确保数据按时间排序
        if 'timestamp' in data.columns:
            data = data.sort_values('timestamp')
            time_col = 'timestamp'
        else:
            time_col = data.index.name
            
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:5]:  # 限制分析前5列
            try:
                # 执行季节性分解
                decomposition = seasonal_decompose(
                    data[col].fillna(method='ffill'), 
                    model='additive',
                    period=min(len(data) // 2, 365)  # 自适应周期
                )
                
                results[col] = {
                    'trend': decomposition.trend.dropna().tolist()[-100:],  # 最近100个点
                    'seasonal': decomposition.seasonal.dropna().tolist()[-100:],
                    'residual': decomposition.resid.dropna().tolist()[-100:],
                    'trend_strength': self.calculate_trend_strength(decomposition)
                }
            except Exception as e:
                logging.warning(f"Trend decomposition failed for {col}: {e}")
                
        return results
    
    def calculate_trend_strength(self, decomposition) -> float:
        """计算趋势强度"""
        try:
            var_residual = np.var(decomposition.resid.dropna())
            var_detrended = np.var(decomposition.observed - decomposition.trend)
            trend_strength = max(0, 1 - var_residual / var_detrended)
            return float(trend_strength)
        except:
            return 0.0
    
    async def diagnose_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """诊断异常"""
        from sklearn.ensemble import IsolationForest
        
        numeric_data = data.select_dtypes(include=[np.number]).fillna(0)
        
        if numeric_data.empty:
            return {}
            
        # 使用Isolation Forest检测异常
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = iso_forest.fit_predict(numeric_data)
        
        # 分析异常模式
        anomaly_indices = np.where(anomaly_labels == -1)[0]
        
        results = {
            'n_anomalies': len(anomaly_indices),
            'anomaly_rate': len(anomaly_indices) / len(data),
            'anomaly_indices': anomaly_indices.tolist()[:100],  # 限制返回数量
            'anomaly_patterns': await self.analyze_anomaly_patterns(
                data.iloc[anomaly_indices] if len(anomaly_indices) > 0 else pd.DataFrame()
            )
        }
        
        return results
    
    async def analyze_anomaly_patterns(self, anomaly_data: pd.DataFrame) -> Dict[str, Any]:
        """分析异常模式"""
        if anomaly_data.empty:
            return {}
            
        patterns = {}
        
        # 时间模式（如果有时间戳）
        if 'timestamp' in anomaly_data.columns:
            timestamps = pd.to_datetime(anomaly_data['timestamp'])
            patterns['temporal'] = {
                'hour_distribution': timestamps.dt.hour.value_counts().to_dict(),
                'weekday_distribution': timestamps.dt.dayofweek.value_counts().to_dict(),
                'month_distribution': timestamps.dt.month.value_counts().to_dict()
            }
            
        # 数值模式
        numeric_cols = anomaly_data.select_dtypes(include=[np.number]).columns
        patterns['numeric'] = {}
        
        for col in numeric_cols:
            patterns['numeric'][col] = {
                'mean_deviation': float(anomaly_data[col].mean()),
                'std_deviation': float(anomaly_data[col].std()),
                'extreme_values': {
                    'min': float(anomaly_data[col].min()),
                    'max': float(anomaly_data[col].max())
                }
            }
            
        return patterns

class PredictiveAnalyzer:
    """预测性分析器"""
    
    async def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """执行预测性分析"""
        results = {}
        
        # 时间序列预测
        if self.has_time_series(data):
            results['time_series_forecast'] = await self.time_series_forecast(data)
            
        # 回归预测
        results['regression_predictions'] = await self.regression_analysis(data)
        
        # 分类预测
        results['classification_predictions'] = await self.classification_analysis(data)
        
        # 聚类分析
        results['clustering'] = await self.clustering_analysis(data)
        
        return results
    
    def has_time_series(self, data: pd.DataFrame) -> bool:
        """检查是否有时间序列数据"""
        return 'timestamp' in data.columns or isinstance(data.index, pd.DatetimeIndex)
    
    async def time_series_forecast(self, data: pd.DataFrame) -> Dict[str, Any]:
        """时间序列预测"""
        results = {}
        
        # 准备时间序列数据
        if 'timestamp' in data.columns:
            data = data.set_index('timestamp').sort_index()
            
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:3]:  # 限制预测前3列
            try:
                # ARIMA模型
                model = ARIMA(data[col].fillna(method='ffill'), order=(1, 1, 1))
                model_fit = model.fit()
                
                # 预测未来10个时间点
                forecast = model_fit.forecast(steps=10)
                
                # 指数平滑
                exp_model = ExponentialSmoothing(
                    data[col].fillna(method='ffill'),
                    seasonal_periods=12,
                    trend='add',
                    seasonal='add'
                )
                exp_fit = exp_model.fit()
                exp_forecast = exp_fit.forecast(steps=10)
                
                results[col] = {
                    'arima_forecast': forecast.tolist(),
                    'exp_smooth_forecast': exp_forecast.tolist(),
                    'model_metrics': {
                        'aic': model_fit.aic,
                        'bic': model_fit.bic
                    }
                }
                
            except Exception as e:
                logging.warning(f"Time series forecast failed for {col}: {e}")
                
        return results
    
    async def regression_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """回归分析"""
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score, mean_squared_error
        
        numeric_data = data.select_dtypes(include=[np.number]).fillna(0)
        
        if numeric_data.shape[1] < 2:
            return {}
            
        results = {}
        
        # 选择目标变量（最后一列）
        X = numeric_data.iloc[:, :-1]
        y = numeric_data.iloc[:, -1]
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 训练多个模型
        models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                results[name] = {
                    'r2_score': r2_score(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'feature_importance': self.get_feature_importance(model, X.columns)
                }
                
            except Exception as e:
                logging.warning(f"Regression with {name} failed: {e}")
                
        return results
    
    def get_feature_importance(self, model, feature_names) -> Dict[str, float]:
        """获取特征重要性"""
        if hasattr(model, 'feature_importances_'):
            return dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            return dict(zip(feature_names, np.abs(model.coef_)))
        else:
            return {}
    
    async def classification_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """分类分析"""
        # 这里实现分类分析逻辑
        return {}
    
    async def clustering_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """聚类分析"""
        numeric_data = data.select_dtypes(include=[np.number]).fillna(0)
        
        if numeric_data.shape[1] < 2:
            return {}
            
        # 标准化数据
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        results = {}
        
        # K-means聚类
        silhouette_scores = []
        K = range(2, min(10, len(data) // 10))
        
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(scaled_data)
            score = silhouette_score(scaled_data, labels)
            silhouette_scores.append(score)
            
        optimal_k = K[np.argmax(silhouette_scores)]
        
        # 使用最优K值进行聚类
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        labels = kmeans.fit_predict(scaled_data)
        
        results['kmeans'] = {
            'optimal_k': optimal_k,
            'silhouette_scores': dict(zip(K, silhouette_scores)),
            'cluster_labels': labels.tolist()[:100],  # 限制返回数量
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'cluster_sizes': dict(zip(*np.unique(labels, return_counts=True)))
        }
        
        # DBSCAN聚类
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(scaled_data)
        
        results['dbscan'] = {
            'n_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
            'n_noise_points': list(dbscan_labels).count(-1),
            'cluster_labels': dbscan_labels.tolist()[:100]
        }
        
        return results

# ==================== 高级可视化系统 ====================

class AdvancedVisualizationEngine:
    """高级可视化引擎"""
    
    def __init__(self):
        self.theme_manager = ThemeManager()
        self.layout_optimizer = LayoutOptimizer()
        self.interaction_handler = InteractionHandler()
        plt.style.use('seaborn-v0_8-darkgrid')
        
    async def create_visualization(self, data: Any, viz_type: str, 
                                 options: Dict[str, Any] = None) -> Dict[str, Any]:
        """创建可视化"""
        options = options or {}
        
        # 应用主题
        theme = options.get('theme', 'default')
        self.theme_manager.apply_theme(theme)
        
        # 创建可视化
        if viz_type == 'dashboard':
            return await self.create_dashboard(data, options)
        elif viz_type == 'time_series':
            return await self.create_time_series_viz(data, options)
        elif viz_type == 'network':
            return await self.create_network_viz(data, options)
        elif viz_type == 'heatmap':
            return await self.create_heatmap(data, options)
        elif viz_type == '3d_scatter':
            return await self.create_3d_scatter(data, options)
        elif viz_type == 'sankey':
            return await self.create_sankey_diagram(data, options)
        elif viz_type == 'treemap':
            return await self.create_treemap(data, options)
        else:
            raise ValueError(f"Unknown visualization type: {viz_type}")
    
    async def create_dashboard(self, data: Dict[str, Any], 
                             options: Dict[str, Any]) -> Dict[str, Any]:
        """创建仪表板"""
        fig = plt.figure(figsize=(16, 10))
        
        # 优化布局
        layout = self.layout_optimizer.optimize_layout(
            n_plots=len(data),
            aspect_ratio=options.get('aspect_ratio', 16/10)
        )
        
        plots = {}
        
        for i, (key, plot_data) in enumerate(data.items()):
            ax = plt.subplot(layout['rows'], layout['cols'], i + 1)
            
            # 根据数据类型选择图表
            if isinstance(plot_data, pd.Series):
                ax.plot(plot_data.index, plot_data.values)
                ax.set_title(key)
            elif isinstance(plot_data, pd.DataFrame):
                plot_data.plot(ax=ax)
                ax.set_title(key)
            elif isinstance(plot_data, dict) and 'x' in plot_data and 'y' in plot_data:
                ax.plot(plot_data['x'], plot_data['y'])
                ax.set_title(key)
                
            plots[key] = ax
            
        plt.tight_layout()
        
        # 保存图表
        dashboard_path = f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        
        return {
            'type': 'dashboard',
            'path': dashboard_path,
            'layout': layout,
            'plots': list(plots.keys())
        }
    
    async def create_time_series_viz(self, data: pd.DataFrame, 
                                   options: Dict[str, Any]) -> Dict[str, Any]:
        """创建时间序列可视化"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # 主时间序列图
        ax1 = axes[0]
        for col in data.select_dtypes(include=[np.number]).columns[:5]:
            ax1.plot(data.index, data[col], label=col, linewidth=2)
        ax1.set_title('Time Series Data')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 变化率图
        ax2 = axes[1]
        for col in data.select_dtypes(include=[np.number]).columns[:5]:
            change_rate = data[col].pct_change() * 100
            ax2.plot(data.index, change_rate, label=f'{col} % Change', 
                    linewidth=1.5, alpha=0.8)
        ax2.set_title('Percentage Change')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        # 累积图
        ax3 = axes[2]
        for col in data.select_dtypes(include=[np.number]).columns[:5]:
            cumsum = data[col].cumsum()
            ax3.plot(data.index, cumsum, label=f'{col} Cumulative', linewidth=2)
        ax3.set_title('Cumulative Sum')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        ts_path = f"timeseries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(ts_path, dpi=300, bbox_inches='tight')
        
        return {
            'type': 'time_series',
            'path': ts_path,
            'n_series': len(data.columns),
            'time_range': {
                'start': str(data.index[0]),
                'end': str(data.index[-1])
            }
        }
    
    async def create_network_viz(self, graph_data: Union[nx.Graph, Dict], 
                               options: Dict[str, Any]) -> Dict[str, Any]:
        """创建网络可视化"""
        if isinstance(graph_data, dict):
            G = nx.from_dict_of_dicts(graph_data)
        else:
            G = graph_data
            
        # 创建交互式网络图
        net = Network(height='750px', width='100%', bgcolor='#222222', 
                     font_color='white')
        
        # 添加节点和边
        for node in G.nodes():
            net.add_node(node, label=str(node), 
                        size=20 + G.degree(node) * 2)
            
        for edge in G.edges():
            net.add_edge(edge[0], edge[1])
            
        # 配置物理引擎
        net.set_options("""
        {
            "physics": {
                "enabled": true,
                "solver": "forceAtlas2Based",
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 100,
                    "springConstant": 0.08
                }
            }
        }
        """)
        
        # 保存网络图
        network_path = f"network_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        net.save_graph(network_path)
        
        # 计算网络统计
        stats = {
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'density': nx.density(G),
            'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
            'clustering_coefficient': nx.average_clustering(G) if not G.is_directed() else None
        }
        
        return {
            'type': 'network',
            'path': network_path,
            'stats': stats
        }
    
    async def create_3d_scatter(self, data: pd.DataFrame, 
                              options: Dict[str, Any]) -> Dict[str, Any]:
        """创建3D散点图"""
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # 选择前三个数值列
        numeric_cols = data.select_dtypes(include=[np.number]).columns[:3]
        
        if len(numeric_cols) < 3:
            raise ValueError("Need at least 3 numeric columns for 3D scatter")
            
        x = data[numeric_cols[0]]
        y = data[numeric_cols[1]]
        z = data[numeric_cols[2]]
        
        # 颜色映射
        colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
        
        scatter = ax.scatter(x, y, z, c=colors, s=50, alpha=0.6)
        
        ax.set_xlabel(numeric_cols[0])
        ax.set_ylabel(numeric_cols[1])
        ax.set_zlabel(numeric_cols[2])
        ax.set_title('3D Scatter Plot')
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 保存图表
        scatter_path = f"3d_scatter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        
        return {
            'type': '3d_scatter',
            'path': scatter_path,
            'axes': numeric_cols.tolist()
        }

class ThemeManager:
    """主题管理器"""
    
    def __init__(self):
        self.themes = {
            'default': {
                'figure.facecolor': 'white',
                'axes.facecolor': 'white',
                'axes.edgecolor': 'black',
                'axes.labelcolor': 'black',
                'text.color': 'black',
                'xtick.color': 'black',
                'ytick.color': 'black',
                'grid.color': 'gray',
                'grid.alpha': 0.3
            },
            'dark': {
                'figure.facecolor': '#1e1e1e',
                'axes.facecolor': '#2d2d2d',
                'axes.edgecolor': '#cccccc',
                'axes.labelcolor': '#cccccc',
                'text.color': '#cccccc',
                'xtick.color': '#cccccc',
                'ytick.color': '#cccccc',
                'grid.color': '#555555',
                'grid.alpha': 0.3
            },
            'minimal': {
                'figure.facecolor': 'white',
                'axes.facecolor': 'white',
                'axes.edgecolor': 'none',
                'axes.labelcolor': '#333333',
                'text.color': '#333333',
                'xtick.color': '#333333',
                'ytick.color': '#333333',
                'grid.color': '#cccccc',
                'grid.alpha': 0.1
            }
        }
        
    def apply_theme(self, theme_name: str):
        """应用主题"""
        if theme_name in self.themes:
            theme = self.themes[theme_name]
            for key, value in theme.items():
                plt.rcParams[key] = value

class LayoutOptimizer:
    """布局优化器"""
    
    def optimize_layout(self, n_plots: int, aspect_ratio: float = 16/9) -> Dict[str, int]:
        """优化布局"""
        # 计算最佳行列数
        rows = int(np.sqrt(n_plots / aspect_ratio))
        cols = int(np.ceil(n_plots / rows))
        
        # 调整以确保能容纳所有图
        while rows * cols < n_plots:
            if cols / rows < aspect_ratio:
                cols += 1
            else:
                rows += 1
                
        return {'rows': rows, 'cols': cols}

# ==================== 洞察生成系统 ====================

class InsightGenerator:
    """洞察生成器"""
    
    def __init__(self):
        self.pattern_detector = PatternDetector()
        self.trend_analyzer = TrendAnalyzer()
        self.anomaly_explainer = AnomalyExplainer()
        self.recommendation_engine = RecommendationEngine()
        
    async def generate_insights(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成综合洞察"""
        insights = {
            'key_findings': [],
            'patterns': [],
            'trends': [],
            'anomalies': [],
            'recommendations': [],
            'action_items': []
        }
        
        # 从各种分析中提取洞察
        for analysis_type, result in analysis_results.items():
            if isinstance(result, AnalysisResult):
                insights['key_findings'].extend(result.insights)
                insights['recommendations'].extend(result.recommendations)
                
                # 检测模式
                patterns = await self.pattern_detector.detect_patterns(result.data)
                insights['patterns'].extend(patterns)
                
                # 分析趋势
                trends = await self.trend_analyzer.analyze_trends(result.data)
                insights['trends'].extend(trends)
                
        # 生成行动项
        insights['action_items'] = await self.generate_action_items(insights)
        
        # 优先级排序
        insights = await self.prioritize_insights(insights)
        
        return insights
    
    async def generate_action_items(self, insights: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """生成行动项"""
        action_items = []
        
        # 基于发现生成行动项
        for finding in insights['key_findings'][:10]:  # 限制数量
            action = {
                'description': f"Investigate: {finding}",
                'priority': 'medium',
                'category': 'investigation',
                'estimated_impact': 'medium'
            }
            action_items.append(action)
            
        # 基于异常生成行动项
        for anomaly in insights['anomalies'][:5]:
            action = {
                'description': f"Address anomaly: {anomaly}",
                'priority': 'high',
                'category': 'remediation',
                'estimated_impact': 'high'
            }
            action_items.append(action)
            
        return action_items
    
    async def prioritize_insights(self, insights: Dict[str, List]) -> Dict[str, List]:
        """优先级排序洞察"""
        # 实现洞察优先级排序逻辑
        return insights

class PatternDetector:
    """模式检测器"""
    
    async def detect_patterns(self, data: Dict[str, Any]) -> List[str]:
        """检测数据模式"""
        patterns = []
        
        # 检测周期性模式
        if 'distributions' in data:
            for col, dist in data['distributions'].items():
                if 'autocorrelation' in dist:
                    if dist['autocorrelation'] > 0.7:
                        patterns.append(f"Strong periodic pattern detected in {col}")
                        
        # 检测相关性模式
        if 'correlations' in data:
            corr_matrix = data['correlations']
            for col1, correlations in corr_matrix.items():
                for col2, corr in correlations.items():
                    if col1 < col2 and abs(corr) > 0.8:
                        patterns.append(f"Strong correlation pattern between {col1} and {col2}")
                        
        return patterns

class TrendAnalyzer:
    """趋势分析器"""
    
    async def analyze_trends(self, data: Dict[str, Any]) -> List[str]:
        """分析趋势"""
        trends = []
        
        # 分析时间序列趋势
        if 'trend_decomposition' in data:
            for col, decomp in data['trend_decomposition'].items():
                if 'trend_strength' in decomp and decomp['trend_strength'] > 0.7:
                    trends.append(f"Strong trend detected in {col}")
                    
        return trends

# ==================== 高级异常检测 ====================

class AdvancedAnomalyDetector:
    """高级异常检测器"""
    
    def __init__(self):
        self.detectors = {
            'statistical': StatisticalAnomalyDetector(),
            'isolation_forest': IsolationForestDetector(),
            'autoencoder': AutoencoderDetector(),
            'clustering': ClusteringAnomalyDetector()
        }
        
    async def detect(self, data: pd.DataFrame) -> Dict[str, Any]:
        """执行综合异常检测"""
        results = {}
        
        # 运行所有检测器
        for name, detector in self.detectors.items():
            try:
                anomalies = await detector.detect(data)
                results[name] = anomalies
            except Exception as e:
                logging.error(f"Anomaly detection failed for {name}: {e}")
                
        # 综合结果
        combined_anomalies = await self.combine_results(results)
        
        # 生成解释
        explanations = await self.generate_explanations(combined_anomalies, data)
        
        return {
            'individual_results': results,
            'combined_anomalies': combined_anomalies,
            'explanations': explanations
        }
    
    async def combine_results(self, results: Dict[str, Any]) -> List[int]:
        """综合多个检测器的结果"""
        # 投票机制
        all_indices = []
        for detector_results in results.values():
            if 'anomaly_indices' in detector_results:
                all_indices.extend(detector_results['anomaly_indices'])
                
        # 计数每个索引出现的次数
        index_counts = Counter(all_indices)
        
        # 至少被两个检测器标记为异常
        combined = [idx for idx, count in index_counts.items() if count >= 2]
        
        return sorted(combined)
    
    async def generate_explanations(self, anomaly_indices: List[int], 
                                  data: pd.DataFrame) -> List[Dict[str, Any]]:
        """生成异常解释"""
        explanations = []
        
        for idx in anomaly_indices[:10]:  # 限制解释数量
            if idx < len(data):
                row = data.iloc[idx]
                
                explanation = {
                    'index': idx,
                    'features': {},
                    'severity': 'medium',
                    'possible_causes': []
                }
                
                # 分析每个特征的异常程度
                for col in data.columns:
                    if pd.api.types.is_numeric_dtype(data[col]):
                        value = row[col]
                        mean = data[col].mean()
                        std = data[col].std()
                        
                        z_score = abs((value - mean) / std) if std > 0 else 0
                        
                        if z_score > 3:
                            explanation['features'][col] = {
                                'value': value,
                                'z_score': z_score,
                                'deviation': 'extreme'
                            }
                            explanation['severity'] = 'high'
                        elif z_score > 2:
                            explanation['features'][col] = {
                                'value': value,
                                'z_score': z_score,
                                'deviation': 'significant'
                            }
                            
                # 生成可能原因
                if explanation['features']:
                    explanation['possible_causes'] = [
                        f"Unusual combination of {', '.join(explanation['features'].keys())}",
                        "Potential data quality issue",
                        "Rare but legitimate event"
                    ]
                    
                explanations.append(explanation)
                
        return explanations

class StatisticalAnomalyDetector:
    """统计异常检测器"""
    
    async def detect(self, data: pd.DataFrame) -> Dict[str, Any]:
        """使用统计方法检测异常"""
        anomalies = []
        
        for col in data.select_dtypes(include=[np.number]).columns:
            # Z-score方法
            z_scores = np.abs(stats.zscore(data[col].fillna(data[col].mean())))
            anomaly_mask = z_scores > 3
            
            # IQR方法
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
            
            # 组合两种方法
            combined_mask = anomaly_mask | iqr_mask
            anomalies.extend(data[combined_mask].index.tolist())
            
        return {
            'anomaly_indices': list(set(anomalies)),
            'method': 'statistical'
        }

# ==================== 报告生成系统 ====================

class ReportGenerator:
    """报告生成器"""
    
    def __init__(self):
        self.template_engine = TemplateEngine()
        self.chart_generator = ChartGenerator()
        self.export_manager = ExportManager()
        
    async def generate_report(self, analysis_results: Dict[str, Any], 
                            report_type: str = 'comprehensive') -> Dict[str, Any]:
        """生成分析报告"""
        # 选择报告模板
        template = self.template_engine.get_template(report_type)
        
        # 准备报告数据
        report_data = await self.prepare_report_data(analysis_results)
        
        # 生成图表
        charts = await self.chart_generator.generate_charts(report_data)
        
        # 生成报告内容
        report_content = await self.generate_content(template, report_data, charts)
        
        # 导出报告
        export_paths = await self.export_manager.export_report(
            report_content,
            formats=['html', 'pdf', 'docx']
        )
        
        return {
            'report_type': report_type,
            'generated_at': datetime.now(),
            'export_paths': export_paths,
            'summary': report_data.get('executive_summary', '')
        }
    
    async def prepare_report_data(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """准备报告数据"""
        report_data = {
            'title': f"Analysis Report - {datetime.now().strftime('%Y-%m-%d')}",
            'executive_summary': await self.generate_executive_summary(analysis_results),
            'key_findings': [],
            'detailed_analysis': {},
            'recommendations': [],
            'appendix': {}
        }
        
        # 提取关键发现
        for analysis_type, results in analysis_results.items():
            if 'insights' in results:
                report_data['key_findings'].extend(results['insights'])
                
        # 准备详细分析
        report_data['detailed_analysis'] = analysis_results
        
        return report_data
    
    async def generate_executive_summary(self, analysis_results: Dict[str, Any]) -> str:
        """生成执行摘要"""
        summary_points = []
        
        # 提取关键信息
        if 'analyses' in analysis_results:
            n_analyses = len(analysis_results['analyses'])
            summary_points.append(f"Conducted {n_analyses} types of analysis")
            
        if 'anomalies' in analysis_results:
            n_anomalies = len(analysis_results['anomalies'].get('combined_anomalies', []))
            if n_anomalies > 0:
                summary_points.append(f"Detected {n_anomalies} anomalies requiring attention")
                
        if 'insights' in analysis_results:
            n_insights = len(analysis_results['insights'].get('key_findings', []))
            summary_points.append(f"Generated {n_insights} key insights")
            
        return " ".join(summary_points)

# ==================== 实时监控和告警系统 ====================

class RealTimeMonitor:
    """实时监控器"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.dashboard_server = DashboardServer()
        self.streaming_analyzer = StreamingAnalyzer()
        
    async def start_monitoring(self, data_source: Any, 
                             monitoring_config: Dict[str, Any]):
        """启动监控"""
        # 设置数据流
        data_stream = await self.setup_data_stream(data_source)
        
        # 启动指标收集
        collector_task = asyncio.create_task(
            self.metrics_collector.collect_metrics(data_stream)
        )
        
        # 启动流式分析
        analyzer_task = asyncio.create_task(
            self.streaming_analyzer.analyze_stream(data_stream)
        )
        
        # 启动告警监控
        alert_task = asyncio.create_task(
            self.alert_manager.monitor_alerts(self.metrics_collector)
        )
        
        # 启动仪表板服务
        dashboard_task = asyncio.create_task(
            self.dashboard_server.serve_dashboard(self.metrics_collector)
        )
        
        # 等待所有任务
        await asyncio.gather(
            collector_task, analyzer_task, alert_task, dashboard_task
        )
    
    async def setup_data_stream(self, data_source: Any) -> AsyncIterator:
        """设置数据流"""
        # 实现数据流设置逻辑
        pass

class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        self.metrics = defaultdict(lambda: deque(maxlen=1000))
        self.aggregators = {
            'mean': np.mean,
            'std': np.std,
            'min': np.min,
            'max': np.max,
            'percentile_95': lambda x: np.percentile(x, 95)
        }
        
    async def collect_metrics(self, data_stream: AsyncIterator):
        """收集指标"""
        async for data_point in data_stream:
            timestamp = datetime.now()
            
            # 提取指标
            metrics = await self.extract_metrics(data_point)
            
            # 存储指标
            for metric_name, value in metrics.items():
                self.metrics[metric_name].append({
                    'timestamp': timestamp,
                    'value': value
                })
                
            # 计算聚合指标
            await self.update_aggregates()
            
    async def extract_metrics(self, data_point: Any) -> Dict[str, float]:
        """提取指标"""
        # 实现指标提取逻辑
        return {}
    
    async def update_aggregates(self):
        """更新聚合指标"""
        for metric_name, values in self.metrics.items():
            if len(values) > 10:  # 至少10个数据点
                metric_values = [v['value'] for v in values]
                
                for agg_name, agg_func in self.aggregators.items():
                    agg_value = agg_func(metric_values)
                    self.metrics[f"{metric_name}_{agg_name}"].append({
                        'timestamp': datetime.now(),
                        'value': agg_value
                    })

class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self.alert_rules = []
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.notification_channels = []
        
    async def monitor_alerts(self, metrics_collector: MetricsCollector):
        """监控告警"""
        while True:
            # 检查所有告警规则
            for rule in self.alert_rules:
                alert_triggered = await self.check_rule(rule, metrics_collector)
                
                if alert_triggered:
                    await self.trigger_alert(rule)
                elif rule['id'] in self.active_alerts:
                    await self.resolve_alert(rule['id'])
                    
            await asyncio.sleep(10)  # 每10秒检查一次
            
    async def check_rule(self, rule: Dict[str, Any], 
                        metrics_collector: MetricsCollector) -> bool:
        """检查告警规则"""
        metric_name = rule['metric']
        threshold = rule['threshold']
        condition = rule['condition']
        
        if metric_name in metrics_collector.metrics:
            recent_values = [v['value'] for v in 
                           list(metrics_collector.metrics[metric_name])[-10:]]
            
            if recent_values:
                current_value = recent_values[-1]
                
                if condition == 'greater_than':
                    return current_value > threshold
                elif condition == 'less_than':
                    return current_value < threshold
                elif condition == 'equals':
                    return current_value == threshold
                    
        return False
    
    async def trigger_alert(self, rule: Dict[str, Any]):
        """触发告警"""
        alert = {
            'id': f"alert_{datetime.now().timestamp()}",
            'rule_id': rule['id'],
            'metric': rule['metric'],
            'severity': rule.get('severity', 'medium'),
            'message': rule.get('message', f"Alert: {rule['metric']} threshold exceeded"),
            'triggered_at': datetime.now(),
            'status': 'active'
        }
        
        self.active_alerts[rule['id']] = alert
        self.alert_history.append(alert)
        
        # 发送通知
        for channel in self.notification_channels:
            await channel.send_notification(alert)

# ==================== 辅助类 ====================

class TemplateEngine:
    """模板引擎"""
    
    def get_template(self, template_type: str) -> str:
        """获取报告模板"""
        # 实现模板获取逻辑
        return ""

class ChartGenerator:
    """图表生成器"""
    
    async def generate_charts(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成图表"""
        # 实现图表生成逻辑
        return []

class ExportManager:
    """导出管理器"""
    
    async def export_report(self, content: str, formats: List[str]) -> Dict[str, str]:
        """导出报告"""
        # 实现报告导出逻辑
        return {}

class DashboardServer:
    """仪表板服务器"""
    
    async def serve_dashboard(self, metrics_collector: MetricsCollector):
        """提供仪表板服务"""
        # 实现仪表板服务逻辑
        pass

class StreamingAnalyzer:
    """流式分析器"""
    
    async def analyze_stream(self, data_stream: AsyncIterator):
        """分析数据流"""
        # 实现流式分析逻辑
        pass

# ==================== 工具类 ====================

class DataQualityChecker:
    """数据质量检查器"""
    
    async def check_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """检查数据质量"""
        quality_report = {
            'completeness': await self.check_completeness(data),
            'consistency': await self.check_consistency(data),
            'accuracy': await self.check_accuracy(data),
            'timeliness': await self.check_timeliness(data),
            'overall_score': 0.0
        }
        
        # 计算总体评分
        scores = [
            quality_report['completeness']['score'],
            quality_report['consistency']['score'],
            quality_report['accuracy']['score'],
            quality_report['timeliness']['score']
        ]
        quality_report['overall_score'] = np.mean(scores)
        
        return quality_report
    
    async def check_completeness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """检查完整性"""
        total_cells = data.size
        missing_cells = data.isnull().sum().sum()
        completeness_score = 1.0 - (missing_cells / total_cells)
        
        return {
            'score': completeness_score,
            'missing_cells': int(missing_cells),
            'total_cells': int(total_cells),
            'missing_by_column': data.isnull().sum().to_dict()
        }
    
    async def check_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """检查一致性"""
        # 实现一致性检查逻辑
        return {'score': 0.9}
    
    async def check_accuracy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """检查准确性"""
        # 实现准确性检查逻辑
        return {'score': 0.85}
    
    async def check_timeliness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """检查时效性"""
        # 实现时效性检查逻辑
        return {'score': 0.95}

# ==================== 集成接口 ====================

class AnalysisIntegrationHub:
    """分析集成中心"""
    
    def __init__(self):
        self.analyzer = AdvancedDataAnalyzer()
        self.visualizer = AdvancedVisualizationEngine()
        self.report_generator = ReportGenerator()
        self.monitor = RealTimeMonitor()
        self.quality_checker = DataQualityChecker()
        
    async def perform_analysis(self, data: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """执行完整分析流程"""
        results = {
            'timestamp': datetime.now(),
            'config': config
        }
        
        try:
            # 数据质量检查
            quality_report = await self.quality_checker.check_quality(data)
            results['data_quality'] = quality_report
            
            if quality_report['overall_score'] < 0.6:
                logging.warning("Data quality is low, results may be unreliable")
                
            # 执行分析
            analysis_results = await self.analyzer.analyze(
                data,
                config.get('analysis_types')
            )
            results['analysis'] = analysis_results
            
            # 生成可视化
            viz_results = []
            for viz_type in config.get('visualizations', ['dashboard']):
                viz = await self.visualizer.create_visualization(
                    data, viz_type, config.get('viz_options', {})
                )
                viz_results.append(viz)
            results['visualizations'] = viz_results
            
            # 生成报告
            if config.get('generate_report', True):
                report = await self.report_generator.generate_report(
                    analysis_results,
                    config.get('report_type', 'comprehensive')
                )
                results['report'] = report
                
            return results
            
        except Exception as e:
            logging.error(f"Analysis failed: {e}")
            results['error'] = str(e)
            results['success'] = False
            return results