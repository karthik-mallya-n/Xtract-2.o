"""
Comprehensive Visualization Engine
Handles all types of data visualizations for the ML platform
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import base64
import io
from datetime import datetime
import warnings
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json

warnings.filterwarnings('ignore')

class VisualizationEngine:
    """
    Comprehensive visualization engine supporting all major chart types
    """
    
    def __init__(self):
        # Configure matplotlib/seaborn defaults
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Configure plotly defaults
        self.plotly_config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
        }
    
    def _convert_to_json_serializable(self, obj):
        """
        Convert numpy/pandas types to JSON serializable Python types
        """
        if hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj
        
    def analyze_columns(self, df):
        """
        Analyze dataset columns and categorize them
        """
        analysis = {
            'total_rows': int(len(df)),
            'total_columns': int(len(df.columns)),
            'columns': {}
        }
        
        for col in df.columns:
            col_info = {
                'type': str(df[col].dtype),
                'non_null_count': int(df[col].count()),
                'null_count': int(df[col].isnull().sum()),
                'unique_count': int(df[col].nunique()),
                'is_categorical': False,
                'is_continuous': False,
                'is_datetime': False
            }
            
            # Determine column category
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                col_info['is_datetime'] = True
                col_info['category'] = 'datetime'
            elif df[col].dtype == 'object' or df[col].nunique() < 20:
                col_info['is_categorical'] = True
                col_info['category'] = 'categorical'
            else:
                col_info['is_continuous'] = True
                col_info['category'] = 'continuous'
                
            # Add sample values (convert to Python native types)
            sample_values = df[col].dropna().head(5).tolist()
            # Convert numpy types to Python native types
            col_info['sample_values'] = [
                item.item() if hasattr(item, 'item') else item 
                for item in sample_values
            ]
            
            analysis['columns'][col] = col_info
            
        return self._convert_to_json_serializable(analysis)
    
    def get_available_visualizations(self, df):
        """
        Get available visualization types based on data
        """
        analysis = self.analyze_columns(df)
        
        categorical_cols = [col for col, info in analysis['columns'].items() 
                           if info['is_categorical']]
        continuous_cols = [col for col, info in analysis['columns'].items() 
                          if info['is_continuous']]
        datetime_cols = [col for col, info in analysis['columns'].items() 
                        if info['is_datetime']]
        
        visualizations = {
            'univariate': {
                'categorical': ['bar_chart', 'count_plot', 'pie_chart', 'donut_chart'],
                'continuous': ['histogram', 'density_plot', 'box_plot', 'violin_plot', 'rug_plot']
            },
            'bivariate': {
                'continuous_vs_continuous': ['scatter_plot', 'line_plot', 'hexbin_plot', 'density_2d', 'joint_plot'],
                'categorical_vs_continuous': ['box_plot', 'violin_plot', 'strip_plot', 'swarm_plot', 'bar_plot'],
                'categorical_vs_categorical': ['heatmap_crosstab', 'stacked_bar', 'side_by_side_bar']
            },
            'multivariate': ['correlation_heatmap', 'pair_plot', 'parallel_coordinates', 'andrews_curves', 
                           'scatter_3d', 'cluster_plot', 'bubble_chart'],
            'time_series': ['line_chart', 'area_chart', 'rolling_mean', 'seasonal_decomposition'] if datetime_cols else [],
            'distribution': ['histogram', 'kde_plot', 'qq_plot', 'box_plot', 'violin_plot', 'ecdf_plot']
        }
        
        result = {
            'visualizations': visualizations,
            'data_context': {
                'total_rows': int(len(df)),
                'total_columns': int(len(df.columns)),
                'categorical_columns': categorical_cols,
                'continuous_columns': continuous_cols,
                'datetime_columns': datetime_cols
            }
        }
        
        return self._convert_to_json_serializable(result)
    
    def create_visualization(self, df, viz_type, params):
        """
        Create visualization based on type and parameters
        """
        try:
            if viz_type == 'histogram':
                return self._create_histogram(df, params)
            elif viz_type == 'bar_chart':
                return self._create_bar_chart(df, params)
            elif viz_type == 'scatter_plot':
                return self._create_scatter_plot(df, params)
            elif viz_type == 'box_plot':
                return self._create_box_plot(df, params)
            elif viz_type == 'correlation_heatmap':
                return self._create_correlation_heatmap(df, params)
            elif viz_type == 'pie_chart':
                return self._create_pie_chart(df, params)
            elif viz_type == 'violin_plot':
                return self._create_violin_plot(df, params)
            elif viz_type == 'density_plot':
                return self._create_density_plot(df, params)
            elif viz_type == 'pair_plot':
                return self._create_pair_plot(df, params)
            elif viz_type == 'line_chart':
                return self._create_line_chart(df, params)
            elif viz_type == 'heatmap_crosstab':
                return self._create_crosstab_heatmap(df, params)
            elif viz_type == 'scatter_3d':
                return self._create_3d_scatter(df, params)
            else:
                return self._create_default_visualization(df, params)
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to create {viz_type}: {str(e)}',
                'html': f'<div style="color: red;">Error: {str(e)}</div>'
            }
    
    def _create_histogram(self, df, params):
        """Create histogram visualization"""
        column = params.get('column')
        bins = params.get('bins', 30)
        
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        # Create plotly histogram
        fig = px.histogram(
            df, 
            x=column, 
            nbins=bins,
            title=f'Distribution of {column}',
            template='plotly_white'
        )
        
        fig.update_layout(
            xaxis_title=column,
            yaxis_title='Frequency',
            showlegend=False
        )
        
        return {
            'success': True,
            'html': fig.to_html(config=self.plotly_config),
            'type': 'histogram',
            'column': column
        }
    
    def _create_bar_chart(self, df, params):
        """Create bar chart visualization"""
        column = params.get('column')
        
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        # Get value counts
        value_counts = df[column].value_counts().head(20)  # Limit to top 20
        
        fig = px.bar(
            x=value_counts.values,
            y=value_counts.index,
            orientation='h',
            title=f'Count of {column}',
            template='plotly_white'
        )
        
        fig.update_layout(
            xaxis_title='Count',
            yaxis_title=column,
            showlegend=False
        )
        
        return {
            'success': True,
            'html': fig.to_html(config=self.plotly_config),
            'type': 'bar_chart',
            'column': column
        }
    
    def _create_scatter_plot(self, df, params):
        """Create scatter plot visualization"""
        x_col = params.get('x_column')
        y_col = params.get('y_column')
        color_col = params.get('color_column')
        
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError("Invalid column selection")
        
        fig = px.scatter(
            df, 
            x=x_col, 
            y=y_col,
            color=color_col if color_col in df.columns else None,
            title=f'{y_col} vs {x_col}',
            template='plotly_white'
        )
        
        return {
            'success': True,
            'html': fig.to_html(config=self.plotly_config),
            'type': 'scatter_plot',
            'x_column': x_col,
            'y_column': y_col
        }
    
    def _create_box_plot(self, df, params):
        """Create box plot visualization"""
        x_col = params.get('x_column')
        y_col = params.get('y_column')
        
        if y_col not in df.columns:
            raise ValueError(f"Column '{y_col}' not found")
        
        if x_col and x_col in df.columns:
            fig = px.box(
                df, 
                x=x_col, 
                y=y_col,
                title=f'Box Plot: {y_col} by {x_col}',
                template='plotly_white'
            )
        else:
            fig = px.box(
                df, 
                y=y_col,
                title=f'Box Plot: {y_col}',
                template='plotly_white'
            )
        
        return {
            'success': True,
            'html': fig.to_html(config=self.plotly_config),
            'type': 'box_plot',
            'y_column': y_col
        }
    
    def _create_correlation_heatmap(self, df, params):
        """Create correlation heatmap"""
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            raise ValueError("No numeric columns found for correlation")
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        fig = px.imshow(
            corr_matrix,
            title='Correlation Heatmap',
            template='plotly_white',
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        
        return {
            'success': True,
            'html': fig.to_html(config=self.plotly_config),
            'type': 'correlation_heatmap'
        }
    
    def _create_pie_chart(self, df, params):
        """Create pie chart visualization"""
        column = params.get('column')
        
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        # Get value counts (limit to top 10 for readability)
        value_counts = df[column].value_counts().head(10)
        
        fig = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=f'Distribution of {column}',
            template='plotly_white'
        )
        
        return {
            'success': True,
            'html': fig.to_html(config=self.plotly_config),
            'type': 'pie_chart',
            'column': column
        }
    
    def _create_violin_plot(self, df, params):
        """Create violin plot visualization"""
        x_col = params.get('x_column')
        y_col = params.get('y_column')
        
        if y_col not in df.columns:
            raise ValueError(f"Column '{y_col}' not found")
        
        if x_col and x_col in df.columns:
            fig = px.violin(
                df, 
                x=x_col, 
                y=y_col,
                title=f'Violin Plot: {y_col} by {x_col}',
                template='plotly_white'
            )
        else:
            fig = px.violin(
                df, 
                y=y_col,
                title=f'Violin Plot: {y_col}',
                template='plotly_white'
            )
        
        return {
            'success': True,
            'html': fig.to_html(config=self.plotly_config),
            'type': 'violin_plot',
            'y_column': y_col
        }
    
    def _create_density_plot(self, df, params):
        """Create density plot visualization"""
        column = params.get('column')
        
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        # Create density plot using plotly
        fig = ff.create_distplot(
            [df[column].dropna()], 
            [column], 
            show_hist=False,
            show_rug=False
        )
        
        fig.update_layout(
            title=f'Density Plot: {column}',
            template='plotly_white',
            showlegend=False
        )
        
        return {
            'success': True,
            'html': fig.to_html(config=self.plotly_config),
            'type': 'density_plot',
            'column': column
        }
    
    def _create_pair_plot(self, df, params):
        """Create pair plot visualization"""
        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            raise ValueError("Need at least 2 numeric columns for pair plot")
        
        # Limit to first 5 columns for performance
        cols_to_plot = numeric_cols[:5]
        
        # Create scatter matrix
        fig = px.scatter_matrix(
            df[cols_to_plot],
            title='Pair Plot (Scatter Matrix)',
            template='plotly_white'
        )
        
        return {
            'success': True,
            'html': fig.to_html(config=self.plotly_config),
            'type': 'pair_plot'
        }
    
    def _create_line_chart(self, df, params):
        """Create line chart visualization"""
        x_col = params.get('x_column')
        y_col = params.get('y_column')
        
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError("Invalid column selection")
        
        fig = px.line(
            df, 
            x=x_col, 
            y=y_col,
            title=f'{y_col} over {x_col}',
            template='plotly_white'
        )
        
        return {
            'success': True,
            'html': fig.to_html(config=self.plotly_config),
            'type': 'line_chart',
            'x_column': x_col,
            'y_column': y_col
        }
    
    def _create_crosstab_heatmap(self, df, params):
        """Create crosstab heatmap"""
        x_col = params.get('x_column')
        y_col = params.get('y_column')
        
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError("Invalid column selection")
        
        # Create crosstab
        crosstab = pd.crosstab(df[x_col], df[y_col])
        
        fig = px.imshow(
            crosstab,
            title=f'Crosstab Heatmap: {x_col} vs {y_col}',
            template='plotly_white',
            color_continuous_scale='Blues',
            aspect='auto'
        )
        
        return {
            'success': True,
            'html': fig.to_html(config=self.plotly_config),
            'type': 'crosstab_heatmap',
            'x_column': x_col,
            'y_column': y_col
        }
    
    def _create_3d_scatter(self, df, params):
        """Create 3D scatter plot"""
        x_col = params.get('x_column')
        y_col = params.get('y_column')
        z_col = params.get('z_column')
        color_col = params.get('color_column')
        
        if not all(col in df.columns for col in [x_col, y_col, z_col]):
            raise ValueError("Invalid column selection for 3D plot")
        
        fig = px.scatter_3d(
            df, 
            x=x_col, 
            y=y_col, 
            z=z_col,
            color=color_col if color_col in df.columns else None,
            title=f'3D Scatter: {x_col}, {y_col}, {z_col}',
            template='plotly_white'
        )
        
        return {
            'success': True,
            'html': fig.to_html(config=self.plotly_config),
            'type': 'scatter_3d',
            'x_column': x_col,
            'y_column': y_col,
            'z_column': z_col
        }
    
    def _create_default_visualization(self, df, params):
        """Create a default overview visualization"""
        # Create a simple data overview
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            fig = px.scatter(
                df, 
                x=numeric_cols[0], 
                y=numeric_cols[1],
                title='Data Overview (First Two Numeric Columns)',
                template='plotly_white'
            )
        else:
            fig = px.histogram(
                df, 
                x=df.columns[0],
                title=f'Distribution of {df.columns[0]}',
                template='plotly_white'
            )
        
        return {
            'success': True,
            'html': fig.to_html(config=self.plotly_config),
            'type': 'default',
            'message': 'Default visualization created'
        }
    
# Global instance
visualization_engine = VisualizationEngine()