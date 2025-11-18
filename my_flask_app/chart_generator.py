"""
Chart Generation Module
Handles the creation of various types of data visualizations using Plotly
"""

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.io as pio
from plotly.offline import plot
import json
import tempfile
import uuid
from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

class ChartGenerator:
    """Handles generation of various chart types using Plotly"""
    
    def __init__(self):
        # Configure Plotly default settings
        pio.templates.default = "plotly_white"
        
    def generate_chart(self, 
                      data: pd.DataFrame, 
                      chart_type: str, 
                      config: Dict[str, Any],
                      export_format: str = 'html') -> Dict[str, Any]:
        """
        Generate a chart based on the specified type and configuration
        
        Args:
            data (pd.DataFrame): The dataset
            chart_type (str): Type of chart to generate
            config (dict): Chart configuration parameters
            export_format (str): Output format ('html', 'png', 'svg', 'pdf')
            
        Returns:
            dict: Result containing chart data and file paths
        """
        try:
            # Validate data
            if data.empty:
                raise ValueError("Dataset is empty")
            
            # Generate the chart based on type
            fig = self._create_chart(data, chart_type, config)
            
            if fig is None:
                raise ValueError(f"Failed to create chart of type: {chart_type}")
            
            # Apply common styling
            self._apply_styling(fig, config)
            
            # Generate output based on format
            result = self._export_chart(fig, export_format, config)
            
            return {
                'success': True,
                'chart_type': chart_type,
                'config': config,
                **result
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'chart_type': chart_type
            }
    
    def _create_chart(self, data: pd.DataFrame, chart_type: str, config: Dict[str, Any]) -> Optional[go.Figure]:
        """Create chart based on type"""
        
        # Basic Charts
        if chart_type == 'line':
            return self._create_line_chart(data, config)
        elif chart_type == 'bar':
            return self._create_bar_chart(data, config)
        elif chart_type == 'stackedBar':
            return self._create_stacked_bar_chart(data, config)
        elif chart_type == 'scatter':
            return self._create_scatter_plot(data, config)
        elif chart_type == 'area':
            return self._create_area_chart(data, config)
        elif chart_type == 'pie':
            return self._create_pie_chart(data, config)
            
        # Statistical Charts
        elif chart_type == 'histogram':
            return self._create_histogram(data, config)
        elif chart_type == 'boxplot':
            return self._create_box_plot(data, config)
        elif chart_type == 'violin':
            return self._create_violin_plot(data, config)
        elif chart_type == 'density':
            return self._create_density_plot(data, config)
            
        # Advanced Charts
        elif chart_type == 'heatmap':
            return self._create_heatmap(data, config)
        elif chart_type == 'correlation':
            return self._create_correlation_matrix(data, config)
        elif chart_type == 'pairplot':
            return self._create_pair_plot(data, config)
        elif chart_type == 'pca':
            return self._create_pca_plot(data, config)
        elif chart_type == 'clustering':
            return self._create_clustering_plot(data, config)
            
        # Time Series Charts
        elif chart_type == 'timeseries':
            return self._create_time_series(data, config)
        elif chart_type == 'movingAverage':
            return self._create_moving_average(data, config)
            
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
    
    def _create_line_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create line chart"""
        x_col = config.get('xColumn')
        y_col = config.get('yColumn')
        color_by = config.get('colorBy')
        
        if not x_col or not y_col:
            raise ValueError("Line chart requires both xColumn and yColumn")
        
        if color_by and color_by in data.columns:
            fig = px.line(data, x=x_col, y=y_col, color=color_by)
        else:
            fig = px.line(data, x=x_col, y=y_col)
        
        return fig
    
    def _create_bar_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create bar chart"""
        x_col = config.get('xColumn')
        y_col = config.get('yColumn')
        
        if not x_col or not y_col:
            raise ValueError("Bar chart requires both xColumn and yColumn")
        
        # Aggregate data if needed
        if data[x_col].dtype == 'object':
            agg_data = data.groupby(x_col)[y_col].sum().reset_index()
        else:
            agg_data = data
        
        fig = px.bar(agg_data, x=x_col, y=y_col)
        return fig
    
    def _create_stacked_bar_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create stacked bar chart"""
        x_col = config.get('xColumn')
        y_col = config.get('yColumn')
        stack_by = config.get('stackBy')
        
        if not x_col or not y_col or not stack_by:
            raise ValueError("Stacked bar chart requires xColumn, yColumn, and stackBy")
        
        fig = px.bar(data, x=x_col, y=y_col, color=stack_by)
        return fig
    
    def _create_scatter_plot(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create scatter plot"""
        x_col = config.get('xColumn')
        y_col = config.get('yColumn')
        size_by = config.get('sizeBy')
        color_by = config.get('colorBy')
        
        if not x_col or not y_col:
            raise ValueError("Scatter plot requires both xColumn and yColumn")
        
        kwargs = {'x': x_col, 'y': y_col}
        if size_by and size_by in data.columns:
            kwargs['size'] = size_by
        if color_by and color_by in data.columns:
            kwargs['color'] = color_by
        
        fig = px.scatter(data, **kwargs)
        return fig
    
    def _create_area_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create area chart"""
        x_col = config.get('xColumn')
        y_col = config.get('yColumn')
        color_by = config.get('colorBy')
        
        if not x_col or not y_col:
            raise ValueError("Area chart requires both xColumn and yColumn")
        
        if color_by and color_by in data.columns:
            fig = px.area(data, x=x_col, y=y_col, color=color_by)
        else:
            fig = px.area(data, x=x_col, y=y_col)
        
        return fig
    
    def _create_pie_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create pie chart"""
        category_col = config.get('categoryColumn')
        value_col = config.get('valueColumn')
        
        if not category_col:
            raise ValueError("Pie chart requires categoryColumn")
        
        # If no value column specified, count occurrences
        if not value_col:
            pie_data = data[category_col].value_counts().reset_index()
            pie_data.columns = [category_col, 'count']
            fig = px.pie(pie_data, names=category_col, values='count')
        else:
            # Aggregate values by category
            pie_data = data.groupby(category_col)[value_col].sum().reset_index()
            fig = px.pie(pie_data, names=category_col, values=value_col)
        
        return fig
    
    def _create_histogram(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create histogram"""
        column = config.get('column')
        bins = config.get('bins', 20)
        
        if not column:
            raise ValueError("Histogram requires a column")
        
        fig = px.histogram(data, x=column, nbins=int(bins))
        return fig
    
    def _create_box_plot(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create box plot"""
        value_col = config.get('valueColumn')
        group_by = config.get('groupBy')
        
        if not value_col:
            raise ValueError("Box plot requires valueColumn")
        
        if group_by and group_by in data.columns:
            fig = px.box(data, y=value_col, x=group_by)
        else:
            fig = px.box(data, y=value_col)
        
        return fig
    
    def _create_violin_plot(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create violin plot"""
        value_col = config.get('valueColumn')
        group_by = config.get('groupBy')
        
        if not value_col:
            raise ValueError("Violin plot requires valueColumn")
        
        if group_by and group_by in data.columns:
            fig = px.violin(data, y=value_col, x=group_by, box=True)
        else:
            fig = px.violin(data, y=value_col, box=True)
        
        return fig
    
    def _create_density_plot(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create density plot using matplotlib and convert to Plotly"""
        column = config.get('column')
        
        if not column:
            raise ValueError("Density plot requires a column")
        
        # Remove NaN values
        clean_data = data[column].dropna()
        
        # Create histogram for density approximation
        fig = px.histogram(data, x=column, marginal="violin", 
                          histnorm='probability density')
        
        return fig
    
    def _create_heatmap(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create heatmap"""
        columns = config.get('columns', [])
        
        if not columns:
            # Use all numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                raise ValueError("Heatmap requires at least 2 numeric columns")
            columns = numeric_cols
        
        # Create correlation matrix if multiple columns
        if len(columns) > 1:
            corr_matrix = data[columns].corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                           color_continuous_scale='RdBu_r', 
                           color_continuous_midpoint=0)
        else:
            raise ValueError("Heatmap requires multiple columns")
        
        return fig
    
    def _create_correlation_matrix(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create correlation matrix"""
        columns = config.get('columns', [])
        
        if not columns:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            columns = numeric_cols
        
        if len(columns) < 2:
            raise ValueError("Correlation matrix requires at least 2 numeric columns")
        
        corr_matrix = data[columns].corr()
        
        fig = px.imshow(corr_matrix, 
                       text_auto=True, 
                       aspect="auto",
                       color_continuous_scale='RdBu_r', 
                       color_continuous_midpoint=0,
                       labels=dict(color="Correlation"))
        
        return fig
    
    def _create_pair_plot(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create pair plot (scatter matrix)"""
        columns = config.get('columns', [])
        
        if not columns:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()[:4]  # Limit for performance
            columns = numeric_cols
        
        if len(columns) < 2:
            raise ValueError("Pair plot requires at least 2 numeric columns")
        
        # Limit to first 4 columns for performance
        columns = columns[:4]
        
        fig = px.scatter_matrix(data, dimensions=columns, 
                               title="Pair Plot (Scatter Matrix)")
        
        return fig
    
    def _create_pca_plot(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create PCA plot"""
        features = config.get('features', [])
        
        if not features:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            features = numeric_cols
        
        if len(features) < 2:
            raise ValueError("PCA requires at least 2 numeric features")
        
        # Prepare data
        X = data[features].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create DataFrame for plotting
        pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        
        # Add explained variance to title
        explained_var = pca.explained_variance_ratio_
        title = f"PCA Plot (PC1: {explained_var[0]:.1%}, PC2: {explained_var[1]:.1%})"
        
        fig = px.scatter(pca_df, x='PC1', y='PC2', title=title)
        
        return fig
    
    def _create_clustering_plot(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create clustering plot"""
        features = config.get('features', [])
        n_clusters = config.get('clusters', 3)
        
        if not features:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            features = numeric_cols[:2]  # Use first 2 for 2D plot
        
        if len(features) < 2:
            raise ValueError("Clustering requires at least 2 numeric features")
        
        # Prepare data
        X = data[features[:2]].fillna(0)  # Use first 2 features for 2D plot
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=int(n_clusters), random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame(X, columns=features[:2])
        plot_df['Cluster'] = clusters.astype(str)
        
        fig = px.scatter(plot_df, x=features[0], y=features[1], color='Cluster',
                        title=f"K-Means Clustering (k={n_clusters})")
        
        return fig
    
    def _create_time_series(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create time series chart"""
        time_col = config.get('timeColumn')
        value_col = config.get('valueColumn')
        
        if not time_col or not value_col:
            raise ValueError("Time series requires both timeColumn and valueColumn")
        
        # Try to convert time column to datetime
        try:
            data[time_col] = pd.to_datetime(data[time_col])
        except:
            pass  # If conversion fails, use as is
        
        # Sort by time column
        sorted_data = data.sort_values(time_col)
        
        fig = px.line(sorted_data, x=time_col, y=value_col,
                     title="Time Series Plot")
        
        return fig
    
    def _create_moving_average(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create moving average chart"""
        time_col = config.get('timeColumn')
        value_col = config.get('valueColumn')
        window = config.get('window', 7)
        
        if not time_col or not value_col:
            raise ValueError("Moving average requires both timeColumn and valueColumn")
        
        # Try to convert time column to datetime
        try:
            data[time_col] = pd.to_datetime(data[time_col])
        except:
            pass
        
        # Sort by time column
        sorted_data = data.sort_values(time_col).copy()
        
        # Calculate moving average
        sorted_data['Moving_Average'] = sorted_data[value_col].rolling(window=int(window)).mean()
        
        # Create figure with both original and moving average
        fig = go.Figure()
        
        # Add original data
        fig.add_trace(go.Scatter(
            x=sorted_data[time_col],
            y=sorted_data[value_col],
            mode='lines+markers',
            name='Original',
            opacity=0.7
        ))
        
        # Add moving average
        fig.add_trace(go.Scatter(
            x=sorted_data[time_col],
            y=sorted_data['Moving_Average'],
            mode='lines',
            name=f'Moving Average ({window})',
            line=dict(width=3)
        ))
        
        fig.update_layout(title=f"Moving Average Plot (Window: {window})")
        
        return fig
    
    def _apply_styling(self, fig: go.Figure, config: Dict[str, Any]):
        """Apply common styling to the figure"""
        title = config.get('title', 'Chart')
        width = config.get('width', 800)
        height = config.get('height', 600)
        theme = config.get('theme', 'default')
        
        # Update layout
        fig.update_layout(
            title=title,
            width=int(width),
            height=int(height),
            title_x=0.5,  # Center title
            font=dict(size=12),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Apply theme-based color schemes
        if theme == 'viridis':
            fig.update_layout(colorway=px.colors.sequential.Viridis)
        elif theme == 'plasma':
            fig.update_layout(colorway=px.colors.sequential.Plasma)
        elif theme == 'inferno':
            fig.update_layout(colorway=px.colors.sequential.Inferno)
        elif theme == 'blues':
            fig.update_layout(colorway=px.colors.sequential.Blues)
        elif theme == 'greens':
            fig.update_layout(colorway=px.colors.sequential.Greens)
    
    def _export_chart(self, fig: go.Figure, export_format: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Export chart in specified format"""
        
        if export_format == 'html':
            # Generate HTML content
            html_content = fig.to_html(include_plotlyjs=True)
            return {
                'html_content': html_content,
                'format': 'html'
            }
        
        elif export_format in ['png', 'svg', 'pdf']:
            # Generate image file
            filename = f"chart_{uuid.uuid4().hex[:8]}.{export_format}"
            filepath = os.path.join(tempfile.gettempdir(), filename)
            
            try:
                if export_format == 'png':
                    fig.write_image(filepath, format='png', engine='kaleido')
                elif export_format == 'svg':
                    fig.write_image(filepath, format='svg', engine='kaleido')
                elif export_format == 'pdf':
                    fig.write_image(filepath, format='pdf', engine='kaleido')
                
                return {
                    'file_path': filepath,
                    'filename': filename,
                    'format': export_format
                }
            except Exception as e:
                # Fallback to HTML if image export fails
                print(f"Image export failed: {e}, falling back to HTML")
                html_content = fig.to_html(include_plotlyjs=True)
                return {
                    'html_content': html_content,
                    'format': 'html',
                    'warning': f'Requested {export_format} format failed, returned HTML instead'
                }
        
        else:
            raise ValueError(f"Unsupported export format: {export_format}")