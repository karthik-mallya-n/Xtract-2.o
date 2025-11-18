'use client';

import { useState, useEffect, useRef } from 'react';
import Image from 'next/image';
import { 
  BarChart3, 
  LineChart, 
  PieChart, 
  TrendingUp, 
  Activity, 
  Map, 
  Upload,
  Download,
  Play,
  Settings,
  Eye,
  Loader2,
  AlertCircle,
  CheckCircle,
  Info,
  Brain,
  Target,
  RotateCcw
} from 'lucide-react';

/**
 * Chart Types Categories
 */
const CHART_CATEGORIES = {
  BASIC: 'Basic Charts',
  STATISTICAL: 'Statistical Visualizations', 
  CATEGORICAL: 'Categorical & Hierarchical',
  TIMESERIES: 'Time-Series Visualizations',
  GEOGRAPHIC: 'Geographic Visualizations',
  ADVANCED: 'Advanced / ML Visualizations',
  RELATIONSHIP: 'Relationship & Network',
  INTERACTIVE: 'Interactive Visualizations'
};

/**
 * Available Chart Types with comprehensive descriptions
 */
const CHART_TYPES = {
  // Basic Charts
  line: {
    id: 'line',
    name: 'Line Chart',
    category: CHART_CATEGORIES.BASIC,
    icon: LineChart,
    description: 'Shows trends over time (time-series data)',
    bestFor: 'Continuous data, trends, time series',
    dataRequirements: 'Numeric data with time/sequence column'
  },
  bar: {
    id: 'bar',
    name: 'Bar Chart',
    category: CHART_CATEGORIES.BASIC,
    icon: BarChart3,
    description: 'Compare quantities across categories',
    bestFor: 'Categorical comparisons, ranking',
    dataRequirements: 'Categorical and numeric columns'
  },
  stackedBar: {
    id: 'stackedBar',
    name: 'Stacked Bar Chart',
    category: CHART_CATEGORIES.BASIC,
    icon: BarChart3,
    description: 'Show composition of categories',
    bestFor: 'Part-to-whole relationships',
    dataRequirements: 'Multiple numeric columns per category'
  },
  scatter: {
    id: 'scatter',
    name: 'Scatter Plot',
    category: CHART_CATEGORIES.BASIC,
    icon: Target,
    description: 'Show correlation between two numeric variables',
    bestFor: 'Correlation analysis, outlier detection',
    dataRequirements: 'Two numeric columns'
  },
  area: {
    id: 'area',
    name: 'Area Chart',
    category: CHART_CATEGORIES.BASIC,
    icon: Activity,
    description: 'Like line chart with filled area; shows cumulative totals',
    bestFor: 'Cumulative data, volume over time',
    dataRequirements: 'Numeric data over time/sequence'
  },
  pie: {
    id: 'pie',
    name: 'Pie/Donut Chart',
    category: CHART_CATEGORIES.BASIC,
    icon: PieChart,
    description: 'Shows proportions of a whole',
    bestFor: 'Proportional data (limited categories)',
    dataRequirements: 'Categorical data with counts/values'
  },
  
  // Statistical Visualizations
  histogram: {
    id: 'histogram',
    name: 'Histogram',
    category: CHART_CATEGORIES.STATISTICAL,
    icon: BarChart3,
    description: 'Distribution of one numeric variable',
    bestFor: 'Data distribution analysis',
    dataRequirements: 'Single numeric column'
  },
  boxplot: {
    id: 'boxplot',
    name: 'Box Plot',
    category: CHART_CATEGORIES.STATISTICAL,
    icon: Target,
    description: 'Shows quartiles, median, and outliers',
    bestFor: 'Distribution summary, outlier detection',
    dataRequirements: 'Numeric data (optionally grouped)'
  },
  violin: {
    id: 'violin',
    name: 'Violin Plot',
    category: CHART_CATEGORIES.STATISTICAL,
    icon: Activity,
    description: 'Distribution + density in one chart',
    bestFor: 'Detailed distribution analysis',
    dataRequirements: 'Numeric data with grouping'
  },
  density: {
    id: 'density',
    name: 'Density Plot (KDE)',
    category: CHART_CATEGORIES.STATISTICAL,
    icon: TrendingUp,
    description: 'Smoothed probability distribution',
    bestFor: 'Continuous distribution visualization',
    dataRequirements: 'Numeric data'
  },
  
  // Categorical & Hierarchical
  treemap: {
    id: 'treemap',
    name: 'Treemap',
    category: CHART_CATEGORIES.CATEGORICAL,
    icon: Target,
    description: 'Nested categories displayed proportionally',
    bestFor: 'Hierarchical data with sizes',
    dataRequirements: 'Hierarchical categorical data with values'
  },
  sunburst: {
    id: 'sunburst',
    name: 'Sunburst Diagram',
    category: CHART_CATEGORIES.CATEGORICAL,
    icon: Target,
    description: 'Hierarchy shown in circular layers',
    bestFor: 'Multi-level categorical data',
    dataRequirements: 'Hierarchical structure with values'
  },
  heatmap: {
    id: 'heatmap',
    name: 'Heatmap',
    category: CHART_CATEGORIES.CATEGORICAL,
    icon: Target,
    description: 'Shows values using color (e.g., correlation matrix)',
    bestFor: 'Matrix data, correlations',
    dataRequirements: 'Matrix-like numeric data'
  },
  mosaic: {
    id: 'mosaic',
    name: 'Mosaic Plot',
    category: CHART_CATEGORIES.CATEGORICAL,
    icon: Target,
    description: 'Shows relationships between categorical variables',
    bestFor: 'Categorical variable relationships',
    dataRequirements: 'Multiple categorical columns'
  },
  
  // Time-Series
  timeseries: {
    id: 'timeseries',
    name: 'Time-Series Line Chart',
    category: CHART_CATEGORIES.TIMESERIES,
    icon: LineChart,
    description: 'Trend of a metric over time',
    bestFor: 'Time-based trends and patterns',
    dataRequirements: 'Date/time column with numeric values'
  },
  movingAverage: {
    id: 'movingAverage',
    name: 'Moving Average Chart',
    category: CHART_CATEGORIES.TIMESERIES,
    icon: TrendingUp,
    description: 'Smooths noisy data',
    bestFor: 'Trend identification in noisy data',
    dataRequirements: 'Time series with numeric values'
  },
  candlestick: {
    id: 'candlestick',
    name: 'Candlestick Chart',
    category: CHART_CATEGORIES.TIMESERIES,
    icon: BarChart3,
    description: 'Finance/stock trading data',
    bestFor: 'OHLC (Open, High, Low, Close) data',
    dataRequirements: 'Date, Open, High, Low, Close columns'
  },
  seasonal: {
    id: 'seasonal',
    name: 'Seasonal Decomposition',
    category: CHART_CATEGORIES.TIMESERIES,
    icon: Activity,
    description: 'Break trends into seasonal + residual components',
    bestFor: 'Seasonal pattern analysis',
    dataRequirements: 'Long time series data'
  },
  
  // Geographic
  choropleth: {
    id: 'choropleth',
    name: 'Choropleth Map',
    category: CHART_CATEGORIES.GEOGRAPHIC,
    icon: Map,
    description: 'Color-coded regions (states, countries, etc.)',
    bestFor: 'Regional data comparison',
    dataRequirements: 'Geographic regions with values'
  },
  bubbleMap: {
    id: 'bubbleMap',
    name: 'Bubble Map',
    category: CHART_CATEGORIES.GEOGRAPHIC,
    icon: Map,
    description: 'Points sized by value on a map',
    bestFor: 'Location-based data with magnitudes',
    dataRequirements: 'Latitude, longitude, and size values'
  },
  heatmapMap: {
    id: 'heatmapMap',
    name: 'Geographic Heatmap',
    category: CHART_CATEGORIES.GEOGRAPHIC,
    icon: Map,
    description: 'Density of points on a geographic map',
    bestFor: 'Point density visualization',
    dataRequirements: 'Latitude and longitude coordinates'
  },
  
  // Advanced/ML
  correlation: {
    id: 'correlation',
    name: 'Correlation Matrix',
    category: CHART_CATEGORIES.ADVANCED,
    icon: Target,
    description: 'Shows relationships between variables',
    bestFor: 'Feature relationship analysis',
    dataRequirements: 'Multiple numeric columns'
  },
  pairplot: {
    id: 'pairplot',
    name: 'Pair Plot (Scatter Matrix)',
    category: CHART_CATEGORIES.ADVANCED,
    icon: Target,
    description: 'Scatter plots for all variable combinations',
    bestFor: 'Comprehensive variable analysis',
    dataRequirements: 'Multiple numeric columns'
  },
  pca: {
    id: 'pca',
    name: 'PCA Plot',
    category: CHART_CATEGORIES.ADVANCED,
    icon: Brain,
    description: 'Reduces high-dimensional data to 2D',
    bestFor: 'Dimensionality reduction visualization',
    dataRequirements: 'Multiple numeric features'
  },
  clustering: {
    id: 'clustering',
    name: 'Clustering Plot (K-Means)',
    category: CHART_CATEGORIES.ADVANCED,
    icon: Target,
    description: 'Shows clusters in your data',
    bestFor: 'Cluster analysis and segmentation',
    dataRequirements: 'Numeric features for clustering'
  },
  
  // Relationship & Network
  network: {
    id: 'network',
    name: 'Network Graph',
    category: CHART_CATEGORIES.RELATIONSHIP,
    icon: Target,
    description: 'If your CSV represents connections (edges/nodes)',
    bestFor: 'Relationship networks, graph data',
    dataRequirements: 'Source-Target relationship data'
  },
  sankey: {
    id: 'sankey',
    name: 'Sankey Diagram',
    category: CHART_CATEGORIES.RELATIONSHIP,
    icon: Activity,
    description: 'Flow of values between stages',
    bestFor: 'Flow analysis, process visualization',
    dataRequirements: 'Source-Target-Value relationship data'
  },
  chord: {
    id: 'chord',
    name: 'Chord Diagram',
    category: CHART_CATEGORIES.RELATIONSHIP,
    icon: Target,
    description: 'Shows relationships between categories',
    bestFor: 'Circular relationship visualization',
    dataRequirements: 'Matrix of relationships between entities'
  }
};

/**
 * File Upload Component
 */
interface FileUploadProps {
  onFileSelect: (file: File) => void;
  isLoading: boolean;
}

function FileUpload({ onFileSelect, isLoading }: FileUploadProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onFileSelect(e.target.files[0]);
    }
  };

  return (
    <div className="border-2 border-dashed border-gray-600 bg-gray-800/30 backdrop-blur-xl rounded-lg p-8 text-center hover:border-cyan-400 transition-colors">
      <input
        ref={fileInputRef}
        type="file"
        accept=".csv,.xlsx,.json"
        onChange={handleFileChange}
        className="hidden"
        disabled={isLoading}
      />
      
      <div className="space-y-4">
        <div className="mx-auto w-16 h-16 bg-cyan-900/30 rounded-full flex items-center justify-center">
          {isLoading ? (
            <Loader2 className="h-8 w-8 text-cyan-400 animate-spin" />
          ) : (
            <Upload className="h-8 w-8 text-cyan-400" />
          )}
        </div>
        
        <div>
          <h3 className="text-lg font-semibold text-white mb-2">
            {isLoading ? 'Processing File...' : 'Upload Your Dataset'}
          </h3>
          <p className="text-gray-400 mb-4">
            Support for CSV, Excel (.xlsx), and JSON files
          </p>
          
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isLoading}
            className={`px-6 py-3 rounded-lg font-medium transition-colors ${
              isLoading
                ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                : 'bg-gradient-to-r from-cyan-500 to-blue-500 text-white hover:from-cyan-600 hover:to-blue-600'
            }`}
          >
            {isLoading ? 'Processing...' : 'Choose File'}
          </button>
        </div>
        
        <p className="text-sm text-gray-500">
          Maximum file size: 50MB
        </p>
      </div>
    </div>
  );
}

/**
 * Chart Type Selector Component
 */
interface ChartTypeSelectorProps {
  selectedCategory: string;
  selectedChart: string;
  onCategoryChange: (category: string) => void;
  onChartChange: (chartId: string) => void;
  dataColumns: string[];
}

function ChartTypeSelector({ 
  selectedCategory, 
  selectedChart, 
  onCategoryChange, 
  onChartChange,
  dataColumns 
}: ChartTypeSelectorProps) {
  const categories = Object.values(CHART_CATEGORIES);
  const chartsInCategory = Object.values(CHART_TYPES).filter(
    chart => chart.category === selectedCategory
  );

  return (
    <div className="space-y-6">
      {/* Category Selection */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-3">
          Chart Category
        </label>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          {categories.map((category) => (
            <button
              key={category}
              onClick={() => onCategoryChange(category)}
              className={`p-3 text-sm rounded-lg border transition-colors ${
                selectedCategory === category
                  ? 'bg-gradient-to-r from-cyan-500 to-blue-500 text-white border-cyan-500 shadow-lg'
                  : 'bg-gray-800/30 text-gray-300 border-gray-600 hover:bg-gray-700/50'
              }`}
            >
              {category}
            </button>
          ))}
        </div>
      </div>

      {/* Chart Type Selection */}
      {selectedCategory && (
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-3">
            Chart Type
          </label>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {chartsInCategory.map((chart) => {
              const Icon = chart.icon;
              return (
                <button
                  key={chart.id}
                  onClick={() => onChartChange(chart.id)}
                  className={`p-4 text-left rounded-lg border transition-all ${
                    selectedChart === chart.id
                      ? 'bg-cyan-900/30 border-cyan-400 shadow-md'
                      : 'bg-gray-800/30 border-gray-600 hover:bg-gray-700/50'
                  }`}
                >
                  <div className="flex items-start space-x-3">
                    <Icon className={`h-5 w-5 mt-1 ${
                      selectedChart === chart.id ? 'text-cyan-400' : 'text-gray-400'
                    }`} />
                    <div className="flex-1">
                      <h4 className={`font-medium ${
                        selectedChart === chart.id ? 'text-cyan-300' : 'text-white'
                      }`}>
                        {chart.name}
                      </h4>
                      <p className="text-sm text-gray-400 mt-1">
                        {chart.description}
                      </p>
                      <p className="text-xs text-gray-500 mt-2">
                        Best for: {chart.bestFor}
                      </p>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Chart Requirements */}
      {selectedChart && (selectedChart in CHART_TYPES) && (
        <div className="bg-cyan-900/20 border border-cyan-500/30 rounded-lg p-4 backdrop-blur-xl">
          <div className="flex items-start space-x-2">
            <Info className="h-5 w-5 text-cyan-400 mt-0.5" />
            <div>
              <h4 className="font-medium text-cyan-300 mb-1">Data Requirements</h4>
              <p className="text-sm text-cyan-200">
                {CHART_TYPES[selectedChart as keyof typeof CHART_TYPES].dataRequirements}
              </p>
              {dataColumns.length > 0 && (
                <div className="mt-2">
                  <p className="text-xs text-blue-700 mb-1">Available columns:</p>
                  <div className="flex flex-wrap gap-1">
                    {dataColumns.slice(0, 10).map((col) => (
                      <span key={col} className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                        {col}
                      </span>
                    ))}
                    {dataColumns.length > 10 && (
                      <span className="text-xs text-blue-600">+{dataColumns.length - 10} more</span>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

type ChartConfigValue = string | number | string[] | boolean;
type ChartConfig = Record<string, ChartConfigValue>;

/**
 * Chart Configuration Component
 */
interface ChartConfigProps {
  chartType: string;
  dataColumns: string[];
  config: ChartConfig;
  onConfigChange: (config: ChartConfig) => void;
  datasetInfo?: {
    total_rows: number; 
    total_columns: number; 
    columns: string[];
    numeric_columns: string[]; 
    categorical_columns: string[];
    datetime_columns: string[];
    sample_data: Array<Record<string, string | number | null>>;
    data_types: Record<string, string>;
  } | null;
}

function ChartConfig({ chartType, dataColumns, config, onConfigChange, datasetInfo }: ChartConfigProps) {
  if (!chartType || !(chartType in CHART_TYPES)) return null;

  const updateConfig = (key: string, value: ChartConfigValue) => {
    onConfigChange({ ...config, [key]: value });
  };

  const renderConfigField = (key: string, label: string, type: string, options?: string[]) => (
    <div key={key} className="mb-4">
      <label className="block text-sm font-medium text-gray-300 mb-2">
        {label}
      </label>
      {type === 'select' && options ? (
        <select
          value={String(config[key] || '')}
          onChange={(e) => updateConfig(key, e.target.value)}
          className="w-full px-3 py-2 border border-gray-600 bg-gray-800 text-white rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
        >
          <option value="">Select {label.toLowerCase()}</option>
          {options.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      ) : type === 'multiselect' && options ? (
        <div className="space-y-2 max-h-32 overflow-y-auto border border-gray-600 bg-gray-800 rounded-md p-2">
          {options.map((option) => {
            const currentValues = Array.isArray(config[key]) ? config[key] as string[] : [];
            return (
            <label key={option} className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={currentValues.includes(option)}
                onChange={(e) => {
                  const newValues = e.target.checked
                    ? [...currentValues, option]
                    : currentValues.filter((v: string) => v !== option);
                  updateConfig(key, newValues);
                }}
                className="rounded text-cyan-600 focus:ring-cyan-500 bg-gray-700 border-gray-600"
              />
              <span className="text-sm text-gray-300">{option}</span>
            </label>
          )})}
        </div>
      ) : (
        <input
          type={type}
          value={String(config[key] || '')}
          onChange={(e) => updateConfig(key, type === 'number' ? parseFloat(e.target.value) : e.target.value)}
          placeholder={`Enter ${label.toLowerCase()}`}
          className="w-full px-3 py-2 border border-gray-600 bg-gray-800 text-white rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500 placeholder-gray-400"
        />
      )}
    </div>
  );

  // Configuration fields based on chart type
  const getConfigFields = () => {
    const numericColumns = dataColumns.filter(col => {
      // Check if we have dataset info to determine column types
      if (datasetInfo && datasetInfo.numeric_columns) {
        return datasetInfo.numeric_columns.includes(col);
      }
      // Fallback: exclude obvious non-numeric columns
      return !['id', 'index', 'name'].includes(col.toLowerCase());
    });
    
    const categoricalColumns = dataColumns.filter(col => {
      if (datasetInfo && datasetInfo.categorical_columns) {
        return datasetInfo.categorical_columns.includes(col);
      }
      return true; // Fallback: assume all can be categorical
    });
    
    const timeColumns = dataColumns.filter(col => {
      if (datasetInfo && datasetInfo.datetime_columns) {
        return datasetInfo.datetime_columns.includes(col);
      }
      // Fallback: look for common time column names
      return col.toLowerCase().includes('date') || 
             col.toLowerCase().includes('time') || 
             col.toLowerCase().includes('year');
    });

    switch (chartType) {
      case 'line':
      case 'area':
        return (
          <>
            {renderConfigField('xColumn', 'X-Axis Column', 'select', dataColumns)}
            {renderConfigField('yColumn', 'Y-Axis Column', 'select', numericColumns)}
            {renderConfigField('colorBy', 'Color By (optional)', 'select', categoricalColumns)}
            {renderConfigField('title', 'Chart Title', 'text')}
          </>
        );
      
      case 'bar':
      case 'stackedBar':
        return (
          <>
            {renderConfigField('xColumn', 'Category Column', 'select', categoricalColumns)}
            {renderConfigField('yColumn', 'Value Column', 'select', numericColumns)}
            {chartType === 'stackedBar' && renderConfigField('stackBy', 'Stack By', 'select', categoricalColumns)}
            {renderConfigField('title', 'Chart Title', 'text')}
          </>
        );
      
      case 'scatter':
        return (
          <>
            {renderConfigField('xColumn', 'X-Axis Column', 'select', numericColumns)}
            {renderConfigField('yColumn', 'Y-Axis Column', 'select', numericColumns)}
            {renderConfigField('sizeBy', 'Size By (optional)', 'select', numericColumns)}
            {renderConfigField('colorBy', 'Color By (optional)', 'select', categoricalColumns)}
            {renderConfigField('title', 'Chart Title', 'text')}
          </>
        );
      
      case 'pie':
        return (
          <>
            {renderConfigField('categoryColumn', 'Category Column', 'select', categoricalColumns)}
            {renderConfigField('valueColumn', 'Value Column', 'select', numericColumns)}
            {renderConfigField('title', 'Chart Title', 'text')}
          </>
        );
      
      case 'histogram':
        return (
          <>
            {renderConfigField('column', 'Numeric Column', 'select', numericColumns)}
            {renderConfigField('bins', 'Number of Bins', 'number')}
            {renderConfigField('title', 'Chart Title', 'text')}
          </>
        );
      
      case 'boxplot':
      case 'violin':
        return (
          <>
            {renderConfigField('valueColumn', 'Value Column', 'select', numericColumns)}
            {renderConfigField('groupBy', 'Group By (optional)', 'select', categoricalColumns)}
            {renderConfigField('title', 'Chart Title', 'text')}
          </>
        );
      
      case 'heatmap':
        return (
          <>
            {renderConfigField('columns', 'Columns for Heatmap', 'multiselect', numericColumns)}
            {renderConfigField('title', 'Chart Title', 'text')}
          </>
        );
      
      case 'timeseries':
      case 'movingAverage':
        return (
          <>
            {renderConfigField('timeColumn', 'Time Column', 'select', timeColumns.length > 0 ? timeColumns : dataColumns)}
            {renderConfigField('valueColumn', 'Value Column', 'select', numericColumns)}
            {chartType === 'movingAverage' && renderConfigField('window', 'Moving Average Window', 'number')}
            {renderConfigField('title', 'Chart Title', 'text')}
          </>
        );
      
      case 'correlation':
      case 'pairplot':
        return (
          <>
            {renderConfigField('columns', 'Columns for Analysis', 'multiselect', numericColumns)}
            {renderConfigField('title', 'Chart Title', 'text')}
          </>
        );
      
      case 'clustering':
        return (
          <>
            {renderConfigField('features', 'Feature Columns', 'multiselect', numericColumns)}
            {renderConfigField('clusters', 'Number of Clusters', 'number')}
            {renderConfigField('title', 'Chart Title', 'text')}
          </>
        );
      
      case 'choropleth':
        return (
          <>
            {renderConfigField('locationColumn', 'Location Column', 'select', categoricalColumns)}
            {renderConfigField('valueColumn', 'Value Column', 'select', numericColumns)}
            {renderConfigField('title', 'Chart Title', 'text')}
          </>
        );
      
      case 'bubbleMap':
      case 'heatmapMap':
        return (
          <>
            {renderConfigField('latColumn', 'Latitude Column', 'select', numericColumns)}
            {renderConfigField('lonColumn', 'Longitude Column', 'select', numericColumns)}
            {chartType === 'bubbleMap' && renderConfigField('sizeColumn', 'Size Column', 'select', numericColumns)}
            {renderConfigField('title', 'Chart Title', 'text')}
          </>
        );
      
      default:
        return (
          <>
            {renderConfigField('title', 'Chart Title', 'text')}
          </>
        );
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
      <div className="flex items-center mb-6">
        <Settings className="h-6 w-6 text-blue-600 mr-3" />
        <h3 className="text-xl font-bold text-gray-900">Chart Configuration</h3>
      </div>
      
      {getConfigFields()}
      
      <div className="pt-4 border-t border-gray-200">
        <h4 className="font-medium text-gray-900 mb-3">Display Options</h4>
        {renderConfigField('width', 'Chart Width (px)', 'number')}
        {renderConfigField('height', 'Chart Height (px)', 'number')}
        {renderConfigField('theme', 'Color Theme', 'select', ['default', 'viridis', 'plasma', 'inferno', 'blues', 'greens'])}
      </div>
    </div>
  );
}

/**
 * Chart Preview Component
 */
interface ChartPreviewProps {
  chartType: string;
  config: ChartConfig;
  isGenerating: boolean;
  onGenerate: () => void;
  generatedChart?: {
    imageUrl?: string;
    htmlContent?: string;
    chartType: string;
    config: ChartConfig;
    timestamp: string;
  } | null;
}

function ChartPreview({ chartType, config, isGenerating, onGenerate, generatedChart }: ChartPreviewProps) {
  const chartInfo = chartType in CHART_TYPES ? CHART_TYPES[chartType as keyof typeof CHART_TYPES] : null;
  
  if (!chartInfo) {
    return (
      <div className="bg-gray-50 rounded-lg p-8 text-center">
        <BarChart3 className="h-16 w-16 text-gray-400 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-gray-600 mb-2">
          Select a chart type to get started
        </h3>
        <p className="text-gray-500">
          Choose from our comprehensive collection of visualization options
        </p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md border border-gray-200">
      {/* Header */}
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <chartInfo.icon className="h-6 w-6 text-blue-600" />
            <div>
              <h3 className="text-xl font-bold text-gray-900">{chartInfo.name}</h3>
              <p className="text-sm text-gray-600">{chartInfo.description}</p>
            </div>
          </div>
          
          <button
            onClick={onGenerate}
            disabled={isGenerating}
            className={`px-6 py-3 rounded-lg font-medium transition-colors flex items-center space-x-2 ${
              isGenerating
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-blue-600 text-white hover:bg-blue-700'
            }`}
          >
            {isGenerating ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>Generating...</span>
              </>
            ) : (
              <>
                <Play className="h-4 w-4" />
                <span>Generate Chart</span>
              </>
            )}
          </button>
        </div>
      </div>
      
      {/* Preview Area */}
      <div className="p-6">
        <div 
          className="w-full bg-gray-50 rounded-lg overflow-hidden"
          style={{ 
            height: typeof config.height === 'number' ? config.height : 400,
            minHeight: '300px'
          }}
        >
          {isGenerating ? (
            <div className="text-center flex items-center justify-center h-full">
              <div>
                <Loader2 className="h-12 w-12 text-blue-600 animate-spin mx-auto mb-4" />
                <p className="text-lg font-medium text-gray-700">Generating your visualization...</p>
                <p className="text-sm text-gray-500 mt-2">This may take a few moments</p>
              </div>
            </div>
          ) : generatedChart ? (
            <div className="w-full h-full">
              {generatedChart.htmlContent ? (
                <iframe
                  srcDoc={generatedChart.htmlContent}
                  className="w-full h-full border-0"
                  style={{ minHeight: '400px' }}
                  sandbox="allow-scripts allow-same-origin"
                  title={`Generated ${chartType} chart preview`}
                  onLoad={() => console.log('Chart iframe loaded successfully')}
                  onError={(e) => {
                    console.error('Chart iframe failed to load:', e);
                  }}
                />
              ) : generatedChart.imageUrl ? (
                <div className="relative w-full h-full">
                  <Image 
                    src={generatedChart.imageUrl} 
                    alt={`Generated ${chartType} chart`}
                    fill
                    className="object-contain"
                    onError={(e) => {
                      console.error('Image failed to load:', generatedChart.imageUrl);
                      const target = e.target as HTMLImageElement;
                      target.style.display = 'none';
                    }}
                  />
                </div>
              ) : (
                <div className="text-center flex items-center justify-center h-full">
                  <div>
                    <CheckCircle className="h-12 w-12 text-green-400 mx-auto mb-4" />
                    <p className="text-lg font-medium text-green-700">Chart Generated Successfully</p>
                    <p className="text-sm text-gray-500 mt-2">Chart content available for export</p>
                    <div className="mt-4 px-4 py-2 bg-green-100 rounded-lg">
                      <p className="text-xs text-green-800">Generated at: {new Date(generatedChart.timestamp).toLocaleString()}</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-center flex items-center justify-center h-full">
              <div>
                <Eye className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <p className="text-lg font-medium text-gray-600">Chart Preview</p>
                <p className="text-sm text-gray-500 mt-2">
                  {String(config.title || `${chartInfo.name} will appear here`)}
                </p>
              </div>
            </div>
          )}
        </div>
        
        {/* Chart Configuration Summary */}
        {Object.keys(config).length > 0 && (
          <div className="mt-6 p-4 bg-gray-50 rounded-lg">
            <h4 className="font-medium text-gray-900 mb-3">Configuration Summary</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
              {Object.entries(config).map(([key, value]) => (
                <div key={key}>
                  <span className="font-medium text-gray-700 capitalize">
                    {key.replace(/([A-Z])/g, ' $1').trim()}:
                  </span>
                  <span className="ml-2 text-gray-600">
                    {Array.isArray(value) ? value.join(', ') : String(value)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Main Visualization Page Component
 */
export default function VisualizationPage() {
  const [currentStep, setCurrentStep] = useState<'upload' | 'configure' | 'preview'>('upload');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [dataColumns, setDataColumns] = useState<string[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string>(CHART_CATEGORIES.BASIC);
  const [selectedChart, setSelectedChart] = useState<string>('');
  const [chartConfig, setChartConfig] = useState<ChartConfig>({});
  const [isProcessingFile, setIsProcessingFile] = useState(false);
  const [isGeneratingChart, setIsGeneratingChart] = useState(false);
  const [dataPreview, setDataPreview] = useState<Array<Record<string, string | number>>>([]);
  const [error, setError] = useState<string>('');
  const [fileId, setFileId] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [datasetInfo, setDatasetInfo] = useState<{
    total_rows: number; 
    total_columns: number; 
    columns: string[];
    numeric_columns: string[]; 
    categorical_columns: string[];
    datetime_columns: string[];
    sample_data: Array<Record<string, string | number | null>>;
    data_types: Record<string, string>;
  } | null>(null);
  const [isLoadingData, setIsLoadingData] = useState(true);
  const [generatedChart, setGeneratedChart] = useState<{
    imageUrl?: string;
    htmlContent?: string;
    chartType: string;
    config: ChartConfig;
    timestamp: string;
  } | null>(null);
  const [exportFormat, setExportFormat] = useState<'png' | 'svg' | 'html' | 'pdf'>('png');
  
  // Add request deduplication to prevent duplicate API calls in StrictMode
  const isDataRequestInProgress = useRef(false);

  // Check for existing uploaded data on component mount
  useEffect(() => {
    const checkExistingData = async () => {
      // Prevent duplicate requests (especially in React StrictMode)
      if (isDataRequestInProgress.current) {
        return;
      }
      
      // Mark request as in progress
      isDataRequestInProgress.current = true;
      
      try {
        const storedFileId = localStorage.getItem('currentFileId');
        const storedFileName = localStorage.getItem('currentFileName');
        
        if (storedFileId && storedFileName) {
          setFileId(storedFileId);
          setFileName(storedFileName);
          
          // Fetch dataset information from backend
          const response = await fetch(`http://localhost:5000/api/dataset/${storedFileId}`);
          if (response.ok) {
            const data = await response.json();
            
            if (data.success && data.dataset_info) {
              setDatasetInfo(data.dataset_info);
              
              // Extract column information
              const allColumns = data.dataset_info.columns || [];
              setDataColumns(allColumns);
              setCurrentStep('configure');
              
              // Use real sample data from backend
              setDataPreview(data.dataset_info.sample_data || []);
            } else {
              setError('Failed to load dataset information');
            }
          } else {
            setError('Failed to connect to backend. Please ensure the Flask server is running.');
          }
        }
      } catch (err) {
        console.error('Error loading existing data:', err);
        setError('Failed to load existing data');
      } finally {
        setIsLoadingData(false);
        // Reset the flag when request completes
        isDataRequestInProgress.current = false;
      }
    };
    
    checkExistingData();
  }, []);

  // Handle file upload
  const handleFileSelect = async (file: File) => {
    setIsProcessingFile(true);
    setError('');
    
    try {
      // Simulate file processing
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Mock data columns based on file type
      let mockColumns: string[] = [];
      
      if (file.name.toLowerCase().includes('iris')) {
        mockColumns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'];
      } else if (file.name.toLowerCase().includes('sales')) {
        mockColumns = ['Date', 'Product', 'Category', 'Sales', 'Profit', 'Quantity', 'Region'];
      } else if (file.name.toLowerCase().includes('stock')) {
        mockColumns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol'];
      } else {
        // Generic columns for any CSV
        mockColumns = ['ID', 'Date', 'Category', 'Value1', 'Value2', 'Value3', 'Label'];
      }
      
      setUploadedFile(file);
      setDataColumns(mockColumns);
      setCurrentStep('configure');
      
      // Mock data preview
      setDataPreview([
        { ID: 1, Date: '2024-01-01', Category: 'A', Value1: 5.1, Value2: 3.5, Value3: 1.4, Label: 'Type 1' },
        { ID: 2, Date: '2024-01-02', Category: 'B', Value1: 4.9, Value2: 3.0, Value3: 1.4, Label: 'Type 1' },
        { ID: 3, Date: '2024-01-03', Category: 'C', Value1: 4.7, Value2: 3.2, Value3: 1.3, Label: 'Type 2' },
      ]);
      
    } catch (error) {
      console.error('File processing error:', error);
      setError('Failed to process file. Please try again.');
    } finally {
      setIsProcessingFile(false);
    }
  };

  // Handle chart generation
  const handleGenerateChart = async () => {
    if (!fileId && !uploadedFile) {
      setError('No file available for chart generation');
      return;
    }

    if (!selectedChart || Object.keys(chartConfig).length === 0) {
      setError('Please configure your chart before generating');
      return;
    }

    setIsGeneratingChart(true);
    setError('');
    
    try {
      // Prepare chart generation request
      const requestData = {
        file_id: fileId,
        chart_type: selectedChart,
        config: {
          ...chartConfig,
          width: chartConfig.width || 800,
          height: chartConfig.height || 600,
          title: chartConfig.title || `${CHART_TYPES[selectedChart as keyof typeof CHART_TYPES]?.name || 'Chart'}`,
          theme: chartConfig.theme || 'default'
        },
        export_format: 'html' // Start with HTML for preview
      };

      console.log('Generating chart with:', requestData);
      console.log('⏱️ Chart generation started at:', new Date().toLocaleTimeString());

      // Call backend API for chart generation with timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 120000); // 2 minute timeout
      
      try {
        const response = await fetch('http://localhost:5000/api/generate-chart', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(requestData),
          signal: controller.signal
        });
        
        clearTimeout(timeoutId);

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
        }

        const result = await response.json();
        console.log('📊 Chart generation result:', result);

        if (result.success) {
          const chartData = {
            imageUrl: result.image_url,
            htmlContent: result.html_content,
            chartType: selectedChart,
            config: chartConfig,
            timestamp: new Date().toISOString()
          };
          
          console.log('✅ Chart data prepared:', {
            hasImageUrl: !!chartData.imageUrl,
            hasHtmlContent: !!chartData.htmlContent,
            htmlContentLength: chartData.htmlContent?.length || 0
          });
          
          setGeneratedChart(chartData);
          setCurrentStep('preview');
          
          if (result.warning) {
            console.warn('⚠️ Chart generation warning:', result.warning);
          }
        } else {
          throw new Error(result.error || 'Chart generation failed');
        }
      } catch (error: unknown) {
        clearTimeout(timeoutId);
        if (error instanceof Error && error.name === 'AbortError') {
          throw new Error('Chart generation timed out. Please try again with a smaller dataset or simpler chart type.');
        }
        throw error;
      }
      
    } catch (error) {
      console.error('Chart generation error:', error);
      setError(`Failed to generate chart: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsGeneratingChart(false);
    }
  };

  // Handle chart export
  const handleExportChart = async () => {
    if (!generatedChart || !fileId) {
      setError('No chart available for export');
      return;
    }

    try {
      const requestData = {
        file_id: fileId,
        chart_type: generatedChart.chartType,
        config: generatedChart.config,
        export_format: exportFormat
      };

      const response = await fetch('http://localhost:5000/api/export-chart', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Handle file download
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      
      // Set filename with proper extension
      const filename = `chart_${generatedChart.chartType}_${Date.now()}.${exportFormat}`;
      a.download = filename;
      
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
    } catch (error) {
      console.error('Export error:', error);
      setError(`Failed to export chart: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  // Reset to start
  const handleReset = () => {
    setCurrentStep('upload');
    setUploadedFile(null);
    setDataColumns([]);
    setSelectedCategory(CHART_CATEGORIES.BASIC);
    setSelectedChart('');
    setChartConfig({});
    setDataPreview([]);
    setError('');
    setFileId(null);
    setFileName(null);
    setDatasetInfo(null);
    setIsLoadingData(false);
    setGeneratedChart(null);
    setExportFormat('png');
  };

  return (
    <div className="min-h-screen relative overflow-hidden bg-gray-900">
      {/* Background Elements */}
      <div className="absolute inset-0 bg-gradient-to-br from-gray-900 via-purple-900 to-cyan-900 opacity-20" />
      <div className="absolute inset-0 geometric-pattern opacity-30" />
      
      <div className="relative z-10" style={{paddingTop: '164px', paddingBottom: '48px'}}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Header */}
          <div className="text-center mb-12">
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-white mb-4">
              Data Visualization Studio
            </h1>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto mb-8">
              Create stunning visualizations from your data with our comprehensive collection of charts and graphs.
              From basic plots to advanced ML visualizations.
            </p>
            
            {/* Step Indicator */}
            <div className="flex justify-center space-x-4 mb-8">
              <div className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
                currentStep === 'upload' 
                  ? 'bg-gradient-to-r from-cyan-500 to-blue-500 text-white shadow-lg'
                  : 'bg-gray-800/50 text-gray-400 border border-gray-700'
              }`}>
                <Upload className="h-4 w-4" />
                <span>Upload Data</span>
              </div>
              <div className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
                currentStep === 'configure' 
                  ? 'bg-gradient-to-r from-cyan-500 to-blue-500 text-white shadow-lg'
                  : 'bg-gray-800/50 text-gray-400 border border-gray-700'
              }`}>
                <Settings className="h-4 w-4" />
                <span>Configure Chart</span>
              </div>
              <div className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
                currentStep === 'preview' 
                  ? 'bg-gradient-to-r from-cyan-500 to-blue-500 text-white shadow-lg'
                  : 'bg-gray-800/50 text-gray-400 border border-gray-700'
              }`}>
                <Eye className="h-4 w-4" />
                <span>Preview & Export</span>
              </div>
            </div>
          </div>

        {/* Error Display */}
        {error && (
          <div className="max-w-4xl mx-auto mb-8">
            <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-4 backdrop-blur-xl">
              <div className="flex items-start">
                <AlertCircle className="h-5 w-5 text-red-400 mr-2 mt-0.5" />
                <div className="text-sm text-red-300">
                  <p className="font-medium mb-1">Error</p>
                  <p>{error}</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Step 1: File Upload */}
        {currentStep === 'upload' && (
          <div className="max-w-2xl mx-auto">
            {isLoadingData ? (
              <div className="text-center py-12">
                <Loader2 className="h-12 w-12 text-cyan-400 animate-spin mx-auto mb-4" />
                <h3 className="text-lg font-medium text-white mb-2">Loading your data...</h3>
                <p className="text-gray-400">Checking for previously uploaded datasets</p>
              </div>
            ) : fileId ? (
              <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-6 text-center backdrop-blur-xl">
                <AlertCircle className="h-12 w-12 text-red-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-red-300 mb-2">Data Loading Error</h3>
                <p className="text-red-400 mb-4">
                  Found uploaded file &quot;{fileName}&quot; but couldn&apos;t load dataset information.
                </p>
                <p className="text-sm text-red-500 mb-4">{error}</p>
                <FileUpload onFileSelect={handleFileSelect} isLoading={isProcessingFile} />
              </div>
            ) : (
              <div>
                <div className="bg-cyan-900/20 border border-cyan-500/30 rounded-lg p-6 mb-6 text-center backdrop-blur-xl">
                  <Info className="h-8 w-8 text-cyan-400 mx-auto mb-3" />
                  <h3 className="text-lg font-medium text-cyan-300 mb-2">No Dataset Found</h3>
                  <p className="text-cyan-200 mb-3">
                    To create visualizations, please first upload a dataset in the Upload Data section.
                  </p>
                  <div className="flex justify-center space-x-3">
                    <a
                      href="/upload"
                      className="px-6 py-3 bg-gradient-to-r from-cyan-500 to-blue-500 text-white rounded-lg hover:from-cyan-600 hover:to-blue-600 transition-all duration-200 font-medium"
                    >
                      Go to Upload
                    </a>
                  </div>
                </div>
                <div className="border-t border-gray-700 pt-6">
                  <p className="text-center text-gray-400 mb-4">Or upload a file directly here for visualization:</p>
                  <FileUpload onFileSelect={handleFileSelect} isLoading={isProcessingFile} />
                </div>
              </div>
            )}
          </div>
        )}

        {/* Step 2: Chart Configuration */}
        {currentStep === 'configure' && (
          <div className="grid lg:grid-cols-3 gap-8">
            {/* Left Column - Chart Selection */}
            <div className="lg:col-span-2 space-y-8">
              {/* File Info */}
              <div className="bg-gray-800/50 backdrop-blur-xl rounded-lg shadow-md p-6 border border-gray-700">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <CheckCircle className="h-6 w-6 text-green-400" />
                    <div>
                      <h3 className="text-lg font-medium text-white">
                        {fileName || uploadedFile?.name}
                      </h3>
                      <p className="text-sm text-gray-400">
                        {dataColumns.length} columns detected
                        {datasetInfo && (
                          <span className="ml-2 text-cyan-400">
                            • {datasetInfo.total_rows} rows
                          </span>
                        )}
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={handleReset}
                    className="px-4 py-2 text-sm font-medium text-gray-400 bg-gray-700/50 hover:bg-gray-600/50 rounded-lg transition-colors border border-gray-600"
                  >
                    <RotateCcw className="h-4 w-4 mr-2 inline" />
                    Upload Different File
                  </button>
                </div>
              </div>

              {/* Chart Type Selection */}
              <div className="bg-gray-800/50 backdrop-blur-xl rounded-lg shadow-md p-6 border border-gray-700">
                <div className="flex items-center mb-6">
                  <BarChart3 className="h-6 w-6 text-cyan-400 mr-3" />
                  <h3 className="text-xl font-bold text-white">Choose Visualization</h3>
                </div>
                
                <ChartTypeSelector
                  selectedCategory={selectedCategory}
                  selectedChart={selectedChart}
                  onCategoryChange={setSelectedCategory}
                  onChartChange={setSelectedChart}
                  dataColumns={dataColumns}
                />
                
                {/* Submit Button for Configuration */}
                {selectedChart && Object.keys(chartConfig).length > 0 && (
                  <div className="mt-8 pt-6 border-t border-gray-700">
                    <div className="flex justify-end">
                      <button
                        onClick={handleGenerateChart}
                        disabled={isGeneratingChart}
                        className={`px-8 py-3 rounded-lg font-medium transition-all duration-200 flex items-center space-x-2 ${
                          isGeneratingChart
                            ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                            : 'bg-gradient-to-r from-cyan-500 to-blue-500 text-white hover:from-cyan-600 hover:to-blue-600 shadow-lg'
                        }`}
                      >
                        {isGeneratingChart ? (
                          <>
                            <Loader2 className="h-4 w-4 animate-spin" />
                            <span>Generating...</span>
                          </>
                        ) : (
                          <>
                            <Play className="h-4 w-4" />
                            <span>Generate Visualization</span>
                          </>
                        )}
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Right Column - Configuration */}
            <div className="lg:col-span-1">
              {selectedChart && (
                <div className="bg-gray-800/50 backdrop-blur-xl rounded-lg shadow-md p-6 border border-gray-700">
                  <div className="flex items-center mb-6">
                    <Settings className="h-6 w-6 text-cyan-400 mr-3" />
                    <h3 className="text-xl font-bold text-white">Chart Configuration</h3>
                  </div>
                  <ChartConfig
                    chartType={selectedChart}
                    dataColumns={dataColumns}
                    config={chartConfig}
                    onConfigChange={setChartConfig}
                    datasetInfo={datasetInfo}
                  />
                </div>
              )}
              
              {/* Data Preview */}
              <div className="mt-8 bg-gray-800/50 backdrop-blur-xl rounded-lg shadow-md p-6 border border-gray-700">
                <h4 className="font-medium text-white mb-4">Data Preview</h4>
                <div className="overflow-x-auto">
                  <table className="min-w-full text-sm">
                    <thead>
                      <tr className="border-b border-gray-600">
                        {dataColumns.slice(0, 4).map((col) => (
                          <th key={col} className="text-left py-2 px-2 font-medium text-gray-300">
                            {col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {dataPreview.slice(0, 3).map((row, index) => (
                        <tr key={index} className="border-b border-gray-700">
                          {dataColumns.slice(0, 4).map((col) => (
                            <td key={col} className="py-2 px-2 text-gray-400">
                              {String(row[col] || '—')}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="text-xs text-gray-500 mt-2">
                  Showing first 3 rows of {dataColumns.length} columns
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Step 3: Chart Preview */}
        {currentStep === 'preview' && (
          <div className="space-y-8">
            {/* Chart Preview */}
            <ChartPreview
              chartType={selectedChart}
              config={chartConfig}
              isGenerating={isGeneratingChart}
              onGenerate={handleGenerateChart}
              generatedChart={generatedChart}
            />

            {/* Action Buttons */}
            <div className="flex justify-center space-x-4">
              <button
                onClick={() => setCurrentStep('configure')}
                className="px-6 py-3 text-lg font-medium text-gray-400 bg-gray-800/50 hover:bg-gray-700/50 rounded-lg transition-colors border border-gray-600"
              >
                Back to Configuration
              </button>
              
              <div className="flex items-center space-x-3">
                <select
                  value={exportFormat}
                  onChange={(e) => setExportFormat(e.target.value as 'png' | 'svg' | 'html' | 'pdf')}
                  className="px-3 py-2 border border-gray-300 bg-white text-gray-900 rounded-lg focus:ring-2 focus:ring-green-500"
                >
                  <option value="png">PNG Image</option>
                  <option value="svg">SVG Vector</option>
                  <option value="html">HTML File</option>
                  <option value="pdf">PDF Document</option>
                </select>
                <button
                  onClick={handleExportChart}
                  disabled={!generatedChart}
                  className={`px-6 py-3 text-lg font-medium rounded-lg transition-all duration-200 flex items-center space-x-2 shadow-lg ${
                    generatedChart
                      ? 'text-white bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600'
                      : 'text-gray-500 bg-gray-300 cursor-not-allowed'
                  }`}
                >
                  <Download className="h-5 w-5" />
                  <span>Export Chart</span>
                </button>
              </div>
              
              <button
                onClick={handleReset}
                className="px-6 py-3 text-lg font-medium text-cyan-400 bg-cyan-900/20 hover:bg-cyan-800/30 rounded-lg transition-colors border border-cyan-500/30"
              >
                Create Another Chart
              </button>
            </div>
          </div>
        )}
        
        </div>
      </div>
    </div>
  );
}