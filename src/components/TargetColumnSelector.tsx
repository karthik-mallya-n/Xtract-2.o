'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { ColumnInfo, DatasetColumnsResponse } from '@/lib/api';

interface TargetColumnSelectorProps {
  fileId: string;
  onColumnSelect: (columnName: string) => void;
  onError: (error: string) => void;
}

export default function TargetColumnSelector({ 
  fileId, 
  onColumnSelect, 
  onError 
}: TargetColumnSelectorProps) {
  const [columns, setColumns] = useState<ColumnInfo[]>([]);
  const [selectedColumn, setSelectedColumn] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [datasetInfo, setDatasetInfo] = useState<{total_rows: number, total_columns: number} | null>(null);

  const fetchDatasetColumns = useCallback(async () => {
    try {
      setLoading(true);
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000'}/api/dataset-columns?file_id=${fileId}`);
      const data: DatasetColumnsResponse = await response.json();
      
      if (data.success && data.columns) {
        setColumns(data.columns);
        setDatasetInfo({
          total_rows: data.total_rows || 0,
          total_columns: data.total_columns || 0
        });
        // Auto-select the last column as default (common convention for target)
        if (data.columns.length > 0) {
          const defaultTarget = data.columns[data.columns.length - 1].name;
          setSelectedColumn(defaultTarget);
          onColumnSelect(defaultTarget);
        }
      } else {
        onError(data.error || 'Failed to load dataset columns');
      }
    } catch (error) {
      onError(`Error loading columns: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  }, [fileId, onColumnSelect, onError]);

  useEffect(() => {
    fetchDatasetColumns();
  }, [fetchDatasetColumns]);

  const handleColumnSelect = (columnName: string) => {
    setSelectedColumn(columnName);
    onColumnSelect(columnName);
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="space-y-2">
            <div className="h-3 bg-gray-200 rounded"></div>
            <div className="h-3 bg-gray-200 rounded w-5/6"></div>
            <div className="h-3 bg-gray-200 rounded w-4/6"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-2">
          Select Target Column
        </h2>
        <p className="text-gray-600 text-sm">
          Choose which column contains the values you want to predict.
        </p>
        {datasetInfo && (
          <div className="mt-2 text-sm text-gray-500">
            Dataset: {datasetInfo.total_rows} rows × {datasetInfo.total_columns} columns
          </div>
        )}
      </div>

      <div className="space-y-3">
        {columns.map((column) => (
          <div
            key={column.name}
            className={`border rounded-lg p-4 cursor-pointer transition-all ${
              selectedColumn === column.name
                ? 'border-blue-500 bg-blue-50'
                : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
            }`}
            onClick={() => handleColumnSelect(column.name)}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <input
                  type="radio"
                  name="target_column"
                  value={column.name}
                  checked={selectedColumn === column.name}
                  onChange={() => handleColumnSelect(column.name)}
                  className="text-blue-600 focus:ring-blue-500"
                />
                <div>
                  <h3 className="font-medium text-gray-800">{column.name}</h3>
                  <div className="flex items-center space-x-4 text-sm text-gray-600 mt-1">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      column.type === 'numeric' 
                        ? 'bg-green-100 text-green-800' 
                        : 'bg-blue-100 text-blue-800'
                    }`}>
                      {column.type}
                    </span>
                    <span>Unique: {column.unique_count}</span>
                    {column.null_count > 0 && (
                      <span className="text-orange-600">Missing: {column.null_count}</span>
                    )}
                  </div>
                </div>
              </div>
              <div className="text-right">
                <div className="text-xs text-gray-500 mb-1">Sample values:</div>
                <div className="text-sm text-gray-700 max-w-32 truncate">
                  {column.sample_values.slice(0, 3).map(val => 
                    typeof val === 'string' && val.length > 10 
                      ? val.substring(0, 10) + '...' 
                      : val
                  ).join(', ')}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {selectedColumn && (
        <div className="mt-4 p-3 bg-blue-50 rounded-lg">
          <p className="text-sm text-blue-700">
            <strong>Selected:</strong> {selectedColumn} will be used as the target column for prediction.
          </p>
        </div>
      )}
    </div>
  );
}