'use client';

import React from 'react';
import { TrendingUp } from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface LiftChartData {
  bin: number;
  avg_prediction: number;
  actual_rate: number;
  count: number;
}

interface LiftChartProps {
  data: LiftChartData[];
}

export function LiftChart({ data }: LiftChartProps) {
  console.log('LiftChart component rendered with data:', data);
  
  if (!data || data.length === 0) {
    console.warn('LiftChart: No data provided or empty data array');
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center space-x-2 mb-4">
          <TrendingUp className="h-5 w-5 text-blue-600" />
          <h3 className="text-lg font-semibold">Model Lift Chart</h3>
        </div>
        <p className="text-gray-500">No data available for lift chart</p>
      </div>
    );
  }

  // Transform data for Recharts
  const chartData = data.map(item => ({
    bin: `Bin ${item.bin}`,
    'Predicted Rate': (item.avg_prediction * 100).toFixed(1),
    'Actual Rate': (item.actual_rate * 100).toFixed(1),
    predictedRaw: item.avg_prediction,
    actualRaw: item.actual_rate,
    count: item.count
  }));

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white p-3 border rounded shadow-lg">
          <p className="font-semibold">{label}</p>
          <p className="text-blue-600">
            Predicted Rate: {payload[0].value}%
          </p>
          <p className="text-green-600">
            Actual Rate: {payload[1].value}%
          </p>
          <p className="text-gray-600">
            Count: {data.count} records
          </p>
        </div>
      );
    }
    return null;
  };


  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center space-x-2 mb-4">
        <TrendingUp className="h-5 w-5 text-blue-600" />
        <h3 className="text-lg font-semibold">Model Lift Chart</h3>
      </div>
      
      <div className="mb-4">
        <p className="text-sm text-gray-600">
          Comparison of predicted probabilities vs actual outcomes across deciles
        </p>
      </div>

      {/* Chart */}
      <div className="h-80 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={chartData}
            margin={{
              top: 20,
              right: 30,
              left: 20,
              bottom: 5,
            }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="bin" 
              tick={{ fontSize: 12 }}
            />
            <YAxis 
              label={{ value: 'Rate (%)', angle: -90, position: 'insideLeft' }}
              tick={{ fontSize: 12 }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Line 
              type="monotone"
              dataKey="Predicted Rate" 
              stroke="#3B82F6" 
              strokeWidth={3}
              dot={{ fill: '#3B82F6', strokeWidth: 2, r: 4 }}
              name="Predicted Rate"
            />
            <Line 
              type="monotone"
              dataKey="Actual Rate" 
              stroke="#10B981" 
              strokeWidth={3}
              dot={{ fill: '#10B981', strokeWidth: 2, r: 4 }}
              name="Actual Rate"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
