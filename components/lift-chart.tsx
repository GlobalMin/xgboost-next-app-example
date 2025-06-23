"use client";

import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

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
  if (!data || data.length === 0) {
    return <p className="text-gray-500">No data available for lift chart</p>;
  }

  // Transform data for Recharts
  const chartData = data.map((item) => ({
    bin: `Bin ${item.bin}`,
    "Predicted Rate": parseFloat((item.avg_prediction * 100).toFixed(1)),
    "Actual Rate": parseFloat((item.actual_rate * 100).toFixed(1)),
    predictedRaw: item.avg_prediction,
    actualRaw: item.actual_rate,
    count: item.count,
  }));

  // Calculate min and max for better Y-axis scaling
  const allRates = chartData.flatMap((d) => [
    d["Predicted Rate"],
    d["Actual Rate"],
  ]);
  const minRate = Math.max(0, Math.floor(Math.min(...allRates) * 0.9)); // 10% padding below, but never below 0
  const maxRate = Math.min(100, Math.ceil(Math.max(...allRates) * 1.1)); // 10% padding above, but never above 100%

  // Custom tooltip with better styling
  const CustomTooltip = ({ active, payload, label }: {
    active?: boolean;
    payload?: Array<{ value: number; payload: { actualRaw: number; predictedRaw: number; count: number } }>;
    label?: string;
  }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      const lift = data.actualRaw > 0 ? ((data.predictedRaw / data.actualRaw) * 100).toFixed(1) : "N/A";
      return (
        <div className="bg-white p-4 border border-gray-200 rounded-lg shadow-xl">
          <p className="font-semibold text-gray-900 mb-2">{label}</p>
          <div className="space-y-1">
            <p className="flex items-center gap-2">
              <span className="w-3 h-3 bg-blue-500 rounded-full"></span>
              <span className="text-sm text-gray-700">Predicted: <span className="font-semibold text-blue-600">{payload[0].value}%</span></span>
            </p>
            <p className="flex items-center gap-2">
              <span className="w-3 h-3 bg-green-500 rounded-full"></span>
              <span className="text-sm text-gray-700">Actual: <span className="font-semibold text-green-600">{payload[1].value}%</span></span>
            </p>
            <div className="border-t pt-1 mt-2">
              <p className="text-sm text-gray-600">Records: <span className="font-semibold">{data.count.toLocaleString()}</span></p>
              <p className="text-sm text-gray-600">Lift: <span className="font-semibold">{lift}%</span></p>
            </div>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <>
      <div className="mb-4">
        <p className="text-sm text-gray-600">
          Comparison of predicted probabilities vs actual outcomes across
          deciles
        </p>
        <p className="text-xs text-gray-500 mt-1">
          Total records in chart:{" "}
          {data.reduce((sum, item) => sum + item.count, 0).toLocaleString()}
        </p>
      </div>

      {/* Chart */}
      <div className="w-full overflow-x-auto">
        <div className="h-[500px] bg-gray-50 rounded-lg p-4" style={{ minWidth: "900px" }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={chartData}
              margin={{
                top: 30,
                right: 40,
                left: 50,
                bottom: 80,
              }}
            >
              <defs>
                <linearGradient id="colorPredicted" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.1}/>
                  <stop offset="95%" stopColor="#3B82F6" stopOpacity={0}/>
                </linearGradient>
                <linearGradient id="colorActual" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10B981" stopOpacity={0.1}/>
                  <stop offset="95%" stopColor="#10B981" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid 
                strokeDasharray="3 3" 
                stroke="#e5e7eb" 
                vertical={true}
                horizontalPoints={[0, 20, 40, 60, 80, 100]}
              />
              <XAxis
                dataKey="bin"
                tick={{ fontSize: 12, fill: "#6b7280" }}
                interval={0}
                angle={-45}
                textAnchor="end"
                height={80}
                stroke="#e5e7eb"
              />
              <YAxis
                label={{
                  value: "Rate (%)",
                  angle: -90,
                  position: "insideLeft",
                  offset: 10,
                  style: { fontSize: 14, fill: "#6b7280", fontWeight: 500 }
                }}
                tick={{ fontSize: 12, fill: "#6b7280" }}
                domain={[minRate, maxRate]}
                tickFormatter={(value) => `${value}%`}
                stroke="#e5e7eb"
              />
              <Tooltip 
                content={<CustomTooltip />} 
                cursor={{ stroke: '#9ca3af', strokeWidth: 1 }}
              />
              <Legend
                verticalAlign="top"
                height={36}
                iconType="line"
                wrapperStyle={{
                  paddingBottom: "10px",
                  fontSize: "14px"
                }}
                formatter={(value: string) => (
                  <span style={{ color: value === "Predicted Rate" ? "#3B82F6" : "#10B981", fontWeight: 500 }}>
                    {value}
                  </span>
                )}
              />
              <Line
                type="monotone"
                dataKey="Predicted Rate"
                stroke="#3B82F6"
                strokeWidth={3}
                dot={{ fill: "#3B82F6", strokeWidth: 2, r: 5 }}
                name="Predicted Rate"
                activeDot={{ r: 7, strokeWidth: 0 }}
              />
              <Line
                type="monotone"
                dataKey="Actual Rate"
                stroke="#10B981"
                strokeWidth={3}
                dot={{ fill: "#10B981", strokeWidth: 2, r: 5 }}
                name="Actual Rate"
                activeDot={{ r: 7, strokeWidth: 0 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </>
  );
}
