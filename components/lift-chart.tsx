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

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white p-3 border rounded shadow-lg">
          <p className="font-semibold">{label}</p>
          <p className="text-blue-600">Predicted Rate: {payload[0].value}%</p>
          <p className="text-green-600">Actual Rate: {payload[1].value}%</p>
          <p className="text-gray-600">Count: {data.count} records</p>
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
        <div className="h-[500px]" style={{ minWidth: "900px" }}>
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
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
              <XAxis
                dataKey="bin"
                tick={{ fontSize: 11 }}
                interval={0}
                angle={-45}
                textAnchor="end"
                height={80}
              />
              <YAxis
                label={{
                  value: "Rate (%)",
                  angle: -90,
                  position: "insideLeft",
                  offset: -10,
                }}
                tick={{ fontSize: 12 }}
                domain={[minRate, maxRate]}
                tickFormatter={(value) => `${value}%`}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend
                verticalAlign="top"
                height={36}
                iconType="line"
                wrapperStyle={{
                  paddingBottom: "10px",
                }}
              />
              <Line
                type="monotone"
                dataKey="Predicted Rate"
                stroke="#3B82F6"
                strokeWidth={3}
                dot={{ fill: "#3B82F6", strokeWidth: 2, r: 5 }}
                name="Predicted Rate"
                activeDot={{ r: 7 }}
              />
              <Line
                type="monotone"
                dataKey="Actual Rate"
                stroke="#10B981"
                strokeWidth={3}
                dot={{ fill: "#10B981", strokeWidth: 2, r: 5 }}
                name="Actual Rate"
                activeDot={{ r: 7 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </>
  );
}
