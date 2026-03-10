// src/Components/PartsPricingTable.jsx
import React, { useState, useEffect } from "react";

const PartsPricingTable = ({ fileStates = [] }) => {
  // Process options for dropdown
  const processOptions = [
    "SLA ABS 35",
    "SLA clear 75",
    "SLS NYLON 40",
    "SLS Nylon PA12 GF 65",
    "DLP Rubber 70",
    "DLP PRO BLACK 80",
    "FDM PLA 15",
    "FDM ABS 15",
    "SLA ABS"
  ];

  const [tableRows, setTableRows] = useState([]);

  // Update table rows when fileStates changes
  useEffect(() => {
    if (fileStates.length > 0) {
      const newRows = fileStates.map((file, index) => ({
        id: index + 1,
        partName: file.fileName || `Part ${index + 1}`,
        cc: file.volume_cc ? file.volume_cc.toFixed(2) : "0.00",
        process: processOptions[0], // Default to first option
        painting: "",
        paintingAmount: 0, // Store painting amount separately
        quantity: file.quantity || 1,
        perPartCost: "",
        totalCost: ""
      }));
      setTableRows(newRows);
    } else {
      setTableRows([]);
    }
  }, [fileStates]);

  // Function to extract number from process string
  const extractProcessValue = (processString) => {
    const matches = processString.match(/\d+/);
    return matches ? parseInt(matches[0], 10) : 0;
  };

  // Calculate per part cost based on process value, cc, and painting amount
  const calculatePerPartCost = (cc, process, paintingAmount) => {
    const processValue = extractProcessValue(process);
    const ccValue = parseFloat(cc) || 0;
    const painting = parseFloat(paintingAmount) || 0;
    
    // Formula: processValue * cc + paintingAmount
    return (processValue * ccValue + painting).toFixed(2);
  };

  const updateTableRow = (id, field, value) => {
    setTableRows(prevRows => 
      prevRows.map(row => {
        if (row.id === id) {
          let updatedRow = { ...row, [field]: value };
          
          // Recalculate perPartCost if cc, process, or paintingAmount changes
          if (field === 'cc' || field === 'process' || field === 'paintingAmount') {
            const cc = field === 'cc' ? value : row.cc;
            const process = field === 'process' ? value : row.process;
            const paintingAmount = field === 'paintingAmount' ? value : row.paintingAmount;
            
            updatedRow.perPartCost = calculatePerPartCost(cc, process, paintingAmount);
          }
          
          // Calculate total cost if quantity or perPartCost changes
          if (field === 'quantity' || field === 'perPartCost') {
            const quantity = field === 'quantity' ? value : row.quantity;
            const perPartCost = field === 'perPartCost' ? value : updatedRow.perPartCost || row.perPartCost;
            
            if (quantity && perPartCost && !isNaN(quantity) && !isNaN(perPartCost)) {
              updatedRow.totalCost = (Number(quantity) * Number(perPartCost)).toFixed(2);
            }
          }
          
          return updatedRow;
        }
        return row;
      })
    );
  };

  // Calculate subtotal (sum of all total costs)
  const subtotal = tableRows.reduce((sum, row) => sum + (Number(row.totalCost) || 0), 0);
  
  // Calculate GST (18% of subtotal)
  const gst = subtotal * 0.18;
  
  // Calculate grand total (subtotal + GST)
  const grandTotal = subtotal + gst;

  if (tableRows.length === 0) {
    return null; // Don't show table if no files uploaded
  }

  return (
    <div className="w-full mx-auto mt-8 p-6 bg-gray-800 rounded-lg">
      <h3 className="text-white text-2xl font-bold mb-6">Parts & Pricing Details</h3>
      
      <div className="overflow-x-auto">
        <table className="w-full text-sm text-left text-gray-300">
          <thead className="text-xs uppercase bg-gray-700 text-gray-300">
            <tr>
              <th className="px-6 py-4">SLNO</th>
              <th className="px-6 py-4">PART NAME</th>
              <th className="px-6 py-4">CC</th>
              <th className="px-6 py-4">Process</th>
              <th className="px-6 py-4">Painting Amount (₹)</th>
              <th className="px-6 py-4">Quantity</th>
              <th className="px-6 py-4">Per Part Cost (₹)</th>
              <th className="px-6 py-4">Total Cost (₹)</th>
            </tr>
          </thead>
          <tbody>
            {tableRows.map((row, index) => (
              <tr key={row.id} className="border-b border-gray-700 hover:bg-gray-750">
                <td className="px-6 py-4 font-medium text-base">{index + 1}</td>
                <td className="px-6 py-4">
                  <span className="text-white text-base">{row.partName}</span>
                </td>
                <td className="px-6 py-4">
                  <input
                    type="number"
                    value={row.cc}
                    onChange={(e) => updateTableRow(row.id, 'cc', e.target.value)}
                    className="w-24 p-2 bg-gray-700 border border-gray-600 rounded text-white text-base focus:ring-2 focus:ring-cyan-500 focus:border-transparent"
                    step="0.01"
                    min="0"
                  />
                </td>
                <td className="px-6 py-4">
                  <select
                    value={row.process}
                    onChange={(e) => updateTableRow(row.id, 'process', e.target.value)}
                    className="w-44 p-2 bg-gray-700 border border-gray-600 rounded text-white text-base focus:ring-2 focus:ring-cyan-500 focus:border-transparent"
                  >
                    {processOptions.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </td>
                <td className="px-6 py-4">
                  <input
                    type="number"
                    value={row.paintingAmount}
                    onChange={(e) => updateTableRow(row.id, 'paintingAmount', e.target.value)}
                    placeholder="Painting amount"
                    className="w-28 p-2 bg-gray-700 border border-gray-600 rounded text-white text-base focus:ring-2 focus:ring-cyan-500 focus:border-transparent"
                    step="0.01"
                    min="0"
                  />
                </td>
                <td className="px-6 py-4">
                  <input
                    type="number"
                    value={row.quantity}
                    onChange={(e) => updateTableRow(row.id, 'quantity', e.target.value)}
                    min="1"
                    className="w-20 p-2 bg-gray-700 border border-gray-600 rounded text-white text-base focus:ring-2 focus:ring-cyan-500 focus:border-transparent"
                  />
                </td>
                <td className="px-6 py-4">
                  <span className="text-cyan-400 font-semibold text-base">
                    ₹{parseFloat(row.perPartCost || '0').toFixed(2)}
                  </span>
                </td>
                <td className="px-6 py-4">
                  <span className="text-cyan-400 font-semibold text-base">
                    ₹{parseFloat(row.totalCost || '0').toFixed(2)}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      {/* Summary Section with Subtotal, GST, and Grand Total */}
      <div className="flex flex-col items-end mt-8 space-y-3">
        {/* Subtotal */}
        <div className="text-white bg-gray-700 px-8 py-3 rounded-lg w-80 flex justify-between">
          <span className="font-semibold text-lg">Sub Total:</span>
          <span className="text-cyan-400 font-bold text-lg">
            ₹{subtotal.toFixed(2)}
          </span>
        </div>
        
        {/* GST 18% */}
        <div className="text-white bg-gray-700 px-8 py-3 rounded-lg w-80 flex justify-between">
          <span className="font-semibold text-lg">GST (18%):</span>
          <span className="text-cyan-400 font-bold text-lg">
            ₹{gst.toFixed(2)}
          </span>
        </div>
        
        {/* Grand Total */}
        <div className="text-white bg-gray-700 px-8 py-4 rounded-lg w-80 flex justify-between border-t-2 border-cyan-500">
          <span className="font-semibold text-xl">Grand Total:</span>
          <span className="text-cyan-400 font-bold text-2xl">
            ₹{grandTotal.toFixed(2)}
          </span>
        </div>
      </div>

      {/* Total Items count */}
      <div className="mt-4 text-gray-400 text-sm text-right">
        Total Items: {tableRows.length}
      </div>
    </div>
  );
};

export default PartsPricingTable;