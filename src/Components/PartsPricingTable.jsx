// src/Components/PartsPricingTable.jsx
import React, { useState, useEffect } from "react";
import { PDFDownloadLink } from "@react-pdf/renderer";
import QuotationDocument from "./PDFGenerator";

const PartsPricingTable = ({ fileStates = [], discount = 0, paintingCost = 0, companyName = "", qrnCode = "", packingOption = "" }) => {
  // Process options for dropdown
  const processOptions = [
    "SLA ABS",
    "SLA clear",
    "SLS NYLON",
    "SLS Nylon PA12 GF",
    "DLP Rubber",
    "DLP PRO BLACK",
    "FDM PLA",
    "FDM ABS",
    "SLA ABS"
  ];

  const [tableRows, setTableRows] = useState([]);
  const [validationErrors, setValidationErrors] = useState({});
  const [showDownloadButton, setShowDownloadButton] = useState(false);
  const [pdfError, setPdfError] = useState("");

  // Update table rows when fileStates changes
  useEffect(() => {
    if (fileStates.length > 0) {
      const newRows = fileStates.map((file, index) => ({
        id: index + 1,
        partName: file.fileName || `Part ${index + 1}`,
        cc: file.volume_cc ? file.volume_cc.toFixed(2) : "",
        process: processOptions[0], // Default to first option
        painting: "",
        amount: "", // Amount field
        quantity: "", // Blank default
        perPartCost: "",
        totalCost: ""
      }));
      setTableRows(newRows);
      setValidationErrors({});
      setPdfError(""); // Clear any previous PDF errors
      setShowDownloadButton(false); // Reset download button when new data loads
    } else {
      setTableRows([]);
      setValidationErrors({});
      setPdfError("");
      setShowDownloadButton(false);
    }
  }, [fileStates]);

  // Calculate per part cost based on amount and cc (amount * cc)
  const calculatePerPartCost = (cc, amount) => {
    const ccValue = parseFloat(cc) || 0;
    const amountValue = parseFloat(amount) || 0;
    
    // Formula: amount * cc
    return (amountValue * ccValue).toFixed(2);
  };

  // Validate amount field
  const validateAmount = (value) => {
    if (!value) return ""; // Empty is allowed
    const numValue = parseFloat(value);
    if (isNaN(numValue) || numValue < 35) {
      return "Amount must be ≥ 35";
    }
    return "";
  };

  const handleAmountChange = (id, value) => {
    // Always update the row with the new value
    setTableRows(prevRows => 
      prevRows.map(row => {
        if (row.id === id) {
          return { ...row, amount: value };
        }
        return row;
      })
    );

    // Validate and set error message
    const error = validateAmount(value);
    setValidationErrors(prev => ({
      ...prev,
      [id]: error
    }));

    // Only calculate perPartCost if valid
    if (!error) {
      setTableRows(prevRows => 
        prevRows.map(row => {
          if (row.id === id) {
            const perPartCost = calculatePerPartCost(row.cc, value);
            return { ...row, perPartCost };
          }
          return row;
        })
      );
    }
  };

  const updateTableRow = (id, field, value) => {
    setTableRows(prevRows => 
      prevRows.map(row => {
        if (row.id === id) {
          let updatedRow = { ...row, [field]: value };
          
          // Recalculate perPartCost if cc changes
          if (field === 'cc') {
            updatedRow.perPartCost = calculatePerPartCost(value, row.amount);
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
  
  // Add painting cost directly to subtotal
  const paintingCostValue = parseFloat(paintingCost) || 0;
  const subtotalWithPainting = subtotal + paintingCostValue;
  
  // Calculate GST (18% of subtotal with painting)
  const gst = subtotalWithPainting * 0.18;
  
  // Calculate Total (subtotal with painting + GST)
  const total = subtotalWithPainting + gst;
  
  // Calculate Discount Amount (discount percentage of Total)
  const discountAmount = total * (parseFloat(discount) / 100);
  
  // Calculate Grand Total (Total - Discount)
  const grandTotal = total - discountAmount;

  // Get packing option label
  const getPackingLabel = (value) => {
    const options = {
      courier: "Courier",
      porter: "Porter",
      outstation: "Outstation"
    };
    return options[value] || "Not Selected";
  };

  // Check if any row has validation errors
  const hasValidationErrors = Object.values(validationErrors).some(error => error !== "");

  // Handle Generate PDF button click
  const handleGeneratePDF = () => {
    if (hasValidationErrors) {
      setPdfError("Please fix validation errors before generating PDF.");
      return;
    }
    setShowDownloadButton(true);
    setPdfError("");
  };

  // Get GST and address info from fileStates if available
  const getGstNo = () => {
    return fileStates.length > 0 && fileStates[0]?.gstNo ? fileStates[0].gstNo : "";
  };

  const getShippingAddress = () => {
    return fileStates.length > 0 && fileStates[0]?.shippingAddress ? fileStates[0].shippingAddress : "";
  };

  const getBillingAddress = () => {
    return fileStates.length > 0 && fileStates[0]?.billingAddress ? fileStates[0].billingAddress : "";
  };

  // Log the data to help debug
  console.log("PartsPricingTable Props:", {
    companyName,
    qrnCode,
    packingOption,
    fileStatesCount: fileStates.length,
    firstFileState: fileStates[0]
  });

  console.log("Address Data:", {
    shippingAddress: getShippingAddress(),
    billingAddress: getBillingAddress(),
    gstNo: getGstNo()
  });

  if (tableRows.length === 0) {
    return null; // Don't show table if no files uploaded
  }

  return (
    <div className="w-full mx-auto mt-8 p-6 bg-gray-800 rounded-lg">
      <h3 className="text-white text-2xl font-bold mb-6">Parts & Pricing Details</h3>
      
      {/* Table container - No scrollbars */}
      <div className="w-full overflow-visible">
        <table className="w-full text-left text-gray-300 border-collapse">
          <thead className="text-sm uppercase bg-gray-700 text-gray-300">
            <tr>
              <th className="px-4 py-4">SLNO</th>
              <th className="px-4 py-4">PART NAME</th>
              <th className="px-4 py-4">CC</th>
              <th className="px-4 py-4">Process</th>
              <th className="px-4 py-4">Amount (₹)</th>
              <th className="px-4 py-4">Quantity</th>
              <th className="px-4 py-4">Per Part Cost (₹)</th>
              <th className="px-4 py-4">Total Cost (₹)</th>
            </tr>
          </thead>
          <tbody>
            {tableRows.map((row, index) => (
              <tr key={row.id} className="border-b border-gray-700 hover:bg-gray-750">
                <td className="px-4 py-4 font-medium text-base">{index + 1}</td>
                <td className="px-4 py-4">
                  <span className="text-white text-base" title={row.partName}>
                    {row.partName.length > 20 ? row.partName.substring(0, 20) + '...' : row.partName}
                  </span>
                </td>
                <td className="px-4 py-4">
                  <input
                    type="number"
                    value={row.cc}
                    onChange={(e) => updateTableRow(row.id, 'cc', e.target.value)}
                    placeholder="CC"
                    className="w-20 p-2 bg-gray-700 border border-gray-600 rounded text-white text-base focus:ring-2 focus:ring-cyan-500 focus:border-transparent"
                    step="0.01"
                    min="0"
                  />
                </td>
                <td className="px-4 py-4">
                  <select
                    value={row.process}
                    onChange={(e) => updateTableRow(row.id, 'process', e.target.value)}
                    className="w-36 p-2 bg-gray-700 border border-gray-600 rounded text-white text-base focus:ring-2 focus:ring-cyan-500 focus:border-transparent"
                  >
                    {processOptions.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </td>
                <td className="px-4 py-4 relative">
                  <input
                    type="number"
                    value={row.amount}
                    onChange={(e) => handleAmountChange(row.id, e.target.value)}
                    placeholder="Amount"
                    className={`w-24 p-2 bg-gray-700 border rounded text-white text-base focus:ring-2 focus:ring-cyan-500 focus:border-transparent ${
                      validationErrors[row.id] ? 'border-red-500' : 'border-gray-600'
                    }`}
                    step="0.01"
                  />
                  {validationErrors[row.id] && (
                    <div className="absolute z-10 mt-1 p-2 bg-red-500 text-white text-xs rounded shadow-lg whitespace-nowrap">
                      ⚠️ {validationErrors[row.id]}
                    </div>
                  )}
                </td>
                <td className="px-4 py-4">
                  <input
                    type="number"
                    value={row.quantity}
                    onChange={(e) => updateTableRow(row.id, 'quantity', e.target.value)}
                    placeholder="Qty"
                    className="w-16 p-2 bg-gray-700 border border-gray-600 rounded text-white text-base focus:ring-2 focus:ring-cyan-500 focus:border-transparent"
                    min="1"
                  />
                </td>
                <td className="px-4 py-4">
                  <span className="text-cyan-400 font-semibold text-base">
                    ₹{parseFloat(row.perPartCost || '0').toFixed(2)}
                  </span>
                </td>
                <td className="px-4 py-4">
                  <span className="text-cyan-400 font-semibold text-base">
                    ₹{parseFloat(row.totalCost || '0').toFixed(2)}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      {/* Summary Section with Subtotal, GST, Total, Discount, and Grand Total */}
      <div className="flex flex-col items-end mt-8 space-y-3">
        {/* Subtotal (includes painting cost from sidebar) */}
        <div className="text-white bg-gray-700 px-8 py-3 rounded-lg w-96 flex justify-between">
          <span className="font-semibold text-lg">Sub Total:</span>
          <span className="text-cyan-400 font-bold text-lg">
            ₹{subtotalWithPainting.toFixed(2)}
          </span>
        </div>
        
        {/* GST 18% */}
        <div className="text-white bg-gray-700 px-8 py-3 rounded-lg w-96 flex justify-between">
          <span className="font-semibold text-lg">GST (18%):</span>
          <span className="text-cyan-400 font-bold text-lg">
            ₹{gst.toFixed(2)}
          </span>
        </div>
        
        {/* Total (Subtotal + GST) */}
        <div className="text-white bg-gray-700 px-8 py-3 rounded-lg w-96 flex justify-between">
          <span className="font-semibold text-lg">Total:</span>
          <span className="text-cyan-400 font-bold text-lg">
            ₹{total.toFixed(2)}
          </span>
        </div>
        
        {/* Discount (if any) */}
        {discount > 0 && (
          <div className="text-white bg-gray-700 px-8 py-3 rounded-lg w-96 flex justify-between">
            <span className="font-semibold text-lg">Discount ({discount}%):</span>
            <span className="text-red-400 font-bold text-lg">
              -₹{discountAmount.toFixed(2)}
            </span>
          </div>
        )}
        
        {/* Grand Total (Total - Discount) */}
        <div className="text-white bg-gray-700 px-8 py-4 rounded-lg w-96 flex justify-between border-t-2 border-cyan-500">
          <span className="font-semibold text-xl">Grand Total:</span>
          <span className="text-cyan-400 font-bold text-2xl">
            ₹{grandTotal.toFixed(2)}
          </span>
        </div>
      </div>

      {/* Validation Summary */}
      {hasValidationErrors && (
        <div className="mt-4 p-3 bg-red-500/20 border border-red-500 rounded-lg">
          <p className="text-red-400 text-sm">
            ⚠️ Please fix the validation errors before proceeding.
          </p>
        </div>
      )}

      {/* PDF Error Message */}
      {pdfError && (
        <div className="mt-4 p-3 bg-red-500/20 border border-red-500 rounded-lg">
          <p className="text-red-400 text-sm">
            ⚠️ {pdfError}
          </p>
        </div>
      )}

      {/* Total Items count */}
      <div className="mt-4 text-gray-400 text-sm text-right">
        Total Items: {tableRows.length}
      </div>

      {/* PDF Action Buttons */}
      <div className="flex justify-end mt-8 pt-4 border-t border-gray-700 gap-4">
        {!showDownloadButton ? (
          <button
            onClick={handleGeneratePDF}
            className="px-8 py-3 bg-cyan-600 text-white font-semibold rounded-lg hover:bg-cyan-700 transition flex items-center gap-2 shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={hasValidationErrors}
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            Generate PDF
          </button>
        ) : (
          <PDFDownloadLink
            document={
              <QuotationDocument 
                formData={{ files: fileStates }}
                total={grandTotal}
                fileCosts={tableRows.map(row => ({ cost: parseFloat(row.totalCost || 0) }))}
                companyName={companyName}
                qrnCode={qrnCode}
                gstNo={getGstNo()}
                shippingAddress={getShippingAddress()}
                billingAddress={getBillingAddress()}
                packingOption={packingOption}
                discount={discount}
                paintingCost={paintingCost}
                tableRows={tableRows}
              />
            }
            fileName={`3DE_Tech_Quotation_${companyName || 'Customer'}_${new Date().toISOString().split('T')[0]}.pdf`}
          >
            {({ loading }) => (
              <button
                className="px-8 py-3 bg-green-600 text-white font-semibold rounded-lg hover:bg-green-700 transition flex items-center gap-2 shadow-lg hover:shadow-xl"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                {loading ? 'Preparing PDF...' : 'Download PDF'}
              </button>
            )}
          </PDFDownloadLink>
        )}
      </div>
    </div>
  );
};

export default PartsPricingTable;