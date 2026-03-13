// FileUpload.jsx
import React, { useState, useEffect } from "react";
import { STLLoader } from "three-stdlib";
import Swipe from "react-easy-swipe";
import Papa from 'papaparse';

const density = 1.05; // g/cm³
const MAX_SIZE_MB = 300;

// Your Google Sheet ID from the URL
const SPREADSHEET_ID = '1pO5WvXYunucJFicbAC1aZZAyXgEIf0o8trwcPx3nnRE'; // Replace with your actual ID

// Column indices (0-based)
const COMPANY_NAME_COL = 1; // Column B
const GST_COL = 2;           // Column C
const SHIPPING_COL = 3;      // Column D
const BILLING_COL = 4;       // Column E

const FileUpload = ({
  onFileLoad,
  onError,
  onLoading,
  reorientedDimensions,
  liveDimensions,
  liveMode,
  setLiveMode,
  onFileStatesChange,
  onDiscountChange,
  onPaintingCostChange,
  onCompanyNameChange,
  onQrnCodeChange,
  onPackingOptionChange,
}) => {
  const [fileStates, setFileStates] = useState([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusLabel, setStatusLabel] = useState("");
  
  // State for company info
  const [companyName, setCompanyName] = useState("");
  const [qrnCode, setQrnCode] = useState("");
  const [gstNo, setGstNo] = useState("");
  const [shippingAddress, setShippingAddress] = useState("");
  const [billingAddress, setBillingAddress] = useState("");
  
  // State for spreadsheet data
  const [companyData, setCompanyData] = useState([]);
  const [isLoadingSpreadsheet, setIsLoadingSpreadsheet] = useState(true);
  const [spreadsheetError, setSpreadsheetError] = useState(null);
  const [retryCount, setRetryCount] = useState(0);
  
  // State for packing and forwarding
  const [packingOption, setPackingOption] = useState("");
  
  // State for discount
  const [discount, setDiscount] = useState("");
  
  // State for painting cost
  const [paintingCost, setPaintingCost] = useState("");

  // Packing options
  const packingOptions = [
    { value: "courier", label: "Courier" },
    { value: "porter", label: "Porter" },
    { value: "outstation", label: "Outstation" },
  ];

  // Load Google Sheet data on component mount using shareable link
  useEffect(() => {
    loadSheetData();
  }, [retryCount]);

  const loadSheetData = async () => {
    setIsLoadingSpreadsheet(true);
    setSpreadsheetError(null);
    
    try {
      // Method: Using shareable link with CSV export
      // This works when the sheet is shared with "Anyone with the link"
      const csvUrl = `https://docs.google.com/spreadsheets/d/${SPREADSHEET_ID}/gviz/tq?tqx=out:csv`;
      
      console.log('Fetching from URL:', csvUrl);
      
      const response = await fetch(csvUrl, {
        method: 'GET',
        mode: 'cors',
        headers: {
          'Accept': 'text/csv',
        },
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}: ${response.statusText}`);
      }
      
      const csvText = await response.text();
      
      if (!csvText || csvText.trim() === '') {
        throw new Error("Sheet returned empty data");
      }
      
      console.log('CSV Data received, length:', csvText.length);
      console.log('First 200 chars:', csvText.substring(0, 200));
      
      // Parse CSV using Papa Parse
      Papa.parse(csvText, {
        header: false,
        skipEmptyLines: true,
        complete: (results) => {
          const rows = results.data;
          console.log('Parsed rows:', rows.length);
          
          if (rows.length < 2) {
            setSpreadsheetError("No data rows found in sheet");
            setCompanyData([]);
            setIsLoadingSpreadsheet(false);
            return;
          }
          
          // Process data rows (skip header row if it exists)
          const companies = [];
          
          for (let i = 1; i < rows.length; i++) {
            const row = rows[i];
            
            // Make sure we have enough columns
            if (row.length > COMPANY_NAME_COL) {
              const companyName = row[COMPANY_NAME_COL]?.trim() || '';
              
              if (companyName !== '') {
                companies.push({
                  companyName: companyName,
                  gstNo: row[GST_COL]?.trim() || '',
                  shippingAddress: row[SHIPPING_COL]?.trim() || '',
                  billingAddress: row[BILLING_COL]?.trim() || ''
                });
              }
            }
          }
          
          setCompanyData(companies);
          
          if (companies.length === 0) {
            setSpreadsheetError('No valid company data found. Make sure column B has company names.');
          } else {
            console.log(`Successfully loaded ${companies.length} companies`);
          }
        },
        error: (error) => {
          throw new Error(`Failed to parse CSV: ${error.message}`);
        }
      });
      
    } catch (error) {
      console.error("Error loading sheet:", error);
      
      // Provide more helpful error messages
      if (error.message.includes('404')) {
        setSpreadsheetError(
          "Sheet not found (404). Please check:\n" +
          "1. The spreadsheet ID is correct\n" +
          "2. The sheet is shared with 'Anyone with the link'\n" +
          "3. Try opening the sheet in an incognito window to verify access"
        );
      } else if (error.message.includes('Failed to fetch')) {
        setSpreadsheetError(
          "Network error. Try:\n" +
          "1. Make sure you're connected to the internet\n" +
          "2. Check if your browser is blocking CORS\n" +
          "3. Try using a different browser"
        );
      } else {
        setSpreadsheetError(error.message || "Failed to load sheet");
      }
    } finally {
      setIsLoadingSpreadsheet(false);
    }
  };

  // Handle retry
  const handleRetry = () => {
    setRetryCount(prev => prev + 1);
  };

  // Test the sheet access
  const testSheetAccess = () => {
    window.open(`https://docs.google.com/spreadsheets/d/${SPREADSHEET_ID}/edit?usp=sharing`, '_blank');
  };

  // Handle company selection from dropdown
  const handleCompanySelect = (e) => {
    const selectedCompanyName = e.target.value;
    setCompanyName(selectedCompanyName);
    
    if (selectedCompanyName) {
      const selectedCompany = companyData.find(c => c.companyName === selectedCompanyName);
      if (selectedCompany) {
        setGstNo(selectedCompany.gstNo || "");
        setShippingAddress(selectedCompany.shippingAddress || "");
        setBillingAddress(selectedCompany.billingAddress || "");
      }
    } else {
      // Clear all fields if "Select a company" is chosen
      setGstNo("");
      setShippingAddress("");
      setBillingAddress("");
    }
    
    // Pass to parent components
    if (onCompanyNameChange) {
      onCompanyNameChange(selectedCompanyName);
    }
  };

  // Handle QRN code change
  const handleQrnCodeChange = (e) => {
    const value = e.target.value;
    setQrnCode(value);
    if (onQrnCodeChange) {
      onQrnCodeChange(value);
    }
  };

  // Handle packing option change
  const handlePackingChange = (e) => {
    const selectedValue = e.target.value;
    setPackingOption(selectedValue);
    if (onPackingOptionChange) {
      onPackingOptionChange(selectedValue);
    }
  };

  // Handle discount change
  const handleDiscountChange = (e) => {
    const value = e.target.value;
    setDiscount(value);
    if (onDiscountChange) {
      onDiscountChange(value);
    }
  };

  // Handle painting cost change
  const handlePaintingCostChange = (e) => {
    const value = e.target.value;
    setPaintingCost(value);
    if (onPaintingCostChange) {
      onPaintingCostChange(value);
    }
  };

  const handleFileChange = async (e) => {
    const files = Array.from(e.target.files);
    if (!files.length) return;

    const isTooBig = files.some((file) => file.size > MAX_SIZE_MB * 1024 * 1024);
    if (isTooBig) {
      onError(`One or more files exceed the ${MAX_SIZE_MB}MB limit.`);
      return;
    }

    setUploading(true);
    setProgress(0);
    setStatusLabel("Uploading...");
    onLoading(true);

    const formData1 = new FormData();
    const formData2 = new FormData();
    
    // Append company info to form data if available
    if (companyName) {
      formData1.append("company_name", companyName);
      formData2.append("company_name", companyName);
    }
    if (qrnCode) {
      formData1.append("qrn_code", qrnCode);
      formData2.append("qrn_code", qrnCode);
    }
    if (gstNo) {
      formData1.append("gst_no", gstNo);
      formData2.append("gst_no", gstNo);
    }
    if (shippingAddress) {
      formData1.append("shipping_address", shippingAddress);
      formData2.append("shipping_address", shippingAddress);
    }
    if (billingAddress) {
      formData1.append("billing_address", billingAddress);
      formData2.append("billing_address", billingAddress);
    }
    
    // Append packing info to form data if available
    if (packingOption) {
      formData1.append("packing_option", packingOption);
      formData2.append("packing_option", packingOption);
    }
    
    // Append discount to form data if available
    if (discount) {
      formData1.append("discount", discount);
      formData2.append("discount", discount);
    }
    
    // Append painting cost to form data if available
    if (paintingCost) {
      formData1.append("painting_cost", paintingCost);
      formData2.append("painting_cost", paintingCost);
    }
    
    files.forEach((file) => {
      formData1.append("files", file);
      formData2.append("files", file);
    });

    try {
      // Upload with real-time progress
      const autoOrientRes = await uploadWithProgress(
        "http://localhost:8000/auto-orient/",
        formData1,
        (percent) => setProgress(percent)
      );

      if (!autoOrientRes || autoOrientRes.error) {
        throw new Error("Auto-orient failed.");
      }

      setProgress(100);
      setStatusLabel("Processing...");

      const ccRes = await fetch("http://localhost:8000/upload-multiple-stl/", {
        method: "POST",
        body: formData2,
      });

      if (!ccRes.ok) {
        const err = await ccRes.text();
        throw new Error("Volume endpoint failed: " + err);
      }

      const ccData = await ccRes.json();

      const merged = files.map((file) => {
        const orient = autoOrientRes.find((item) => item.filename === file.name);
        const cc = ccData.processed.find((item) => item.filename === file.name);
        return {
          file,
          fileName: file.name,
          dimensions: orient
            ? {
                dimensions_before: orient.dimensions_before,
                dimensions_after: orient.dimensions_after,
              }
            : null,
          volume_cc: cc ? cc.volume_cm3 : null,
          viewOrientation: "before",
          multiplier: "",
          multipliedWeight: null,
          error: orient?.error || cc?.error || null,
          geometry: null,
          quantity: 1,
          // Add company information to each file object - THIS IS THE KEY FIX
          companyName: companyName,
          qrnCode: qrnCode,
          gstNo: gstNo,
          shippingAddress: shippingAddress,
          billingAddress: billingAddress
        };
      });

      setFileStates(merged);
      if (onFileStatesChange) {
        onFileStatesChange(merged);
      }
      
      setSelectedIndex(0);
      onError(null);

      files.forEach((file) => {
        if (file.name.toLowerCase().endsWith(".stl")) {
          const reader = new FileReader();
          reader.onload = (event) => {
            try {
              const geometry = new STLLoader().parse(event.target.result);
              onFileLoad(geometry);
            } catch (err) {
              onError("Failed to load STL geometry.");
            }
          };
          reader.readAsArrayBuffer(file);
        }
      });

      setStatusLabel("Done");
    } catch (error) {
      onError(error.message || "Something went wrong.");
      setStatusLabel("Error");
    } finally {
      setUploading(false);
      onLoading(false);
      setTimeout(() => {
        setProgress(0);
        setStatusLabel("");
      }, 2000);
    }
  };

  const uploadWithProgress = (url, formData, onProgress) => {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", url);

      xhr.upload.addEventListener("progress", (e) => {
        if (e.lengthComputable) {
          const percent = Math.round((e.loaded / e.total) * 100);
          onProgress(percent);
        }
      });

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText));
          } catch (e) {
            resolve([]);
          }
        } else {
          reject(new Error("Upload failed: " + xhr.statusText));
        }
      };

      xhr.onerror = () => reject(new Error("Upload error."));
      xhr.send(formData);
    });
  };

  const getCurrentDimensions = (fileState) => {
    if (!fileState.dimensions) return null;
    if (liveMode && liveDimensions) return liveDimensions;
    return fileState.viewOrientation === "before"
      ? fileState.dimensions.dimensions_before
      : reorientedDimensions || fileState.dimensions.dimensions_after;
  };

  const calculateWeight = (fileState) => {
    if (fileState.volume_cc) return Math.round(fileState.volume_cc * density);
    return 0;
  };

  const handleMultiplierChange = (e, idx) => {
    const value = e.target.value;
    setFileStates((prev) => {
      const updated = prev.map((state, i) =>
        i === idx
          ? {
              ...state,
              multiplier: value,
              multipliedWeight:
                !isNaN(value) && value !== "" && state.volume_cc
                  ? (Number(value) * calculateWeight(state)).toFixed(2)
                  : null,
            }
          : state
      );
      
      if (onFileStatesChange) {
        onFileStatesChange(updated);
      }
      
      return updated;
    });
  };

  const toggleOrientation = (idx) => {
    const newOrientationIsBefore = fileStates[idx]?.viewOrientation === "after";
    setFileStates((prev) => {
      const updated = prev.map((state, i) =>
        i === idx
          ? {
              ...state,
              viewOrientation: state.viewOrientation === "before" ? "after" : "before",
            }
          : state
      );
      
      if (onFileStatesChange) {
        onFileStatesChange(updated);
      }
      
      return updated;
    });
    setLiveMode(newOrientationIsBefore);
  };

  const handleSwipeLeft = () => {
    if (selectedIndex < fileStates.length - 1) setSelectedIndex(selectedIndex + 1);
  };

  const handleSwipeRight = () => {
    if (selectedIndex > 0) setSelectedIndex(selectedIndex - 1);
  };

  const fileState = fileStates[selectedIndex];
  const dims = fileState && getCurrentDimensions(fileState);

  return (
    <div className="w-full">
      {/* Company Information Section */}
      <div className="w-full max-w-md mx-auto mb-4 p-4 bg-gray-800 rounded-lg">
        <h3 className="text-white text-lg font-semibold mb-3">Company Information</h3>
        
        {/* Loading indicator */}
        {isLoadingSpreadsheet && (
          <div className="mb-3 text-cyan-400 text-sm flex items-center">
            <svg className="animate-spin h-4 w-4 mr-2" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            Loading company data from Google Sheet...
          </div>
        )}
        
        {/* Error message if sheet fails to load */}
        {spreadsheetError && (
          <div className="mb-3 text-red-400 text-sm p-2 bg-red-900/20 rounded whitespace-pre-line">
            ⚠️ {spreadsheetError}
            
            <div className="flex gap-2 mt-2">
              <button 
                onClick={handleRetry}
                className="px-3 py-1 bg-cyan-500 text-white text-xs rounded hover:bg-cyan-600"
              >
                Retry
              </button>
              <button 
                onClick={testSheetAccess}
                className="px-3 py-1 bg-gray-600 text-white text-xs rounded hover:bg-gray-700"
              >
                Open Sheet
              </button>
            </div>
          </div>
        )}
        
        {/* Company Name Dropdown - Only show if sheet data is loaded */}
        {!isLoadingSpreadsheet && companyData.length > 0 && (
          <div className="mb-3">
            <label htmlFor="company-select" className="block text-white text-sm mb-1">
              Select Company
            </label>
            <select
              id="company-select"
              onChange={handleCompanySelect}
              value={companyName}
              className="w-full p-2 border border-cyan-500 rounded-lg focus:ring-2 focus:ring-cyan-400 text-white bg-gray-700"
            >
              <option value="">Select a company</option>
              {companyData.map((company, index) => (
                <option key={index} value={company.companyName}>
                  {company.companyName}
                </option>
              ))}
            </select>
            <p className="text-gray-400 text-xs mt-1">
              Loaded {companyData.length} companies
            </p>
          </div>
        )}
        
        {/* Display message if no companies found */}
        {!isLoadingSpreadsheet && companyData.length === 0 && !spreadsheetError && (
          <div className="mb-3 text-yellow-400 text-sm">
            No companies found in the spreadsheet.
          </div>
        )}
        
        {/* Display GST and Addresses - only show if a company is selected */}
        {companyName && (
          <div className="space-y-3 mt-4">
            <div>
              <label htmlFor="gst-no" className="block text-white text-sm mb-1">
                GST Number
              </label>
              <input
                type="text"
                id="gst-no"
                value={gstNo}
                onChange={(e) => setGstNo(e.target.value)}
                placeholder="GST Number"
                className="w-full p-2 border border-cyan-500 rounded-lg focus:ring-2 focus:ring-cyan-400 text-white bg-gray-700"
                readOnly={true}
              />
            </div>
            
            <div>
              <label htmlFor="shipping-address" className="block text-white text-sm mb-1">
                Shipping Address
              </label>
              <textarea
                id="shipping-address"
                value={shippingAddress}
                onChange={(e) => setShippingAddress(e.target.value)}
                placeholder="Shipping Address"
                rows="2"
                className="w-full p-2 border border-cyan-500 rounded-lg focus:ring-2 focus:ring-cyan-400 text-white bg-gray-700"
                readOnly={true}
              />
            </div>
            
            <div>
              <label htmlFor="billing-address" className="block text-white text-sm mb-1">
                Billing Address
              </label>
              <textarea
                id="billing-address"
                value={billingAddress}
                onChange={(e) => setBillingAddress(e.target.value)}
                placeholder="Billing Address"
                rows="2"
                className="w-full p-2 border border-cyan-500 rounded-lg focus:ring-2 focus:ring-cyan-400 text-white bg-gray-700"
                readOnly={true}
              />
            </div>
          </div>
        )}
        
        {/* QRN Code Input */}
        <div className="mt-4">
          <label htmlFor="qrn-code" className="block text-white text-sm mb-1">
            QRN Code
          </label>
          <input
            type="text"
            id="qrn-code"
            value={qrnCode}
            onChange={handleQrnCodeChange}
            placeholder="Enter QRN code"
            className="w-full p-2 border border-cyan-500 rounded-lg focus:ring-2 focus:ring-cyan-400 text-white bg-gray-700"
          />
        </div>
      </div>

      {/* Packing and Forwarding Section */}
      <div className="w-full max-w-md mx-auto mb-4 p-4 bg-gray-800 rounded-lg">
        <h3 className="text-white text-lg font-semibold mb-3">Packing & Forwarding</h3>
        
        <div className="mb-3">
          <select
            id="packing-option"
            value={packingOption}
            onChange={handlePackingChange}
            className="w-full p-2 border border-cyan-500 rounded-lg focus:ring-2 focus:ring-cyan-400 text-white bg-gray-700"
          >
            <option value="" disabled>Select an option</option>
            {packingOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Main File Upload Button */}
      <div className="flex justify-center">
        <label
          htmlFor="upload-file"
          className="border-1 border-gray-200 w-40 text-center rounded-[50px] p-2 px-4 text-white cursor-pointer hover:bg-cyan-700 text-xl font-bold hover:text-zinc-50 bg-cyan-500 shadow-lg shadow-cyan-500/50 transition"
        >
          Upload Files
        </label>
        <input
          className="hidden"
          type="file"
          accept=".stl,.step,.stp"
          multiple
          onChange={handleFileChange}
          id="upload-file"
        />
      </div>

      {/* Upload Progress UI */}
      {uploading && (
        <div className="w-full max-w-md mx-auto mt-4">
          <div className="w-full bg-gray-200 rounded-full h-5 overflow-hidden">
            <div
              className="bg-cyan-500 h-full transition-all duration-300 ease-in-out"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-center mt-2 text-cyan-700 font-semibold">
            {statusLabel} ({progress}%)
          </p>
        </div>
      )}

      {/* Model Dimensions Display */}
      {fileStates.length > 0 && !uploading && (
        <div className="flex flex-col items-center mt-4 w-full">
          <Swipe
            onSwipeLeft={handleSwipeLeft}
            onSwipeRight={handleSwipeRight}
            allowMouseEvents={true}
            style={{ touchAction: "pan-y" }}
          >
            <div
              className="bg-white p-4 rounded-3xl shadow min-w-[310px] mx-2"
              style={{ userSelect: "none" }}
            >
              <h3 className="text-lg font-semibold mb-2">Model Dimensions</h3>
              <h3 className="text-lg font-semibold mb-2">
                Uploaded File: {fileState.fileName}
              </h3>
              {dims ? (
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-medium">
                      {fileState.viewOrientation === "before" ? "Before" : "After"} Orientation:
                    </h4>
                    <p>X: {dims.x?.toFixed(2) ?? "N/A"} mm</p>
                    <p>Y: {dims.y?.toFixed(2) ?? "N/A"} mm</p>
                    <p>Z: {dims.z?.toFixed(2) ?? "N/A"} mm</p>
                  </div>
                  <div>
                    {fileState.volume_cc !== null && fileState.volume_cc !== undefined && (
                      <p className="mt-2 text-sm text-gray-700">
                        Volume(cc):{" "}
                        <span className="font-bold">{fileState.volume_cc.toFixed(3)} cm³</span>
                      </p>
                    )}
                  </div>
                </div>
              ) : (
                <div className="text-red-500">{fileState.error || "Loading..."}</div>
              )}
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  toggleOrientation(selectedIndex);
                }}
                className="mt-4 px-6 py-2 bg-cyan-500 text-white font-semibold rounded-lg hover:bg-cyan-700 transition"
              >
                Orientation
              </button>
            </div>
          </Swipe>

          <div className="text-sm text-gray-500 mt-2">
            {selectedIndex + 1} / {fileStates.length}
          </div>
        </div>
      )}

      {/* Painting Cost Section */}
      <div className="w-full max-w-md mx-auto mb-4 p-4 bg-gray-800 rounded-lg">
        <h3 className="text-white text-lg font-semibold mb-3">Painting Cost</h3>
        
        <div className="mb-3">
          <label htmlFor="painting-cost" className="block text-white text-sm mb-1">
            Enter painting cost (₹)
          </label>
          <input
            type="number"
            id="painting-cost"
            value={paintingCost}
            onChange={handlePaintingCostChange}
            placeholder="Enter painting cost"
            min="0"
            step="0.01"
            className="w-full p-2 border border-cyan-500 rounded-lg focus:ring-2 focus:ring-cyan-400 text-white bg-gray-700"
          />
        </div>

        {/* Display painting cost if entered */}
        {paintingCost && (
          <div className="mt-3 p-2 bg-gray-700 rounded">
            <p className="text-white text-sm">
              <span className="text-cyan-400">Painting Cost:</span> ₹{parseFloat(paintingCost).toFixed(2)}
            </p>
          </div>
        )}
      </div>

      {/* Discount Section */}
      <div className="w-full max-w-md mx-auto mb-4 p-4 bg-gray-800 rounded-lg">
        <h3 className="text-white text-lg font-semibold mb-3">Discount</h3>
        
        <div className="mb-3">
          <label htmlFor="discount" className="block text-white text-sm mb-1">
            Enter discount percentage
          </label>
          <input
            type="number"
            id="discount"
            value={discount}
            onChange={handleDiscountChange}
            placeholder="Enter discount percentage"
            min="0"
            max="100"
            step="0.1"
            className="w-full p-2 border border-cyan-500 rounded-lg focus:ring-2 focus:ring-cyan-400 text-white bg-gray-700"
          />
        </div>

        {/* Display discount if entered */}
        {discount && (
          <div className="mt-3 p-2 bg-gray-700 rounded">
            <p className="text-white text-sm">
              <span className="text-cyan-400">Discount:</span> {discount}%
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default FileUpload;