// FileUpload.jsx
import React, { useState } from "react";
import { STLLoader } from "three-stdlib";
import Swipe from "react-easy-swipe";
import SLVPanel from "./SLVPanel";
import VCComponent from "./VCComponent";

const density = 1.05; // g/cm³
const MAX_SIZE_MB = 300;

const FileUpload = ({
  onFileLoad,
  onError,
  onLoading,
  reorientedDimensions,
  liveDimensions,
  liveMode,
  setLiveMode,
  onFileStatesChange, // Add this new prop
}) => {
  const [fileStates, setFileStates] = useState([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [excelData, setExcelData] = useState([]);
  const [globalCost, setGlobalCost] = useState("");
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusLabel, setStatusLabel] = useState("");
  
  // State for company info
  const [companyName, setCompanyName] = useState("");
  const [qrnCode, setQrnCode] = useState("");
  
  // State for packing and forwarding
  const [packingOption, setPackingOption] = useState("");
  
  // State for discount
  const [discount, setDiscount] = useState("");

  // Packing options
  const packingOptions = [
    { value: "courier", label: "Courier" },
    { value: "porter", label: "Porter" },
    { value: "outstation", label: "Outstation" },
  ];

  const handlePackingChange = (e) => {
    const selectedValue = e.target.value;
    setPackingOption(selectedValue);
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
        };
      });

      setFileStates(merged);
      // Send fileStates to parent component (Hero)
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
      
      // Send updated fileStates to parent component
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
      
      // Send updated fileStates to parent component
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
        
        {/* Manual Input Fields */}
        <div className="mb-3">
          <label htmlFor="company-name" className="block text-white text-sm mb-1">
            Company Name
          </label>
          <input
            type="text"
            id="company-name"
            value={companyName}
            onChange={(e) => setCompanyName(e.target.value)}
            placeholder="Enter company name"
            className="w-full p-2 border border-cyan-500 rounded-lg focus:ring-2 focus:ring-cyan-400 text-white bg-gray-700"
          />
        </div>
        
        <div className="mb-4">
          <label htmlFor="qrn-code" className="block text-white text-sm mb-1">
            QRN Code
          </label>
          <input
            type="text"
            id="qrn-code"
            value={qrnCode}
            onChange={(e) => setQrnCode(e.target.value)}
            placeholder="Enter QRN code"
            className="w-full p-2 border border-cyan-500 rounded-lg focus:ring-2 focus:ring-cyan-400 text-white bg-gray-700"
          />
        </div>

        {/* Display entered info */}
        {(companyName || qrnCode) && (
          <div className="mt-3 p-2 bg-gray-700 rounded">
            {companyName && (
              <p className="text-white text-sm">
                <span className="text-cyan-400">Company:</span> {companyName}
              </p>
            )}
            {qrnCode && (
              <p className="text-white text-sm">
                <span className="text-cyan-400">QRN:</span> {qrnCode}
              </p>
            )}
          </div>
        )}
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
            onChange={(e) => setDiscount(e.target.value)}
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

      {/* Model Dimensions Display - Only show when file is uploaded */}
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

      {/* SLVPanel and VCComponent - Only show when files are uploaded */}
      {fileStates.length > 0 && !uploading && (
        <>
          <SLVPanel
            fileStates={fileStates}
            setFileStates={setFileStates}
            excelData={excelData}
            setExcelData={setExcelData}
            globalCost={globalCost}
            setGlobalCost={setGlobalCost}
            companyName={companyName}
            qrnCode={qrnCode}
            packingOption={packingOption}
            discount={discount}
          />
          <VCComponent fileStates={fileStates} />
        </>
      )}
    </div>
  );
};

export default FileUpload;