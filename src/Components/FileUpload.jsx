// import React, { useState } from "react";
// import { STLLoader } from "three-stdlib";
// import Swipe from "react-easy-swipe";
// import SLVPanel from "./SLVPanel";
// import VCComponent from "./VCComponent";

// const density = 1.05; // g/cm³
// const MAX_SIZE_MB = 300;

// const FileUpload = ({
//   onFileLoad,
//   onError,
//   onLoading,
//   reorientedDimensions,
//   liveDimensions,
//   liveMode,
//   setLiveMode,
// }) => {
//   const [fileStates, setFileStates] = useState([]);
//   const [selectedIndex, setSelectedIndex] = useState(0);
//   const [excelData, setExcelData] = useState([]);
//   const [globalCost, setGlobalCost] = useState("");
//   const [uploading, setUploading] = useState(false); // 🆕 Uploading state

//   const handleFileChange = async (e) => {
//     const files = Array.from(e.target.files);
//     if (!files.length) return;
  
//     // Optional: Remove or adjust size check if backend can handle large files
//     const isTooBig = files.some(file => file.size > MAX_SIZE_MB * 1024 * 1024);
//     if (isTooBig) {
//       onError(`One or more files exceed the ${MAX_SIZE_MB}MB limit.`);
//       return;
//     }
  
//     setUploading(true);
//     onLoading(true);
  
//     // Clone FormData for each request (IMPORTANT!)
//     const formData1 = new FormData();
//     const formData2 = new FormData();
//     files.forEach((file) => {
//       formData1.append("files", file);
//       formData2.append("files", file);
//     });
  
//     try {
//       // Sequential fetch (or use await Promise.all with different formData objects)
//       const autoOrientRes = await fetch("http://localhost:8000/auto-orient/", {
//         method: "POST",
//         body: formData1,
//       });
  
//       if (!autoOrientRes.ok) {
//         const err = await autoOrientRes.text();
//         throw new Error("Auto-orient failed: " + err);
//       }
  
//       const ccRes = await fetch("http://localhost:8000/upload-multiple-stl/", {
//         method: "POST",
//         body: formData2,
//       });
  
//       if (!ccRes.ok) {
//         const err = await ccRes.text();
//         throw new Error("Volume endpoint failed: " + err);
//       }
  
//       const autoOrientData = await autoOrientRes.json();
//       const ccData = await ccRes.json();
  
//       const merged = files.map((file) => {
//         const orient = autoOrientData.find((item) => item.filename === file.name);
//         const cc = ccData.processed.find((item) => item.filename === file.name);
//         return {
//           file,
//           fileName: file.name,
//           dimensions: orient
//             ? {
//                 dimensions_before: orient.dimensions_before,
//                 dimensions_after: orient.dimensions_after,
//               }
//             : null,
//           volume_cc: cc ? cc.volume_cc : null,
//           viewOrientation: "before",
//           multiplier: "",
//           multipliedWeight: null,
//           error: orient?.error || cc?.error || null,
//           geometry: null,
//           quantity: 1,
//         };
//       });
  
//       setFileStates(merged);
//       setSelectedIndex(0);
//       onError(null);
  
//       files.forEach((file) => {
//         if (file.name.toLowerCase().endsWith(".stl")) {
//           const reader = new FileReader();
//           reader.onload = (event) => {
//             try {
//               const geometry = new STLLoader().parse(event.target.result);
//               onFileLoad(geometry);
//             } catch (err) {
//               onError("Failed to load STL geometry.");
//             }
//           };
//           reader.readAsArrayBuffer(file);
//         }
//       });
//     } catch (error) {
//       onError(error.message || "Something went wrong.");
//     } finally {
//       setUploading(false);
//       onLoading(false);
//     }
//   };
  

//   const getCurrentDimensions = (fileState) => {
//     if (!fileState.dimensions) return null;
//     if (liveMode && liveDimensions) return liveDimensions;
//     return fileState.viewOrientation === "before"
//       ? fileState.dimensions.dimensions_before
//       : reorientedDimensions || fileState.dimensions.dimensions_after;
//   };

//   const calculateWeight = (fileState) => {
//     if (fileState.volume_cc) return Math.round(fileState.volume_cc * density);
//     return 0;
//   };

//   const handleMultiplierChange = (e, idx) => {
//     const value = e.target.value;
//     setFileStates((prev) =>
//       prev.map((state, i) =>
//         i === idx
//           ? {
//               ...state,
//               multiplier: value,
//               multipliedWeight:
//                 !isNaN(value) && value !== "" && state.volume_cc
//                   ? (Number(value) * calculateWeight(state)).toFixed(2)
//                   : null,
//             }
//           : state
//       )
//     );
//   };

//   const toggleOrientation = (idx) => {
//     const newOrientationIsBefore = fileStates[idx]?.viewOrientation === "after";
//     setFileStates((prev) =>
//       prev.map((state, i) =>
//         i === idx
//           ? {
//               ...state,
//               viewOrientation: state.viewOrientation === "before" ? "after" : "before",
//             }
//           : state
//       )
//     );
//     setLiveMode(newOrientationIsBefore);
//   };

//   const handleSwipeLeft = () => {
//     if (selectedIndex < fileStates.length - 1) setSelectedIndex(selectedIndex + 1);
//   };

//   const handleSwipeRight = () => {
//     if (selectedIndex > 0) setSelectedIndex(selectedIndex - 1);
//   };

//   const fileState = fileStates[selectedIndex];
//   const dims = fileState && getCurrentDimensions(fileState);

//   return (
//     <>
//       <label
//         htmlFor="upload-file"
//         className="border-1 border-gray-200 w-40 text-center rounded-[50px] p-2 px-4 text-white m-2 cursor-pointer hover:bg-cyan-700 text-xl font-bold hover:text-zinc-50 bg-cyan-500 shadow-lg shadow-cyan-500/50 transition"
//       >
//         Upload Files
//       </label>
//       <input
//         className="hidden"
//         type="file"
//         accept=".stl,.step,.stp"
//         multiple
//         onChange={handleFileChange}
//         id="upload-file"
//       />

//       {uploading && (
//         <div className="text-cyan-700 font-semibold text-lg mt-4 animate-pulse">
//           Uploading and processing files...
//         </div>
//       )}

//       {fileStates.length > 0 && !uploading && (
//         <div className="flex flex-col items-center mt-4">
//           <Swipe
//             onSwipeLeft={handleSwipeLeft}
//             onSwipeRight={handleSwipeRight}
//             allowMouseEvents={true}
//             style={{ touchAction: "pan-y" }}
//           >
//             <div
//               className="bg-white p-4 rounded-3xl shadow min-w-[310px] mx-2"
//               style={{ userSelect: "none" }}
//             >
//               <h3 className="text-lg font-semibold mb-2">Model Dimensions</h3>
//               <h3 className="text-lg font-semibold mb-2">
//                 Uploaded File: {fileState.fileName}
//               </h3>
//               {dims ? (
//                 <div className="grid grid-cols-2 gap-4">
//                   <div>
//                     <h4 className="font-medium">
//                       {fileState.viewOrientation === "before" ? "Before" : "After"} Orientation:
//                     </h4>
//                     <p>X: {dims.x.toFixed(2)} mm</p>
//                     <p>Y: {dims.y.toFixed(2)} mm</p>
//                     <p>Z: {dims.z.toFixed(2)} mm</p>
//                   </div>
//                   <div>
//                     {fileState.volume_cc !== null && (
//                       <p className="mt-2 text-sm text-gray-700">
//                         Volume(cc):{" "}
//                         <span className="font-bold">{fileState.volume_cc.toFixed(3)} cm³</span>
//                       </p>
//                     )}
//                   </div>
//                 </div>
//               ) : (
//                 <div className="text-red-500">{fileState.error || "Loading..."}</div>
//               )}
//               <button
//                 onClick={(e) => {
//                   e.stopPropagation();
//                   toggleOrientation(selectedIndex);
//                 }}
//                 className="mt-4 px-6 py-2 bg-cyan-500 text-white font-semibold rounded-lg hover:bg-cyan-700 transition"
//               >
//                 Orientation
//               </button>
//             </div>
//           </Swipe>

//           <div className="text-sm text-gray-500 mt-2">
//             {selectedIndex + 1} / {fileStates.length}
//           </div>

//           {/* SLV Panel & Volume/Cost Components */}
//           <SLVPanel 
//             fileStates={fileStates}
//             setFileStates={setFileStates}
//             excelData={excelData}
//             setExcelData={setExcelData}
//             globalCost={globalCost}
//             setGlobalCost={setGlobalCost}
//           /> 
//           <VCComponent fileStates={fileStates} />
//         </div>
//       )}
//     </>
//   );
// };

// export default FileUpload;


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
}) => {
  const [fileStates, setFileStates] = useState([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [excelData, setExcelData] = useState([]);
  const [globalCost, setGlobalCost] = useState("");
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusLabel, setStatusLabel] = useState("");

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
    volume_cc: cc ? cc.volume_cm3 : null, // ✅ Use volume_cm3 from backend
    viewOrientation: "before",
    multiplier: "",
    multipliedWeight: null,
    error: orient?.error || cc?.error || null,
    geometry: null,
    quantity: 1,
  };
});

      setFileStates(merged);
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
    setFileStates((prev) =>
      prev.map((state, i) =>
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
      )
    );
  };

  const toggleOrientation = (idx) => {
    const newOrientationIsBefore = fileStates[idx]?.viewOrientation === "after";
    setFileStates((prev) =>
      prev.map((state, i) =>
        i === idx
          ? {
              ...state,
              viewOrientation: state.viewOrientation === "before" ? "after" : "before",
            }
          : state
      )
    );
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
    <>
      <label
        htmlFor="upload-file"
        className="border-1 border-gray-200 w-40 text-center rounded-[50px] p-2 px-4 text-white m-2 cursor-pointer hover:bg-cyan-700 text-xl font-bold hover:text-zinc-50 bg-cyan-500 shadow-lg shadow-cyan-500/50 transition"
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

      {/* 🔄 Upload Progress UI */}
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

      {fileStates.length > 0 && !uploading && (
        <div className="flex flex-col items-center mt-4">
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

          <SLVPanel
            fileStates={fileStates}
            setFileStates={setFileStates}
            excelData={excelData}
            setExcelData={setExcelData}
            globalCost={globalCost}
            setGlobalCost={setGlobalCost}
          />
          <VCComponent fileStates={fileStates} />
        </div>
      )}
    </>
  );
};

export default FileUpload;
