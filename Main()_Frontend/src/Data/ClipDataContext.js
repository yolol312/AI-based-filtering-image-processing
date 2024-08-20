// ClipDataContext.js
import React, { createContext, useState, useEffect } from "react";

const ClipDataContext = createContext();

const ClipDataProvider = ({ children }) => {
  const [clipInfo, setClipInfo] = useState(() => {
    const savedClipInfo = localStorage.getItem("clipInfo");
    return savedClipInfo ? JSON.parse(savedClipInfo) : [];
  });

  useEffect(() => {
    console.log("ClipDataContext에 저장된 데이터:", clipInfo);
  }, [clipInfo]);

  useEffect(() => {
    localStorage.setItem("clipInfo", JSON.stringify(clipInfo));
  }, [clipInfo]);

  const updateClipInfo = (newClipInfo) => {
    setClipInfo(Array.isArray(newClipInfo) ? newClipInfo : []);
  };

  const clearClipInfo = () => {
    setClipInfo([]);
  };

  return (
    <ClipDataContext.Provider
      value={{ clipInfo, updateClipInfo, clearClipInfo, setClipInfo }}
    >
      {children}
    </ClipDataContext.Provider>
  );
};

export { ClipDataContext, ClipDataProvider };
