import React, { createContext, useState, useEffect } from "react";

const ProgressBarDataContext = createContext();

const ProgressBarDataProvider = ({ children }) => {
  const [ProgressBarInfo, setProgressBarInfo] = useState(() => {
    const savedProgressBarInfo = localStorage.getItem("total_length");
    return savedProgressBarInfo ? JSON.parse(savedProgressBarInfo) : 0; // 초기값을 0으로 설정
  });

  useEffect(() => {
    //console.log('ProgressBarDataContext에 저장된 데이터:', ProgressBarInfo);
  }, [ProgressBarInfo]);

  useEffect(() => {
    localStorage.setItem("total_length", JSON.stringify(ProgressBarInfo));
  }, [ProgressBarInfo]);

  const updateProgressBarInfo = (newProgressBarInfo) => {
    setProgressBarInfo(
      (prevProgressBarInfo) => prevProgressBarInfo + newProgressBarInfo
    ); // 이전 값에 새 값을 더함
  };

  const clearProgressBarInfo = () => {
    setProgressBarInfo(0); // 값을 0으로 초기화
  };

  return (
    <ProgressBarDataContext.Provider
      value={{
        ProgressBarInfo,
        updateProgressBarInfo,
        clearProgressBarInfo,
        setProgressBarInfo,
      }}
    >
      {children}
    </ProgressBarDataContext.Provider>
  );
};

export { ProgressBarDataContext, ProgressBarDataProvider };
