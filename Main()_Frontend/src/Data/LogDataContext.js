import React, { createContext, useState, useEffect } from "react";

// 컨텍스트 생성 시 기본값 설정
const LogDataContext = createContext({
  logInfo: [], // 초기 로그 정보는 빈 배열
  updateLogInfo: () => {}, // 기본적으로 아무 동작도 하지 않는 함수
  clearLogInfo: () => {}, // 기본적으로 아무 동작도 하지 않는 함수
});

const LogDataProvider = ({ children }) => {
  const [logInfo, setLogInfo] = useState(() => {
    const savedLogInfo = localStorage.getItem("logInfo");
    return savedLogInfo ? JSON.parse(savedLogInfo) : [];
  });

  useEffect(() => {
    // console.log("LogDataContext에 저장된 데이터:", logInfo);
  }, [logInfo]);

  useEffect(() => {
    localStorage.setItem("logInfo", JSON.stringify(logInfo));
  }, [logInfo]);

  const updateLogInfo = (newLogInfo) => {
    setLogInfo((prevLogInfo) => [...prevLogInfo, newLogInfo]);
  };

  const clearLogInfo = () => {
    setLogInfo([]);
  };

  return (
    <LogDataContext.Provider value={{ logInfo, updateLogInfo, clearLogInfo }}>
      {children}
    </LogDataContext.Provider>
  );
};

export { LogDataContext, LogDataProvider };
