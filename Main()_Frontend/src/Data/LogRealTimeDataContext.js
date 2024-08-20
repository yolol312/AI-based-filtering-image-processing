import React, { createContext, useState, useEffect } from "react";

// 컨텍스트 생성 시 기본값 설정
const LogRealTimeDataContext = createContext({
  LogRealTimeInfo: [], // 초기 로그 정보는 빈 배열
  updateLogRealTimeInfo: () => {}, // 기본적으로 아무 동작도 하지 않는 함수
  clearLogRealTimeInfo: () => {}, // 기본적으로 아무 동작도 하지 않는 함수
});

const LogRealTimeDataProvider = ({ children }) => {
  const [LogRealTimeInfo, setLogRealTimeInfo] = useState(() => {
    const savedLogRealTimeInfo = localStorage.getItem("LogRealTimeInfo");
    return savedLogRealTimeInfo ? JSON.parse(savedLogRealTimeInfo) : [];
  });

  useEffect(() => {
    console.log("LogRealTimeDataContext에 저장된 데이터:", LogRealTimeInfo);
  }, [LogRealTimeInfo]);

  useEffect(() => {
    localStorage.setItem("LogRealTimeInfo", JSON.stringify(LogRealTimeInfo));
  }, [LogRealTimeInfo]);

  const updateLogRealTimeInfo = (newLogs) => {
    // newLogs가 배열인지 확인하고, 배열이 아닐 경우 배열로 만듭니다.
    const logsArray = Array.isArray(newLogs) ? newLogs : [newLogs];

    setLogRealTimeInfo((prevLogRealTimeInfo) => [
      ...prevLogRealTimeInfo,
      ...logsArray,
    ]);
  };

  const clearLogRealTimeInfo = () => {
    setLogRealTimeInfo([]);
  };

  return (
    <LogRealTimeDataContext.Provider
      value={{ LogRealTimeInfo, updateLogRealTimeInfo, clearLogRealTimeInfo }}
    >
      {children}
    </LogRealTimeDataContext.Provider>
  );
};

export { LogRealTimeDataContext, LogRealTimeDataProvider };
